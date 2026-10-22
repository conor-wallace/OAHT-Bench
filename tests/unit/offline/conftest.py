"""A fully-decodable synthetic dataset, for testing TAO/OMIS's data-sampling
pipeline against what it actually does, not just the shapes it returns.

Every scalar field (ego/teammate action, observation, reward) is an invertible
encoding of ``(agent, policy, episode, timestep, channel)``:

    value = agent*10000 + policy*1000 + episode*100 + timestep*10 + channel

so any value pulled out of a sampled batch decodes back to exactly which
agent/teammate/episode/timestep/field it came from. That turns "does
``build_d_epi``'s context really come from a different episode of the same
teammate" (Appendix C) or "does OMIS's D_step shift really read timestep
``t-1``, not ``t``" (Eq. 2) from something inferred by reading the sampling
code into something a test asserts directly on real decoded values.

Return-to-go is the one field this can't cover the same way:
:func:`~oaht_bench.dataset.windowing.window_slice` computes it as a
reverse-cumulative-sum of *rewards*, so ``ego_rtg`` at position ``t`` is a
sum of several different ``t'`` reward values, not a single invertible one.
Rewards themselves stay per-step decodable; anything that needs RTG asserts
against the known closed-form sum instead of decoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

CHANNEL_ACTION = 0
CHANNEL_OBS = 1
CHANNEL_REWARD = 2

#: Agent axis convention shared with every other module (dataset/schema.py):
#: 0 is the ego seat, 1 is the teammate seat.
EGO, MATE = 0, 1


def encode(agent: int, policy: int, episode: int, timestep: int, channel: int) -> int:
    for name, v in (
        ("agent", agent),
        ("policy", policy),
        ("episode", episode),
        ("timestep", timestep),
        ("channel", channel),
    ):
        if not (0 <= v <= 9):
            raise ValueError(f"{name}={v} must be a single digit (0-9) to stay decodable")
    return agent * 10000 + policy * 1000 + episode * 100 + timestep * 10 + channel


def decode(value) -> tuple[int, int, int, int, int]:
    """Inverse of :func:`encode`: ``(agent, policy, episode, timestep, channel)``."""
    v = int(round(float(value)))
    agent, v = divmod(v, 10000)
    policy, v = divmod(v, 1000)
    episode, v = divmod(v, 100)
    timestep, channel = divmod(v, 10)
    return agent, policy, episode, timestep, channel


def expected_rtg(policy: int, episode: int, timestep: int, horizon: int) -> float:
    """The closed-form return-to-go at ``timestep`` -- ``sum`` of the ego's own
    per-step reward encoding from ``timestep`` through ``horizon - 1``, matching
    ``return_to_go()``'s reverse-cumsum exactly. Use this instead of ``decode()``
    for anything reading ``ego_rtg``/``context_mate_rewards``-derived sums.
    """
    return float(
        sum(encode(EGO, policy, episode, t, CHANNEL_REWARD) for t in range(timestep, horizon))
    )


@dataclass
class DeterministicDataset:
    """A minimal stand-in for :class:`~oaht_bench.dataset.dataset.Dataset` --
    same ``batch``/``ego``/``mate``/``norm`` attributes, so a test calls
    :func:`~oaht_bench.dataset.windowing.window_slice`/``teammate_episode_ids``
    (or a baseline's own ``_sample_batch`` helpers) exactly as training code
    does, without a vault on disk to back a real ``Dataset``.
    """

    batch: object
    ego: int = EGO
    mate: int = MATE
    norm: object = None


def _deterministic_batch(teammate_sequence, *, T, obs_dim):
    """The ``EpisodeBatch`` the production windowing is built from."""
    from oaht_bench.dataset.schema import Episode, EpisodeBatch

    episodes = []
    member_rows = []
    for e, k in enumerate(teammate_sequence):
        ego_obs = np.array(
            [[encode(EGO, k, e, t, CHANNEL_OBS)] * obs_dim for t in range(T)], dtype=np.float32
        )
        mate_obs = np.array(
            [[encode(MATE, k, e, t, CHANNEL_OBS)] * obs_dim for t in range(T)],
            dtype=np.float32,
        )
        ego_act = np.array([encode(EGO, k, e, t, CHANNEL_ACTION) for t in range(T)])
        mate_act = np.array([encode(MATE, k, e, t, CHANNEL_ACTION) for t in range(T)])
        ego_rew = np.array(
            [float(encode(EGO, k, e, t, CHANNEL_REWARD)) for t in range(T)], dtype=np.float32
        )
        mate_rew = np.array(
            [float(encode(MATE, k, e, t, CHANNEL_REWARD)) for t in range(T)], dtype=np.float32
        )
        episodes.append(
            Episode(
                obs=np.stack([ego_obs, mate_obs]),
                actions=np.stack([ego_act, mate_act]),
                rewards=np.stack([ego_rew, mate_rew]),
                # Not identity-bearing -- kept trivial (all-legal) on purpose.
                avail_actions=np.ones((2, T, 2), dtype=np.float32),
                dones=np.zeros(T, dtype=bool),
            )
        )
        member_rows.append([EGO, k])
    return EpisodeBatch(episodes=episodes, member_ids=np.array(member_rows), ego_index=EGO, meta={})


@pytest.fixture
def deterministic_dataset():
    """Factory: ``build(teammate_sequence, *, T=6, obs_dim=1)``.

    ``teammate_sequence[e]`` is the teammate policy id for episode ``e`` --
    episode numbering is positional, matching ``member_ids``'s row order
    (the ``episode_id`` every ``window_slice``/``teammate_episode_ids`` call
    takes). Episodes are exactly ``T`` steps unless ``episode_lengths`` is
    given (to exercise padding), and every action/observation/reward is
    ``encode(agent, teammate_sequence[e], e, t, channel)``.

    Returns a :class:`DeterministicDataset`, not a production ``Dataset``
    (there's no vault on disk to back one) -- a test calls the same
    ``window_slice``/``teammate_episode_ids``/baseline ``_sample_batch``
    helpers training does, directly on its ``.batch``.
    """

    def build(teammate_sequence, *, T=6, obs_dim=1, episode_lengths=None):
        batch = _deterministic_batch(teammate_sequence, T=T, obs_dim=obs_dim)
        if episode_lengths is not None:
            # Truncate to exercise padding/masking -- the encoded values at the
            # remaining real positions are unchanged, only the episode is shorter.
            import dataclasses

            batch = dataclasses.replace(
                batch,
                episodes=[
                    dataclasses.replace(
                        e,
                        obs=e.obs[:, :n],
                        actions=e.actions[:, :n],
                        rewards=e.rewards[:, :n],
                        avail_actions=e.avail_actions[:, :n],
                        dones=e.dones[:n],
                    )
                    for e, n in zip(batch.episodes, episode_lengths, strict=True)
                ],
            )
        # norm=None always: any rescaling would destroy the exact encoded values.
        return DeterministicDataset(batch=batch)

    return build
