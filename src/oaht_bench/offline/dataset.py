"""Turn a collected :class:`~oaht_bench.data.schema.EpisodeBatch` into training windows.

Every trajectory-view baseline consumes the same tensors — an ego stream to
predict from and a teammate stream to model — so the split happens once here
rather than inside each method. What differs between methods is what they *do*
with the teammate stream: LIAM reconstructs it from the ego embeddings, TAO
encodes it into a policy embedding and cross-attends.

Return-to-go is computed rather than stored, because it is a function of the
rewards already in the artifact and storing it would let the two disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oaht_bench.data.schema import EpisodeBatch


@dataclass(frozen=True)
class Windows:
    """Fixed-length windows over the ego and teammate streams.

    Leading axis is the window. ``T`` is the context length: TAO and TAGET both
    train on fixed-length fragments rather than whole episodes, and
    :mod:`oaht_bench.offline.backbone` needs a static shape to jit.
    """

    #: (N, T, obs_dim) — the learner's observations.
    ego_obs: np.ndarray
    #: (N, T) — the learner's actions, the behaviour-cloning target.
    ego_actions: np.ndarray
    #: (N, T) — return-to-go for the learner, the DT conditioning signal.
    ego_rtg: np.ndarray
    #: (N, T, obs_dim) — teammate observations, LIAM's reconstruction target
    #: and part of TAO's encoder input.
    mate_obs: np.ndarray
    #: (N, T) — teammate actions.
    mate_actions: np.ndarray
    #: (N, T) — teammate rewards, fused into TAO's encoder tokens.
    mate_rewards: np.ndarray
    #: (N, T) — timestep within the episode, for the positional encoding.
    timesteps: np.ndarray
    #: (N, T) — False where the window ran past the end of its episode.
    mask: np.ndarray
    #: (N,) — which population member was the teammate. TAO's InfoNCE positives
    #: are defined by this label; it is the field §4.2 anticipated.
    teammate_id: np.ndarray

    def __len__(self) -> int:
        return int(self.ego_obs.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.ego_obs.shape[-1])

    @property
    def context_length(self) -> int:
        return int(self.ego_obs.shape[1])


def return_to_go(rewards: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Reverse-cumulative reward, zeroed past the end of the episode.

    ``rewards`` and ``valid`` are ``(episode, T)``. Padding contributes nothing,
    which matters because episodes are padded to a common length and a nonzero
    tail would inflate every earlier target.
    """
    r = np.asarray(rewards) * np.asarray(valid)
    return np.flip(np.cumsum(np.flip(r, axis=-1), axis=-1), axis=-1)


def make_windows(
    batch: EpisodeBatch,
    *,
    context_length: int,
    stride: int = 1,
    teammate_index: int | None = None,
) -> Windows:
    """Slice every episode into overlapping windows of ``context_length``.

    Args:
        batch: A collected dataset.
        context_length: Timesteps per window. AD found in-context RL only emerges
            with multi-episode context; for behaviour-cloning baselines like LIAM
            a within-episode window is what the specification asks for.
        stride: Step between window starts.
        teammate_index: Seat treated as the teammate. Defaults to the seat that
            is not ``batch.ego_index``, which is unambiguous only while every
            environment has two seats — hence the explicit error below.
    """
    ego = batch.ego_index
    if teammate_index is None:
        others = [i for i in range(batch.num_agents) if i != ego]
        if len(others) != 1:
            raise ValueError(
                f"{batch.num_agents} agents, so 'the teammate' is ambiguous; pass "
                f"teammate_index explicitly. (Every current environment has two "
                f"seats, but the schema deliberately does not assume it.)"
            )
        teammate_index = others[0]

    rtg = return_to_go(batch.rewards[:, ego], batch.valid)
    T = context_length
    ego_o, ego_a, ego_g = [], [], []
    mate_o, mate_a, mate_r = [], [], []
    steps, masks, ids = [], [], []

    for ep in range(batch.num_episodes):
        length = int(batch.valid[ep].sum())
        if length == 0:
            continue
        for start in range(0, max(1, length - T + 1), stride):
            sl = slice(start, start + T)
            n = min(T, length - start)
            if n <= 0:
                continue

            def pad(arr, width=T):
                out = np.zeros((width, *arr.shape[1:]), dtype=arr.dtype)
                out[: arr.shape[0]] = arr
                return out

            ego_o.append(pad(batch.obs[ep, ego][sl][:n]))
            ego_a.append(pad(batch.actions[ep, ego][sl][:n]))
            ego_g.append(pad(rtg[ep][sl][:n]))
            mate_o.append(pad(batch.obs[ep, teammate_index][sl][:n]))
            mate_a.append(pad(batch.actions[ep, teammate_index][sl][:n]))
            mate_r.append(pad(batch.rewards[ep, teammate_index][sl][:n]))
            steps.append(pad(np.arange(start, start + n)))
            m = np.zeros(T, dtype=bool)
            m[:n] = True
            masks.append(m)
            ids.append(batch.member_ids[ep, teammate_index])

    return Windows(
        ego_obs=np.stack(ego_o).astype(np.float32),
        ego_actions=np.stack(ego_a).astype(np.int32),
        ego_rtg=np.stack(ego_g).astype(np.float32),
        mate_obs=np.stack(mate_o).astype(np.float32),
        mate_actions=np.stack(mate_a).astype(np.int32),
        mate_rewards=np.stack(mate_r).astype(np.float32),
        timesteps=np.stack(steps).astype(np.int32),
        mask=np.stack(masks),
        teammate_id=np.asarray(ids, dtype=np.int32),
    )
