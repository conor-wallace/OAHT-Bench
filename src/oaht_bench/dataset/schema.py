"""In-memory shape of a collected dataset (§4.1).

**Agent-count generality is a schema decision, not a runtime one.** Every
environment currently instantiates two seats and three of the four generators
assert ``num_agents == 2``, so nothing can populate a third seat today. But the
runtime is one module and cheap to change, whereas the schema is what every
baseline reads — changing it later means regenerating every dataset.

So per-agent quantities carry a leading ``agent`` axis rather than being split
into ``agent_0_*`` / ``agent_1_*`` fields, and the ego seat is recorded
explicitly instead of assumed to be index 0. Two-player is then just
``num_agents == 2`` and no consumer has to be rewritten when it isn't.

This is the padded, read-side shape :func:`~oaht_bench.offline.dataset.make_windows`
consumes. It is not serialised: the on-disk store is a flat-transition Flashbax
Vault (:mod:`oaht_bench.dataset.vault`), and :func:`~oaht_bench.dataset.vault.read_vault`
reconstructs this batch from it. Padding is a read-side artifact, not a stored one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EpisodeBatch:
    """A stack of episodes, padded to a common length.

    Leading axes are ``(episode, agent, timestep)`` for per-agent quantities and
    ``(episode, timestep)`` for shared ones. ``valid`` marks real steps, since
    episodes terminate at different times and are padded to ``max_steps``.
    """

    #: (num_episodes, num_agents, T, obs_dim)
    obs: np.ndarray
    #: (num_episodes, num_agents, T)
    actions: np.ndarray
    #: (num_episodes, num_agents, T)
    rewards: np.ndarray
    #: (num_episodes, T) — environment-level termination, shared across seats.
    dones: np.ndarray
    #: (num_episodes, T) — True where the step happened rather than padding.
    valid: np.ndarray
    #: (num_episodes, num_agents, T, num_actions) — which actions the
    #: environment permitted at each step. **Not** all-ones on LBF: every step
    #: masks something there, 4.77 of 6 actions available on average and action 5
    #: unavailable 67% of the time. An earlier version of this comment claimed
    #: LBF did not mask, which is why the offline baselines were built without
    #: consulting the field.
    avail_actions: np.ndarray
    #: Which population member sat in each seat, per episode.
    #: (num_episodes, num_agents)
    member_ids: np.ndarray
    #: Seat the learner occupies. Recorded rather than assumed to be 0.
    ego_index: int
    #: Free-form provenance: config hash, variant, generator, env.
    meta: dict[str, Any]

    @property
    def num_episodes(self) -> int:
        return int(self.obs.shape[0])

    @property
    def num_agents(self) -> int:
        return int(self.obs.shape[1])

    def episode_returns(self) -> np.ndarray:
        """Per-episode, per-agent undiscounted return. (num_episodes, num_agents)"""
        return (self.rewards * self.valid[:, None, :]).sum(axis=-1)

    def episode_lengths(self) -> np.ndarray:
        """(num_episodes,) real step count, excluding padding."""
        return self.valid.sum(axis=-1)

    def describe(self) -> str:
        ret = self.episode_returns()
        return (
            f"episodes      {self.num_episodes}\n"
            f"agents        {self.num_agents} (ego seat {self.ego_index})\n"
            f"steps         mean {self.episode_lengths().mean():.1f}, "
            f"max {self.obs.shape[2]}\n"
            f"return        ego {ret[:, self.ego_index].mean():.4f}, "
            f"joint {ret.sum(axis=1).mean():.4f}"
        )
