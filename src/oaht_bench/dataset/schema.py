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

An :class:`Episode` is one episode's transitions, real steps only; an
:class:`EpisodeBatch` is a list of them plus seat provenance. Collection produces
``Episode``\\ s (:func:`~oaht_bench.dataset.construction.collect.collect_episode`)
and the pipeline runs off either -- one ``Episode`` or a batch -- so nothing is
ever padded to a rectangle. Neither is serialised: the on-disk store is a
flat-transition Flashbax Vault (:mod:`oaht_bench.dataset.vault`), and
:func:`~oaht_bench.dataset.vault.read_vault` rebuilds the batch from it. Padding
only reappears window-by-window inside ``Dataset``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Episode:
    """One episode's transitions -- real steps only, no padding.

    Per-agent fields keep a leading ``agent`` axis (the schema-level generality
    argument above); ``dones`` is shared across seats. ``T`` is the episode's real
    length. This is what ``collect_episode`` produces and what a Vault
    transition-group reconstructs.
    """

    #: (num_agents, T, obs_dim)
    obs: np.ndarray
    #: (num_agents, T)
    actions: np.ndarray
    #: (num_agents, T)
    rewards: np.ndarray
    #: (num_agents, T, num_actions) — which actions the environment permitted.
    #: **Not** all-ones on LBF: 4.77 of 6 actions available on average and action 5
    #: unavailable 67% of the time. An earlier comment claimed LBF did not mask,
    #: which is why the offline baselines were built without the field.
    avail_actions: np.ndarray
    #: (T,) — environment-level termination, shared across seats.
    dones: np.ndarray

    @property
    def length(self) -> int:
        return int(self.dones.shape[0])

    @property
    def num_agents(self) -> int:
        return int(self.obs.shape[0])

    def returns(self) -> np.ndarray:
        """Per-agent undiscounted return. (num_agents,)"""
        return self.rewards.sum(axis=1)


@dataclass(frozen=True)
class EpisodeBatch:
    """A batch of ragged :class:`Episode`\\ s plus their seat provenance.

    Episodes end at different times, so the batch is just a list of them rather
    than a rectangle with a ``valid`` mask (which every consumer immediately
    un-padded anyway). ``member_ids`` is a plain ``(num_episodes, num_agents)``
    array -- one seat assignment per episode, not per step -- and the ego seat is
    recorded rather than assumed to be index 0.
    """

    episodes: list[Episode]
    #: (num_episodes, num_agents)
    member_ids: np.ndarray
    #: Seat the learner occupies. Recorded rather than assumed to be 0.
    ego_index: int
    #: Free-form provenance: config hash, variant, generator, env.
    meta: dict[str, Any]

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def num_agents(self) -> int:
        return self.episodes[0].num_agents

    def episode_returns(self) -> np.ndarray:
        """Per-episode, per-agent undiscounted return. (num_episodes, num_agents)"""
        return np.stack([e.returns() for e in self.episodes])

    def episode_lengths(self) -> np.ndarray:
        """(num_episodes,) real step count per episode."""
        return np.asarray([e.length for e in self.episodes])

    def describe(self) -> str:
        ret = self.episode_returns()
        lengths = self.episode_lengths()
        return (
            f"episodes      {self.num_episodes}\n"
            f"agents        {self.num_agents} (ego seat {self.ego_index})\n"
            f"steps         mean {lengths.mean():.1f}, max {int(lengths.max())}\n"
            f"return        ego {ret[:, self.ego_index].mean():.4f}, "
            f"joint {ret.sum(axis=1).mean():.4f}"
        )
