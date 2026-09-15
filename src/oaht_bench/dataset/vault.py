"""Flashbax Vault storage for a collected dataset (``docs/dataset_design.md`` §2).

The padded ``(episode, agent, T, …)`` :class:`~oaht_bench.dataset.schema.EpisodeBatch`
does not scale: padding wastes space and forces whole-file loads. OG-MARL and
D4RL both store a single *flat* buffer of transitions with explicit episode
boundaries instead. This module is that store, using a Flashbax Vault -- the same
format OG-MARL publishes (``<env>.vlt/{Good,Medium,Poor}/``), JAX-native and
memory-mapped so it reads past RAM.

**The agent axis is kept, not split into ego/teammate.** ``schema.py`` argues the
2-player assumption belongs in the runtime, not the artifact, so per-timestep
fields carry a leading ``agent`` axis and the ego seat is recorded rather than
assumed. The flat store mirrors that: ``observations``/``actions``/``rewards``/
``avail_actions`` keep the agent axis, and the ego/teammate split stays a
:class:`~oaht_bench.dataset.dataset.Dataset` concern. That is what lets the
store swap under ``EpisodeBatch`` without touching any baseline -- :func:`read_vault`
reconstructs the identical ragged batch.

Flat layout (Flashbax experience is ``(B, T, …)``; ``B=1`` for one stream):

    observations   (1, N, A, obs_dim)   actions      (1, N, A)
    rewards        (1, N, A)            avail_actions (1, N, A, num_actions)
    member_ids     (1, N, A)            terminals    (1, N)   episode_id (1, N)
    ego_response_quality (1, N)  -- broadcast per episode, present in pooled mode

``N`` is the total number of transitions. Collection is already ragged -- each
episode is real steps only -- so :func:`write_vault` concatenates them straight
into the buffer; padding never exists on the write side. :func:`read_vault` groups
transitions by ``episode_id`` into the ragged :class:`EpisodeBatch`
:class:`~oaht_bench.dataset.dataset.Dataset` consumes; padding only ever
reappears window-by-window inside ``Dataset``. Dataset-level metadata -- env, variant,
population/matrix hashes, the roster manifest, ``ego_index`` -- rides in the
Vault's own metadata, the small fixed-size part; per-episode labels are broadcast
into the flat fields.

This is the only dataset store: there is no ``.npz`` artifact. ``EpisodeBatch`` is
purely the in-memory, read-side shape ``Dataset`` consumes, produced by
:func:`read_vault`; it is never serialised.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oaht_bench.dataset.schema import Episode, EpisodeBatch

#: Bump when the flat field set or reconstruction changes in a way that makes an
#: older vault unreadable. Written into the vault metadata.
SCHEMA_VERSION = 1


def _json_safe(meta: dict) -> dict:
    """Coerce ``meta`` to JSON-round-trippable values (numpy -> str, etc.).

    The vault metadata is stored as JSON, and reading it back must give plain
    Python types; running it through ``json.dumps(default=str)`` / ``json.loads``
    here makes the coercion explicit and order-stable.
    """
    return json.loads(json.dumps(meta, sort_keys=True, default=str))


def to_flat(
    episodes: list[Episode],
    member_ids,
    *,
    ego_index: int,
    meta: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    """Concatenate collected episodes into flat transitions for a vault write.

    ``episodes`` are the ragged :class:`~oaht_bench.dataset.schema.Episode`\\ s
    :func:`~oaht_bench.dataset.construction.collect.collect_episode` returns -- real
    steps only, so there is nothing to drop; ``episode_id`` records which episode
    each transition came from. ``member_ids`` is ``(num_episodes, num_agents)``.
    Per-episode labels (``member_ids``, and ``ego_response_quality`` when the
    collection recorded it) are broadcast across their episode's transitions.
    """
    member_ids = np.asarray(member_ids)
    num_agents = int(member_ids.shape[1])
    per_episode_eps = meta.get("ego_response_quality")

    obs, acts, rews, avail, mem, term, epid, erq = [], [], [], [], [], [], [], []
    for ep, episode in enumerate(episodes):
        n = episode.length
        # (agent, T, …) -> (T, agent, …): one transition per row, agent axis kept.
        obs.append(np.asarray(episode.obs).transpose(1, 0, 2))
        acts.append(np.asarray(episode.actions).transpose(1, 0))
        rews.append(np.asarray(episode.rewards).transpose(1, 0))
        avail.append(np.asarray(episode.avail_actions).transpose(1, 0, 2))
        term.append(np.asarray(episode.dones))
        epid.append(np.full(n, ep, dtype=np.int32))
        mem.append(np.broadcast_to(member_ids[ep], (n, num_agents)))
        if per_episode_eps is not None:
            erq.append(np.full(n, float(per_episode_eps[ep]), dtype=np.float32))

    def stack(parts, dtype):
        # Prepend the (B=1) batch axis Flashbax expects.
        return np.concatenate(parts, axis=0)[None].astype(dtype)

    experience = {
        "observations": stack(obs, np.float32),
        "actions": stack(acts, np.int32),
        "rewards": stack(rews, np.float32),
        "avail_actions": stack(avail, np.float32),
        "member_ids": stack(mem, np.int32),
        "terminals": stack(term, np.bool_),
        "episode_id": stack(epid, np.int32),
    }
    if erq:
        experience["ego_response_quality"] = stack(erq, np.float32)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "ego_index": int(ego_index),
        "num_agents": num_agents,
        # The whole batch meta, minus the per-episode array now carried flat.
        "episode_batch_meta": {
            k: v for k, v in _json_safe(meta).items() if k != "ego_response_quality"
        },
    }
    return experience, metadata


def _split_dir(vault_dir: Path, variant: str | None) -> tuple[str, str, str]:
    """Map a vault directory to Flashbax's ``(rel_dir, vault_name, vault_uid)``.

    Flashbax writes ``<rel_dir>/<vault_name>/<vault_uid>/``. We use the variant as
    the uid so several variants share one ``<name>`` directory, reproducing
    OG-MARL's ``<env>.vlt/{Good,Medium,Poor}/`` quality-folder layout.
    """
    vault_dir = Path(vault_dir)
    uid = variant or "data"
    return str(vault_dir.parent), vault_dir.name, uid


def write_vault(
    episodes: list[Episode],
    member_ids,
    vault_dir: str | Path,
    *,
    ego_index: int,
    meta: dict,
) -> Path:
    """Write collected ``episodes`` to a Flashbax Vault at ``vault_dir/<variant>/``.

    ``episodes`` are the ragged :class:`~oaht_bench.dataset.schema.Episode`\\ s from
    collection and ``member_ids`` is ``(num_episodes, num_agents)``; nothing is
    padded. ``vault_dir`` is the
    ``<name>.vlt`` root and the variant (from ``meta``) becomes the sub-directory,
    so ``expert``/``mixed``/``br_vs_worst`` collections of one environment can live
    side by side. Returns the vault root.
    """
    from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
    from flashbax.vault import Vault

    experience, metadata = to_flat(episodes, member_ids, ego_index=ego_index, meta=meta)
    n = int(experience["episode_id"].shape[1])
    state = TrajectoryBufferState(
        experience=experience,
        current_index=np.asarray(n),
        is_full=np.asarray(True),
    )
    rel_dir, name, uid = _split_dir(Path(vault_dir), meta.get("variant"))
    vault = Vault(
        vault_name=name,
        experience_structure=state.experience,
        rel_dir=rel_dir,
        vault_uid=uid,
        metadata=metadata,
    )
    vault.write(state, source_interval=(0, n))
    return Path(vault_dir)


def read_vault(vault_dir: str | Path, *, variant: str | None = None) -> EpisodeBatch:
    """Reconstruct the ragged :class:`EpisodeBatch` from a vault.

    Re-groups the flat transitions by ``episode_id`` into one variable-length array
    per episode and restores ``meta`` -- the read-side view
    :class:`~oaht_bench.dataset.dataset.Dataset` consumes. ``variant`` selects
    the sub-directory; if omitted and the vault holds exactly one, that one is used.
    """
    from flashbax.vault import Vault

    vault_dir = Path(vault_dir)
    if variant is None:
        subs = sorted(p.name for p in vault_dir.iterdir() if p.is_dir())
        if len(subs) == 1:
            variant = subs[0]
        elif not subs:
            raise FileNotFoundError(f"no variant sub-directory under {vault_dir}")
        else:
            raise ValueError(f"{vault_dir} holds variants {subs}; pass variant= to pick one.")

    rel_dir, name, uid = _split_dir(vault_dir, variant)
    vault = Vault(vault_name=name, rel_dir=rel_dir, vault_uid=uid)
    state = vault.read()
    exp = {k: np.asarray(v)[0] for k, v in state.experience.items()}  # drop B axis
    meta = dict(vault._metadata)  # flashbax adds structure_* keys; we want ours

    ego_index = int(meta["ego_index"])
    batch_meta = dict(meta["episode_batch_meta"])

    epid = exp["episode_id"]

    # Regroup the flat transitions per episode. The store is transition-major with
    # the agent axis kept, ``(N, agent, …)``; transpose back to each Episode's
    # ``(agent, T_ep, …)`` layout.
    #
    # Group with a single stable sort rather than a boolean ``epid == ep`` scan per
    # episode: the mask-per-episode form is O(episodes × transitions) and on a large
    # pooled dataset (e.g. 25k episodes over 1.8M transitions) that is tens of minutes
    # of pure Python. The stable sort clusters each episode's transitions while
    # preserving their original (temporal) order, so ``np.split`` at the group
    # boundaries yields the same per-episode index blocks the mask did -- O(N log N).
    # ``np.unique`` returns sorted ids, so episodes come out in ascending id order,
    # matching the previous behaviour.
    order = np.argsort(epid, kind="stable")
    _, starts = np.unique(epid[order], return_index=True)
    groups = np.split(order, starts[1:])

    out_episodes, member_ids = [], []
    for g in groups:
        out_episodes.append(
            Episode(
                obs=exp["observations"][g].transpose(1, 0, 2).astype(np.float32),
                actions=exp["actions"][g].transpose(1, 0).astype(np.int64),
                rewards=exp["rewards"][g].transpose(1, 0).astype(np.float32),
                avail_actions=exp["avail_actions"][g].transpose(1, 0, 2).astype(np.float32),
                dones=exp["terminals"][g].astype(bool),
            )
        )
        # member_ids is per-episode: take it off any (the first) transition.
        member_ids.append(exp["member_ids"][g[0]])

    if "ego_response_quality" in exp:
        q = exp["ego_response_quality"]
        batch_meta["ego_response_quality"] = [float(q[g[0]]) for g in groups]

    return EpisodeBatch(
        episodes=out_episodes,
        member_ids=np.stack(member_ids),
        ego_index=ego_index,
        meta=batch_meta,
    )


class _DiskEpisode:
    """One episode read from the vault on access -- an :class:`Episode` work-alike.

    Holds only ``(source, ep)``; the arrays come from the source's slice-read cache,
    so a list of these is cheap to keep even for hundreds of thousands of episodes.
    """

    __slots__ = ("_src", "_ep")

    def __init__(self, src, ep):
        self._src, self._ep = src, ep

    @property
    def length(self):
        return int(self._src._lengths[self._ep])

    @property
    def num_agents(self):
        return self._src.num_agents

    @property
    def obs(self):
        return self._src._read(self._ep)["obs"]

    @property
    def actions(self):
        return self._src._read(self._ep)["actions"]

    @property
    def rewards(self):
        return self._src._read(self._ep)["rewards"]

    @property
    def avail_actions(self):
        return self._src._read(self._ep)["avail_actions"]

    @property
    def dones(self):
        return self._src._read(self._ep)["dones"]

    def returns(self):
        return self.rewards.sum(axis=1)


class _DiskEpisodeList:
    def __init__(self, src):
        self._src = src

    def __len__(self):
        return self._src.num_episodes

    def __getitem__(self, i):
        i = int(i)
        if i < 0:
            i += self._src.num_episodes
        if not 0 <= i < self._src.num_episodes:
            raise IndexError(i)
        return _DiskEpisode(self._src, i)

    def __iter__(self):
        for i in range(self._src.num_episodes):
            yield _DiskEpisode(self._src, i)


class DiskEpisodeSource:
    """An :class:`~oaht_bench.dataset.schema.EpisodeBatch` work-alike that streams
    transitions from the vault on disk instead of loading them into RAM.

    ``read_vault`` pulls the entire flat experience into memory (tens of GiB for a
    hundred-thousand-episode Hanabi dataset). The vault's fields are tensorstore
    arrays that support lazy slice reads, and episodes are stored contiguously, so
    this reads only per-episode metadata up front (``episode_id`` boundaries,
    ``member_ids``, ``rewards`` for returns -- all small) and fetches each episode's
    observations/actions on demand, behind a bounded LRU. Host memory is then set by
    the cache, not the dataset size, so windowing (via :class:`LazyWindows`) scales
    to arbitrarily large vaults.

    Quacks like ``EpisodeBatch`` for the fields :class:`~oaht_bench.dataset.dataset.Dataset`
    and :class:`LazyWindows` read: ``ego_index``, ``num_agents``, ``num_episodes``,
    ``episodes`` (a lazy list), ``member_ids``, ``episode_returns()`` and ``meta``.
    """

    def __init__(self, vault_dir, *, variant=None, cache_episodes: int = 1024):
        from collections import OrderedDict

        from flashbax.vault import Vault

        vault_dir = Path(vault_dir)
        if variant is None:
            subs = sorted(p.name for p in vault_dir.iterdir() if p.is_dir())
            if len(subs) == 1:
                variant = subs[0]
            elif not subs:
                raise FileNotFoundError(f"no variant sub-directory under {vault_dir}")
            else:
                raise ValueError(f"{vault_dir} holds variants {subs}; pass variant= to pick one.")

        rel_dir, name, uid = _split_dir(vault_dir, variant)
        vault = Vault(vault_name=name, rel_dir=rel_dir, vault_uid=uid)
        n = int(vault.vault_index)
        self._ds = vault._all_datastores
        meta = dict(vault._metadata)
        self.ego_index = int(meta["ego_index"])
        self.meta = dict(meta["episode_batch_meta"])

        # Small metadata read fully; observations/actions/avail stay on disk.
        epid = np.asarray(self._ds["episode_id"][0, 0:n].read().result()).reshape(-1)
        if np.any(np.diff(epid) < 0):
            raise ValueError(
                "vault transitions are not episode-contiguous; DiskEpisodeSource assumes "
                "each episode's transitions form one contiguous block (to_flat writes them "
                "that way). Fall back to read_vault for this vault."
            )
        uniq, counts = np.unique(epid, return_counts=True)
        self.num_episodes = int(len(uniq))
        self._lengths = counts.astype(np.int64)
        self._offsets = np.concatenate([[0], np.cumsum(self._lengths)[:-1]]).astype(np.int64)

        member_ids_full = np.asarray(self._ds["member_ids"][0, 0:n].read().result())
        self.member_ids = member_ids_full[self._offsets]
        self.num_agents = int(self.member_ids.shape[1])
        rewards_full = np.asarray(self._ds["rewards"][0, 0:n].read().result()).astype(np.float32)
        self._returns = np.stack(
            [
                rewards_full[o : o + ln].sum(0)
                for o, ln in zip(self._offsets, self._lengths, strict=True)
            ]
        )

        q = None
        if "ego_response_quality" in self._ds:
            qf = np.asarray(self._ds["ego_response_quality"][0, 0:n].read().result()).reshape(-1)
            q = [float(qf[o]) for o in self._offsets]
        if q is not None:
            self.meta["ego_response_quality"] = q

        self.episodes = _DiskEpisodeList(self)
        self._cache: OrderedDict = OrderedDict()
        self._cache_max = int(cache_episodes)

    def _read(self, ep):
        hit = self._cache.get(ep)
        if hit is not None:
            self._cache.move_to_end(ep)
            return hit
        o, ln = int(self._offsets[ep]), int(self._lengths[ep])
        s = slice(o, o + ln)
        rec = {
            "obs": np.asarray(self._ds["observations"][0, s].read().result())
            .transpose(1, 0, 2)
            .astype(np.float32),
            "actions": np.asarray(self._ds["actions"][0, s].read().result())
            .transpose(1, 0)
            .astype(np.int64),
            "rewards": np.asarray(self._ds["rewards"][0, s].read().result())
            .transpose(1, 0)
            .astype(np.float32),
            "avail_actions": np.asarray(self._ds["avail_actions"][0, s].read().result())
            .transpose(1, 0, 2)
            .astype(np.float32),
            "dones": np.asarray(self._ds["terminals"][0, s].read().result())
            .reshape(-1)
            .astype(bool),
        }
        self._cache[ep] = rec
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return rec

    @property
    def num_episodes_(self):  # pragma: no cover - convenience mirror
        return self.num_episodes

    def episode_returns(self):
        return self._returns

    def episode_lengths(self):
        return self._lengths


class VaultWriter:
    """Append episodes to a vault in chunks, so collection never holds them all in RAM.

    ``read_vault``'s inverse for large collections: each :meth:`write` flat-packs a
    chunk and appends it (flashbax vaults grow on write), continuing episode ids
    across chunks so the store stays **episode-contiguous** -- which
    :class:`DiskEpisodeSource` relies on. The vault-level metadata is fixed at the
    first chunk, so the batch-level ``meta`` must be constant across chunks;
    per-episode labels that vary (``ego_response_quality``) are passed per chunk and
    written as flat fields (never into the fixed metadata).

    Collecting straight through this keeps host memory bounded by the chunk size
    rather than the dataset, so hundred-thousand-episode vaults can be *produced*,
    not just read.
    """

    def __init__(self, vault_dir, *, ego_index: int, meta: dict):
        self._dir = Path(vault_dir)
        self._ego_index = int(ego_index)
        self._meta = dict(meta)
        self._vault = None
        self._ep_offset = 0
        self._n = 0

    def write(self, episodes, member_ids, *, ego_response_quality=None) -> None:
        if len(episodes) == 0:
            return
        from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
        from flashbax.vault import Vault

        chunk_meta = dict(self._meta)
        if ego_response_quality is not None:
            chunk_meta["ego_response_quality"] = [float(x) for x in ego_response_quality]
        experience, metadata = to_flat(
            episodes, np.asarray(member_ids), ego_index=self._ego_index, meta=chunk_meta
        )
        # Continue episode ids across chunks so the flat store stays contiguous.
        experience["episode_id"] = experience["episode_id"] + np.int32(self._ep_offset)
        n = int(experience["episode_id"].shape[1])
        state = TrajectoryBufferState(
            experience=experience, current_index=np.asarray(n), is_full=np.asarray(True)
        )
        if self._vault is None:
            rel_dir, name, uid = _split_dir(self._dir, self._meta.get("variant"))
            self._vault = Vault(
                vault_name=name,
                experience_structure=state.experience,
                rel_dir=rel_dir,
                vault_uid=uid,
                metadata=metadata,
            )
        self._vault.write(state, source_interval=(0, n))
        self._ep_offset += len(episodes)
        self._n += n

    @property
    def num_episodes(self) -> int:
        return self._ep_offset

    def close(self) -> Path:
        return self._dir
