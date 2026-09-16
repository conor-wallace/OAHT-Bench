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
    """Load a vault fully into an in-RAM :class:`EpisodeBatch`.

    The eager counterpart to :class:`VaultReader` -- it materializes every episode --
    for the rare caller that wants the whole dataset resident (tests, small tools).
    :class:`~oaht_bench.dataset.dataset.Dataset` does *not* use this; it streams
    through :class:`VaultReader`. Kept as a thin wrapper so the flat-store-to-episode
    layout lives in exactly one place (:class:`VaultReader`) rather than a second,
    divergent copy. ``variant`` selects the sub-directory; if omitted and the vault
    holds exactly one, that one is used.
    """
    reader = VaultReader(vault_dir, variant=variant)
    episodes = [
        Episode(
            obs=e.obs,
            actions=e.actions,
            rewards=e.rewards,
            avail_actions=e.avail_actions,
            dones=e.dones,
        )
        for e in reader.episodes
    ]
    return EpisodeBatch(
        episodes=episodes,
        member_ids=np.asarray(reader.member_ids),
        ego_index=reader.ego_index,
        meta=reader.meta,
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


class VaultReader:
    """An :class:`~oaht_bench.dataset.schema.EpisodeBatch` work-alike that streams
    transitions from the vault on disk instead of loading them into RAM.

    ``read_vault`` pulls the entire flat experience into memory (tens of GiB for a
    hundred-thousand-episode Hanabi dataset). The vault's fields are tensorstore
    arrays that support lazy slice reads, and episodes are stored contiguously, so
    this reads only per-episode metadata up front (``episode_id`` boundaries,
    ``member_ids``, ``rewards`` for returns -- all small) and fetches each episode's
    observations/actions on demand, behind a bounded LRU. Host memory is then set by
    the cache, not the dataset size, so windowing (via :class:`~oaht_bench.dataset.dataset.Windows`) scales
    to arbitrarily large vaults.

    Quacks like ``EpisodeBatch`` for the fields :class:`~oaht_bench.dataset.dataset.Dataset`
    and :class:`~oaht_bench.dataset.dataset.Windows` read: ``ego_index``, ``num_agents``, ``num_episodes``,
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
        self._norm_sidecar = vault_dir / uid / "norm_stats.json"
        n = int(vault.vault_index)
        self._ds = vault._all_datastores
        meta = dict(vault._metadata)
        self.ego_index = int(meta["ego_index"])
        self.meta = dict(meta["episode_batch_meta"])

        # Small metadata read fully; observations/actions/avail stay on disk.
        epid = np.asarray(self._ds["episode_id"][0, 0:n].read().result()).reshape(-1)
        if np.any(np.diff(epid) < 0):
            raise ValueError(
                "vault transitions are not episode-contiguous; VaultReader assumes "
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
        # Kept in RAM (small: one float per transition per agent) so the training
        # normalisation's rtg_scale needs no per-episode disk read.
        self._rewards = rewards_full
        self._n_transitions = n
        self._obs_dim = int(self._ds["observations"].shape[-1])

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

    def _read_norm_sidecar(self):
        """The stored norm iff it exists and still covers the whole vault.

        The count guard is what keeps a stored derived quantity honest: it is used
        only while it describes exactly the transitions present, so an appended-to
        vault recomputes rather than trusting stale statistics.
        """
        import json

        if not self._norm_sidecar.exists():
            return None
        d = json.loads(self._norm_sidecar.read_text())
        if int(d.get("n_transitions", -1)) != self._n_transitions:
            return None
        return (
            np.asarray(d["obs_mean"], np.float32),
            np.asarray(d["obs_std"], np.float32),
            float(d["rtg_scale"]),
        )

    def norm_stats(self, *, chunk: int = 100_000):
        """Exact observation mean/std and rtg-scale for the training normalisation.

        Returns raw ``(obs_mean, obs_std, rtg_scale)`` (not a ``Normalization`` --
        that lives in :mod:`dataset` and would close an import cycle) so the caller
        can wrap them.

        Prefers the ``norm_stats.json`` sidecar :class:`VaultWriter` writes at
        collection time (the moments accumulated for free while the data streamed
        through RAM), so at training time the load costs nothing. The sidecar is
        trusted only while its ``n_transitions`` still matches the vault -- if the
        vault was appended to since, the stored norm no longer covers it and we fall
        through to recomputing.

        The fallback is the *exact* dataset statistics, accumulated in one streaming
        pass over the ego observation column in large contiguous ``chunk`` blocks. The
        per-episode streaming path this replaces was slow not because of total bytes
        but read *granularity* -- one tensorstore request per episode, each pulling all
        five fields. Reading the obs field only, in a few hundred bulk blocks, is ~5x
        less data in ~1/1000th the requests, so the exact mean is affordable (tens of
        seconds on a hundred-thousand-episode vault) and there is no sampling bias --
        which matters because the batched collector writes episodes **grouped by
        pairing**, so any partial sample skews toward whichever pairings it lands in.

        ``rtg_scale`` comes from the rewards already resident in RAM.
        """
        cached = self._read_norm_sidecar()
        if cached is not None:
            return cached

        ego = self.ego_index
        obs_ds = self._ds["observations"]
        n = self._n_transitions

        total = np.zeros(self._obs_dim, np.float64)
        total_sq = np.zeros(self._obs_dim, np.float64)
        # Clamp the final block to n: the vault's tensorstore is allocated past the
        # written length, and reading into that tail would fold zero padding into the
        # moments (shrinking mean and std toward zero).
        for st in range(0, n, chunk):
            block = np.asarray(obs_ds[0, st : min(st + chunk, n), ego].read().result())
            block = block.astype(np.float64)
            total += block.sum(0)
            total_sq += (block * block).sum(0)
        obs_mean = total / n
        obs_std = np.sqrt(np.maximum(total_sq / n - obs_mean * obs_mean, 0.0))
        obs_std = np.maximum(obs_std, 1e-6).astype(np.float32)
        obs_mean = obs_mean.astype(np.float32)

        # rtg over every episode's ego rewards (reverse cumsum), std as the scale.
        rtgs = []
        for o, ln in zip(self._offsets, self._lengths, strict=True):
            r = self._rewards[o : o + ln, ego]
            rtgs.append(np.cumsum(r[::-1])[::-1])
        rtg_scale = float(max(np.concatenate(rtgs).std(), 1e-6))
        return obs_mean, obs_std, rtg_scale


class VaultWriter:
    """Append episodes to a vault in chunks, so collection never holds them all in RAM.

    ``read_vault``'s inverse for large collections: each :meth:`write` flat-packs a
    chunk and appends it (flashbax vaults grow on write), continuing episode ids
    across chunks so the store stays **episode-contiguous** -- which
    :class:`VaultReader` relies on. The vault-level metadata is fixed at the
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
        # Running moments for the training normalisation, accumulated over the ego
        # observation and return-to-go as each chunk streams through -- the data is
        # already in RAM here, so the norm costs nothing extra and need not be
        # recomputed by a per-episode disk pass at load. Written as a sidecar on
        # ``close`` (see :meth:`VaultReader.norm_stats`).
        self._obs_sum = None
        self._obs_sqsum = None
        self._rtg_sum = 0.0
        self._rtg_sqsum = 0.0
        self._rtg_count = 0

    def _accumulate_norm(self, experience, episodes) -> None:
        ego = self._ego_index
        # Reduce over the flat observation array ``to_flat`` already built for this
        # chunk -- no extra copy, one memory-bound pass -- rather than looping over
        # episodes (overhead-bound: it cost ~10% of collection wall-clock). ``einsum``
        # gives the sum of squares without materialising a squared temporary.
        flat_obs = experience["observations"][0, :, ego]  # (N_transitions, obs_dim)
        if flat_obs.shape[0]:
            if self._obs_sum is None:
                self._obs_sum = np.zeros(flat_obs.shape[-1], np.float64)
                self._obs_sqsum = np.zeros(flat_obs.shape[-1], np.float64)
            self._obs_sum += flat_obs.sum(0, dtype=np.float64)
            self._obs_sqsum += np.einsum("ij,ij->j", flat_obs, flat_obs, dtype=np.float64)
        # rtg is inherently per-trajectory (a reverse-cumsum), but over length-T
        # reward vectors it is negligible next to the observation reduction.
        for ep in episodes:
            r = np.asarray(ep.rewards[ego], np.float64)  # (T,)
            if r.shape[0] == 0:
                continue
            rtg = np.cumsum(r[::-1])[::-1]
            self._rtg_sum += float(rtg.sum())
            self._rtg_sqsum += float((rtg * rtg).sum())
            self._rtg_count += rtg.shape[0]

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
        # Accumulate the training norm off the flat array to_flat just built.
        self._accumulate_norm(experience, episodes)
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
        """Finalise the vault and write the ``norm_stats.json`` sidecar.

        The norm is exact (accumulated over the full contents as they were written)
        and tagged with the transition count it covers, so a reader trusts it only
        while the vault still has that many transitions -- if the vault is later
        appended to, the count no longer matches and the reader recomputes.
        """
        import json

        if self._obs_sum is not None and self._n > 0:
            n = self._n
            obs_mean = self._obs_sum / n
            obs_std = np.sqrt(np.maximum(self._obs_sqsum / n - obs_mean * obs_mean, 0.0))
            obs_std = np.maximum(obs_std, 1e-6)
            c = max(self._rtg_count, 1)
            rtg_mean = self._rtg_sum / c
            rtg_scale = max((self._rtg_sqsum / c - rtg_mean * rtg_mean) ** 0.5, 1e-6)
            uid = self._meta.get("variant") or "data"
            sidecar = self._dir / uid / "norm_stats.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "n_transitions": int(n),
                        "obs_mean": obs_mean.tolist(),
                        "obs_std": obs_std.tolist(),
                        "rtg_scale": float(rtg_scale),
                    }
                )
            )
        return self._dir
