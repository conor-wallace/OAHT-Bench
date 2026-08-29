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
:func:`~oaht_bench.dataset.windows.make_windows` concern. That is what lets the
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
:func:`~oaht_bench.dataset.windows.make_windows` consumes; padding only ever
reappears window-by-window inside ``make_windows``. Dataset-level metadata -- env, variant,
population/matrix hashes, the roster manifest, ``ego_index`` -- rides in the
Vault's own metadata, the small fixed-size part; per-episode labels are broadcast
into the flat fields.

This is the only dataset store: there is no ``.npz`` artifact. ``EpisodeBatch`` is
purely the in-memory, read-side shape ``make_windows`` consumes, produced by
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
    :func:`~oaht_bench.dataset.windows.make_windows` consumes. ``variant`` selects
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
    episodes = [int(e) for e in np.unique(epid)]

    # Regroup the flat transitions per episode. The store is transition-major with
    # the agent axis kept, ``(N, agent, …)``; transpose back to each Episode's
    # ``(agent, T_ep, …)`` layout.
    out_episodes, member_ids = [], []
    for ep in episodes:
        m = epid == ep
        out_episodes.append(
            Episode(
                obs=exp["observations"][m].transpose(1, 0, 2).astype(np.float32),
                actions=exp["actions"][m].transpose(1, 0).astype(np.int64),
                rewards=exp["rewards"][m].transpose(1, 0).astype(np.float32),
                avail_actions=exp["avail_actions"][m].transpose(1, 0, 2).astype(np.float32),
                dones=exp["terminals"][m].astype(bool),
            )
        )
        # member_ids is per-episode: take it off any (the first) transition.
        member_ids.append(exp["member_ids"][m][0])

    if "ego_response_quality" in exp:
        batch_meta["ego_response_quality"] = [
            float(exp["ego_response_quality"][epid == ep][0]) for ep in episodes
        ]

    return EpisodeBatch(
        episodes=out_episodes,
        member_ids=np.stack(member_ids),
        ego_index=ego_index,
        meta=batch_meta,
    )
