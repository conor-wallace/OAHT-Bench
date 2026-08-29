"""Flashbax Vault storage for a collected dataset (``docs/dataset_design.md`` §2).

The padded ``(episode, agent, T, …)`` :class:`~oaht_bench.data.schema.EpisodeBatch`
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
:func:`~oaht_bench.offline.dataset.make_windows` concern. That is what lets the
store swap under ``EpisodeBatch`` without touching any baseline -- :func:`read_vault`
reconstructs the identical padded batch.

Flat layout (Flashbax experience is ``(B, T, …)``; ``B=1`` for one stream):

    observations   (1, N, A, obs_dim)   actions      (1, N, A)
    rewards        (1, N, A)            avail_actions (1, N, A, num_actions)
    member_ids     (1, N, A)            terminals    (1, N)   episode_id (1, N)
    ego_response_quality (1, N)  -- broadcast per episode, present in pooled mode

``N`` is the total number of *real* transitions (padding is dropped on write and
re-derived on read from ``episode_id``). Dataset-level metadata -- env, variant,
population/matrix hashes, the roster manifest, ``ego_index`` -- rides in the
Vault's own metadata, the small fixed-size part; per-episode labels are broadcast
into the flat fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oaht_bench.data.schema import EpisodeBatch

#: Bump when the flat field set or reconstruction changes in a way that makes an
#: older vault unreadable. Written into the vault metadata.
SCHEMA_VERSION = 1


def _json_safe(meta: dict) -> dict:
    """Coerce ``meta`` the same way :meth:`EpisodeBatch.load` does on read.

    ``EpisodeBatch`` round-trips its meta through ``json.dumps(default=str)`` then
    ``json.loads``; matching that here means a batch written to a vault and read
    back has a byte-identical meta to one written to ``.npz`` and read back.
    """
    return json.loads(json.dumps(meta, sort_keys=True, default=str))


def to_flat(batch: EpisodeBatch) -> tuple[dict[str, np.ndarray], dict]:
    """Flatten a padded batch into ``(experience, metadata)`` for a vault write.

    Only the ``valid`` steps of each episode survive; ``episode_id`` marks the
    boundaries so :func:`read_vault` can re-pad. Per-episode labels
    (``member_ids``, and ``ego_response_quality`` when the collection recorded it)
    are broadcast across their episode's transitions.
    """
    per_episode_eps = batch.meta.get("ego_response_quality")

    obs, acts, rews, avail, mem, term, epid, erq = [], [], [], [], [], [], [], []
    for ep in range(batch.num_episodes):
        v = np.asarray(batch.valid[ep], dtype=bool)
        n = int(v.sum())
        if n == 0:
            continue
        # (agent, T, …) -> (T_real, agent, …): transpose the agent axis inward so
        # a transition is one row, matching the flat convention.
        obs.append(np.asarray(batch.obs[ep])[:, v].transpose(1, 0, 2))
        acts.append(np.asarray(batch.actions[ep])[:, v].transpose(1, 0))
        rews.append(np.asarray(batch.rewards[ep])[:, v].transpose(1, 0))
        avail.append(np.asarray(batch.avail_actions[ep])[:, v].transpose(1, 0, 2))
        term.append(np.asarray(batch.dones[ep])[v])
        epid.append(np.full(n, ep, dtype=np.int32))
        mem.append(np.broadcast_to(np.asarray(batch.member_ids[ep]), (n, batch.num_agents)))
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
        "ego_index": int(batch.ego_index),
        "num_agents": int(batch.num_agents),
        # The whole batch meta, minus the per-episode array now carried flat.
        "episode_batch_meta": {
            k: v for k, v in _json_safe(batch.meta).items() if k != "ego_response_quality"
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


def write_vault(batch: EpisodeBatch, vault_dir: str | Path) -> Path:
    """Write ``batch`` to a Flashbax Vault at ``vault_dir/<variant>/``.

    ``vault_dir`` is the ``<name>.vlt`` root; the variant (from ``batch.meta``)
    becomes the sub-directory, so ``expert``/``mixed``/``br_vs_worst`` collections
    of one environment can live side by side. Returns the vault root.
    """
    from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
    from flashbax.vault import Vault

    experience, metadata = to_flat(batch)
    n = int(experience["episode_id"].shape[1])
    state = TrajectoryBufferState(
        experience=experience,
        current_index=np.asarray(n),
        is_full=np.asarray(True),
    )
    rel_dir, name, uid = _split_dir(Path(vault_dir), batch.meta.get("variant"))
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
    """Reconstruct the padded :class:`EpisodeBatch` written by :func:`write_vault`.

    Re-groups the flat transitions by ``episode_id``, pads each episode back to the
    common length with zeros (matching :func:`~oaht_bench.data.collect.pad_and_stack`),
    and restores the ``valid`` mask and ``meta``. The result equals the batch that
    was written -- so a consumer that took an ``EpisodeBatch`` cannot tell which
    store it came from.
    """
    from flashbax.vault import Vault

    rel_dir, name, uid = _split_dir(Path(vault_dir), variant)
    vault = Vault(vault_name=name, rel_dir=rel_dir, vault_uid=uid)
    state = vault.read()
    exp = {k: np.asarray(v)[0] for k, v in state.experience.items()}  # drop B axis
    meta = dict(vault._metadata)  # flashbax adds structure_* keys; we want ours

    ego_index = int(meta["ego_index"])
    num_agents = int(meta["num_agents"])
    batch_meta = dict(meta["episode_batch_meta"])

    epid = exp["episode_id"]
    episodes = [int(e) for e in np.unique(epid)]
    lengths = {ep: int((epid == ep).sum()) for ep in episodes}
    max_len = max(lengths.values())

    def pad_field(arr_per_ep, trailing):
        """Stack episodes into ``(E, T, *trailing)``, zero-padded to ``max_len``."""
        out = np.zeros((len(episodes), max_len, *trailing), dtype=arr_per_ep[0].dtype)
        for i, a in enumerate(arr_per_ep):
            out[i, : a.shape[0]] = a
        return out

    # Regroup contiguous transitions per episode, preserving order.
    def per_ep(field):
        return [exp[field][epid == ep] for ep in episodes]

    obs_ep = per_ep("observations")
    obs = pad_field(obs_ep, obs_ep[0].shape[1:]).transpose(0, 2, 1, 3)  # -> (E, A, T, obs)
    acts = pad_field(per_ep("actions"), (num_agents,)).transpose(0, 2, 1)
    rews = pad_field(per_ep("rewards"), (num_agents,)).transpose(0, 2, 1)
    avail_ep = per_ep("avail_actions")
    avail = pad_field(avail_ep, avail_ep[0].shape[1:]).transpose(0, 2, 1, 3)
    dones = pad_field(per_ep("terminals"), ())
    valid = np.zeros((len(episodes), max_len), dtype=bool)
    for i, ep in enumerate(episodes):
        valid[i, : lengths[ep]] = True
    # member_ids is per-episode: take it off any (the first) transition.
    member_ids = np.stack([exp["member_ids"][epid == ep][0] for ep in episodes])

    if "ego_response_quality" in exp:
        batch_meta["ego_response_quality"] = [
            float(exp["ego_response_quality"][epid == ep][0]) for ep in episodes
        ]

    return EpisodeBatch(
        obs=obs.astype(np.float32),
        actions=acts.astype(np.int64),
        rewards=rews.astype(np.float32),
        dones=dones.astype(bool),
        valid=valid,
        avail_actions=avail.astype(np.float32),
        member_ids=member_ids,
        ego_index=ego_index,
        meta=batch_meta,
    )
