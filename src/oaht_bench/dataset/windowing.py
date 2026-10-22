"""Window a single episode slice on demand -- the one place this is implemented.

Each baseline builds its own training batch (see ``offline/bc.py``,
``offline/liam.py``, ``offline/tao.py``, ``offline/omis.py``): a loop over
sampled ``(episode_id, t)`` pairs calling :func:`window_slice`, composed
however that baseline's own paper describes its batch. This module holds only
the two primitives every one of those loops needs and that aren't one-liners
on data the episode store (:class:`~oaht_bench.dataset.schema.EpisodeBatch` or
:class:`~oaht_bench.dataset.vault.VaultReader`) already exposes directly.

``window_slice`` returns raw, un-normalized arrays -- observation and
return-to-go normalization is the caller's job, via
:meth:`~oaht_bench.dataset.dataset.Normalization.apply_obs`/``apply_rtg`` on
the dataset's own ``norm``, kept separate so this module has no notion of a
particular dataset's statistics.
"""

from __future__ import annotations

import numpy as np

#: Vault store key -> in-RAM :class:`~oaht_bench.dataset.schema.Episode` attribute.
_STORE_ATTR = {
    "observations": "obs",
    "actions": "actions",
    "rewards": "rewards",
    "avail_actions": "avail_actions",
}


def resolve_seats(source, mate: int | None = None) -> tuple[int, int]:
    """``(ego, mate)`` seat indices -- ``mate`` is the seat ``source``'s ego
    plays against, inferred when there are exactly two agents, else required.

    The one place this ambiguity is resolved; :class:`~oaht_bench.dataset.
    dataset.Dataset` calls it once at construction and :func:`window_slice`
    calls it again per window (cheap -- no state to share) so it works from a
    bare source too, not just through ``Dataset``.
    """
    ego = source.ego_index
    if mate is not None:
        return ego, mate
    others = [i for i in range(source.num_agents) if i != ego]
    if len(others) != 1:
        raise ValueError(
            f"{source.num_agents} agents, so 'the teammate' is ambiguous; pass mate explicitly."
        )
    return ego, others[0]


def _episode_length(source, episode_id: int) -> int:
    if hasattr(source, "episode_lengths"):
        return int(np.asarray(source.episode_lengths())[episode_id])
    return int(source.episodes[episode_id].length)


def _read_raw(source, store_key: str, seat: int, episode_id: int, start: int, n: int) -> np.ndarray:
    """The unpadded ``[start, start + n)`` slice of one field/seat -- read
    concurrently off disk (``VaultReader.read_slices``) when the source is
    disk-backed, else sliced directly from a resident ``EpisodeBatch``."""
    if hasattr(source, "read_slices"):
        offset = int(source.transition_offsets[episode_id]) + start
        return source.read_slices(store_key, seat, np.asarray([offset]), np.asarray([n]))[0]
    attr = _STORE_ATTR[store_key]
    episode = source.episodes[episode_id]
    return np.asarray(getattr(episode, attr)[seat][start : start + n])


def _pad(arr: np.ndarray, h: int, fill) -> np.ndarray:
    n = arr.shape[0]
    if n == h:
        return arr
    head = np.full((h - n, *arr.shape[1:]), fill, dtype=arr.dtype)
    return np.concatenate([head, arr], axis=0)


def _build_obs(source, seat, episode_id, start, n, h):
    raw = _read_raw(source, "observations", seat, episode_id, start, n)
    return _pad(raw, h, 0.0).astype(np.float32)


def _build_next_obs(source, seat, episode_id, start, n, h):
    # Read one extra row so the shift is available, clamped at the episode end
    # (the last row repeats) -- matches concat(o[1:], o[-1:])[start:start+n].
    length = _episode_length(source, episode_id)
    read_n = min(n + 1, length - start)
    raw = _read_raw(source, "observations", seat, episode_id, start, read_n)
    nxt = raw[1 : 1 + n]
    if nxt.shape[0] < n:
        tail = np.repeat(raw[-1:], n - nxt.shape[0], axis=0)
        nxt = np.concatenate([nxt, tail], axis=0)
    return _pad(nxt, h, 0.0).astype(np.float32)


def _build_rtg(source, seat, episode_id, start, n, h):
    # Return-to-go needs rewards from the window start to the episode end;
    # reverse-cumsum gives rtg at each position, of which we keep the window.
    length = _episode_length(source, episode_id)
    raw = _read_raw(source, "rewards", seat, episode_id, start, length - start)
    rtg = np.cumsum(raw[::-1])[::-1][:n]
    return _pad(rtg, h, 0.0).astype(np.float32)


def _build_int_field(store_key, fill):
    def build(source, seat, episode_id, start, n, h):
        raw = _read_raw(source, store_key, seat, episode_id, start, n)
        return _pad(raw, h, fill)

    return build


def _build_timesteps(source, seat, episode_id, start, n, h):
    del source, seat, episode_id  # seat- and episode-independent
    ts = np.arange(start + 1, start + n + 1, dtype=np.int32)
    return _pad(ts, h, 0)


def _build_mask(source, seat, episode_id, start, n, h):
    del source, seat, episode_id, start  # seat- and start-independent
    mask = np.zeros(h, dtype=bool)
    mask[h - n :] = True
    return mask


#: Field name -> (seat role, builder). Mirrors the field set the production
#: windowing used to expose as lazy properties, restructured as one function
#: per window instead of a class with per-field lazy promises.
_FIELDS = {
    "ego_obs": ("ego", _build_obs),
    "mate_obs": ("mate", _build_obs),
    "mate_next_obs": ("mate", _build_next_obs),
    "ego_rtg": ("ego", _build_rtg),
    "ego_actions": ("ego", _build_int_field("actions", -10)),
    "mate_actions": ("mate", _build_int_field("actions", -10)),
    "ego_avail": ("ego", _build_int_field("avail_actions", 1)),
    "mate_avail": ("mate", _build_int_field("avail_actions", 1)),
    "mate_rewards": ("mate", _build_int_field("rewards", 0)),
    "timesteps": (None, _build_timesteps),
    "mask": (None, _build_mask),
}


def window_slice(
    source, episode_id: int, t: int, h: int, fields, *, mate: int | None = None
) -> dict[str, np.ndarray]:
    """The window ``[t - h, t)`` for `fields`, left-padded if ``t < h``.

    ``source`` is the episode store -- a disk-backed
    :class:`~oaht_bench.dataset.vault.VaultReader` or an in-RAM
    :class:`~oaht_bench.dataset.schema.EpisodeBatch`; both expose ``ego_index``,
    ``num_agents``, and either ``read_slices``/``transition_offsets`` or
    ``episodes``, which is all this needs. ``fields`` is any subset of
    ``{"ego_obs", "mate_obs", "mate_next_obs", "ego_rtg", "ego_actions",
    "mate_actions", "ego_avail", "mate_avail", "mate_rewards", "timesteps",
    "mask"}``. ``mate`` disambiguates the teammate seat when there are more
    than two agents; with exactly two it is inferred the same way
    ``mate``/``ego`` are inferred everywhere else in this codebase.
    """
    ego, resolved_mate = resolve_seats(source, mate)
    start = max(0, t - h)
    n = t - start
    out = {}
    for name in fields:
        role, build = _FIELDS[name]
        seat = {"ego": ego, "mate": resolved_mate, None: None}[role]
        out[name] = build(source, seat, episode_id, start, n, h)
    return out


def enumerate_windows(batch, h: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Every ``(episode_id, t)`` pair a batch method can sample from, one row
    per window start at this ``stride`` -- the same enumeration the production
    windowing used to materialize as a class, now two plain arrays a baseline
    builds once (in ``prepare``) and samples from with its own ``np_rng``.

    ``stride`` enumerates start offsets rather than leaving them random so a
    run is reproducible from the config alone (``configs/job.py``'s own
    rationale for the field, unchanged by this rewrite). Returns
    ``(episode_ids, ts)``, both ``(num_windows,)``; sample an index into
    either array with the same draw to get a matched pair.
    """
    if hasattr(batch, "episode_lengths"):
        lengths = np.asarray(batch.episode_lengths(), dtype=np.int64)
    else:
        lengths = np.asarray([e.length for e in batch.episodes], dtype=np.int64)
    episode_ids, ts = [], []
    for episode_id, length in enumerate(lengths):
        length = int(length)
        if length == 0:
            continue
        starts = np.arange(0, max(1, length - h + 1), stride)
        ends = starts + np.minimum(h, length - starts)
        episode_ids.extend([episode_id] * len(starts))
        ts.extend(ends.tolist())
    return np.asarray(episode_ids, dtype=np.int32), np.asarray(ts, dtype=np.int32)


def teammate_episode_ids(batch, mate: int, teammate_id: int) -> np.ndarray:
    """The ids of episodes where seat ``mate`` was played by ``teammate_id`` --
    NOT the episodes themselves, just which ones (indices into ``batch``).

    Filters ONLY on the mate seat, never on which ego policy played opposite:
    sampling that needs "this teammate, paired with any ego policy" (D_epi,
    Appendix C) reads this directly; sampling that needs "this teammate,
    paired with its matching best-response ego policy" (D_step, Eq. 2) never
    calls this at all, because it reads the decoder window's own episode id.
    """
    return np.flatnonzero(batch.member_ids[:, mate] == teammate_id)
