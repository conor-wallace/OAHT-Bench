"""Window many episode slices at once -- the one place this is implemented.

Each baseline builds its own training batch (see ``offline/bc.py``,
``offline/liam.py``, ``offline/tao.py``, ``offline/omis.py``): assemble the
``(episode_id, t)`` pairs a sampling scheme calls for, then one call to
:func:`window_slice_batch`, composed however that baseline's own paper
describes its batch. This module holds only the primitives every one of
those call sites needs and that aren't one-liners on data the episode store
(:class:`~oaht_bench.dataset.schema.EpisodeBatch` or
:class:`~oaht_bench.dataset.vault.VaultReader`) already exposes directly.

``window_slice_batch`` always takes arrays, even for a batch of one --
there is no separate single-window function. The reason is
``VaultReader.read_slices``'s own docstring: it issues every offset as a
tensorstore future *before* awaiting any of them, so many windows are
fetched concurrently rather than one blocking read at a time, which is "the
single biggest lever on training throughput off a disk-backed vault." A
per-window call defeats that -- not because concurrency and the OS page
cache stop working, but because the *fixed* dispatch cost of entering
``read_slices`` (building the index transform, scheduling the async read)
is paid once per *call*, not once per *offset*. One call with ``N`` offsets
pays that fixed cost once; ``N`` calls with one offset each pay it ``N``
times and get none of the overlap. A baseline that looped this function
once per row (an earlier version of this file did) turned an O(1)-per-step
sampling cost into an O(batch_size) one -- and for OMIS, whose ``D_epi``
fans out ``C`` more reads per row, that loop compounded into an O(batch_size
* C) one. Keeping only the batched form closes the door on reintroducing
that loop by accident.

``window_slice_batch`` returns raw, un-normalized arrays -- observation and
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
    dataset.Dataset` calls it once at construction and
    :func:`window_slice_batch` calls it again per call (cheap -- no state to
    share) so it works from a bare source too, not just through ``Dataset``.
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


def _episode_lengths(source, episode_ids: np.ndarray) -> np.ndarray:
    """``(length of episode_ids[i])_i`` -- one lookup into the store's own
    (already-resident) length table, not a per-row call."""
    if hasattr(source, "episode_lengths"):
        lengths = np.asarray(source.episode_lengths())
    else:
        lengths = np.asarray([e.length for e in source.episodes])
    return lengths[episode_ids]


def _read_raw_batch(
    source, store_key: str, seat: int, episode_ids: np.ndarray, starts: np.ndarray, ns: np.ndarray
) -> list[np.ndarray]:
    """The unpadded ``[start, start + n)`` slice of one field/seat, for every
    row -- issued as ONE concurrent, awaited-together read when the source is
    disk-backed (:meth:`~oaht_bench.dataset.vault.VaultReader.read_slices`),
    else sliced directly from a resident ``EpisodeBatch`` (cheap either way;
    looping there costs no I/O)."""
    if hasattr(source, "read_slices"):
        offsets = source.transition_offsets[episode_ids] + starts
        return source.read_slices(store_key, seat, offsets, ns)
    attr = _STORE_ATTR[store_key]
    return [
        np.asarray(getattr(source.episodes[int(ep)], attr)[seat][int(s) : int(s) + int(n)])
        for ep, s, n in zip(episode_ids, starts, ns, strict=True)
    ]


def _pad(arr: np.ndarray, h: int, fill) -> np.ndarray:
    n = arr.shape[0]
    if n == h:
        return arr
    head = np.full((h - n, *arr.shape[1:]), fill, dtype=arr.dtype)
    return np.concatenate([head, arr], axis=0)


def _pad_stack(slices: list[np.ndarray], h: int, fill) -> np.ndarray:
    return np.stack([_pad(sl, h, fill) for sl in slices], axis=0)


def _build_obs(source, seat, episode_ids, starts, ns, h):
    raw = _read_raw_batch(source, "observations", seat, episode_ids, starts, ns)
    return _pad_stack(raw, h, 0.0).astype(np.float32)


def _build_next_obs(source, seat, episode_ids, starts, ns, h):
    # Read one extra row so the shift is available, clamped at the episode end
    # (the last row repeats) -- matches concat(o[1:], o[-1:])[start:start+n].
    # read_n varies per row but that's still one read_slices call: counts is
    # just another per-row array, the same as offsets.
    lengths = _episode_lengths(source, episode_ids)
    read_ns = np.minimum(ns + 1, lengths - starts)
    raw = _read_raw_batch(source, "observations", seat, episode_ids, starts, read_ns)
    slices = []
    for r, n in zip(raw, ns, strict=True):
        n = int(n)
        nxt = r[1 : 1 + n]
        if nxt.shape[0] < n:
            tail = np.repeat(r[-1:], n - nxt.shape[0], axis=0)
            nxt = np.concatenate([nxt, tail], axis=0)
        slices.append(nxt)
    return _pad_stack(slices, h, 0.0).astype(np.float32)


def _build_rtg(source, seat, episode_ids, starts, ns, h):
    # Return-to-go needs rewards from the window start to the episode end;
    # reverse-cumsum gives rtg at each position, of which we keep the window.
    lengths = _episode_lengths(source, episode_ids)
    raw = _read_raw_batch(source, "rewards", seat, episode_ids, starts, lengths - starts)
    slices = [np.cumsum(r[::-1])[::-1][: int(n)] for r, n in zip(raw, ns, strict=True)]
    return _pad_stack(slices, h, 0.0).astype(np.float32)


def _build_int_field(store_key, fill):
    def build(source, seat, episode_ids, starts, ns, h):
        raw = _read_raw_batch(source, store_key, seat, episode_ids, starts, ns)
        return _pad_stack(raw, h, fill)

    return build


def _build_timesteps(source, seat, episode_ids, starts, ns, h):
    del source, seat, episode_ids  # seat- and episode-independent
    return np.stack(
        [
            _pad(np.arange(int(s) + 1, int(s) + int(n) + 1, dtype=np.int32), h, 0)
            for s, n in zip(starts, ns, strict=True)
        ],
        axis=0,
    )


def _build_mask(source, seat, episode_ids, starts, ns, h):
    del source, seat, episode_ids, starts  # seat- and start-independent
    out = np.zeros((len(ns), h), dtype=bool)
    for row, n in enumerate(ns):
        out[row, h - int(n) :] = True
    return out


#: Field name -> (seat role, builder). Mirrors the field set the production
#: windowing used to expose as lazy properties, restructured as plain
#: functions over whole batches instead of a class with per-field lazy
#: promises.
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


def window_slice_batch(
    source, episode_ids, ts, h: int, fields, *, mate: int | None = None
) -> dict[str, np.ndarray]:
    """The window ``[t - h, t)`` for `fields`, for every ``(episode_id, t)``
    pair in ``episode_ids``/``ts`` at once -- left-padded wherever ``t < h``.
    Returns ``(len(episode_ids), h, ...)``-shaped arrays, one call per field
    regardless of how many rows (see the module docstring for why that
    matters: a single read per row here would undo ``read_slices``'s own
    concurrency).

    ``source`` is the episode store -- a disk-backed
    :class:`~oaht_bench.dataset.vault.VaultReader` or an in-RAM
    :class:`~oaht_bench.dataset.schema.EpisodeBatch`; both expose ``ego_index``,
    ``num_agents``, and either ``read_slices``/``transition_offsets`` or
    ``episodes``, which is all this needs. ``fields`` is any subset of
    ``{"ego_obs", "mate_obs", "mate_next_obs", "ego_rtg", "ego_actions",
    "mate_actions", "ego_avail", "mate_avail", "mate_rewards", "timesteps",
    "mask"}``. ``mate`` disambiguates the teammate seat when there are more
    than two agents; with exactly two it is inferred the same way
    ``mate``/``ego`` are inferred everywhere else in this codebase. A batch
    of one row is just ``episode_ids=[episode_id], ts=[t]`` -- there is no
    separate scalar form.
    """
    ego, resolved_mate = resolve_seats(source, mate)
    episode_ids = np.asarray(episode_ids)
    ts = np.asarray(ts)
    starts = np.maximum(0, ts - h)
    ns = ts - starts
    out = {}
    for name in fields:
        role, build = _FIELDS[name]
        seat = {"ego": ego, "mate": resolved_mate, None: None}[role]
        out[name] = build(source, seat, episode_ids, starts, ns, h)
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


def identity_matched_episode_ids(batch, ego: int, mate: int) -> np.ndarray:
    """Ids of episodes where the ego seat IS the mate seat's own identity --
    roster-index equality, not a quality/epsilon threshold.

    Every baseline that clones the ego's actions as a "near-optimal response"
    target (BC, LIAM, MeLIBA's policy, OMIS's D_step, TAO's stage 2) needs its
    decoder/BC window restricted to these episodes, or a mismatched partner's
    suboptimal actions pollute the target the moment a vault isn't 100%
    expert (``mismatch_fraction`` in pooled collection). The opposite-pool
    functions (:func:`teammate_episode_ids`, used by D_epi/GetOffD) stay
    unrestricted by design -- diversity across any ego is the point there --
    so this is deliberately a separate function, not a parameter on that one.
    """
    return np.flatnonzero(batch.member_ids[:, ego] == batch.member_ids[:, mate])
