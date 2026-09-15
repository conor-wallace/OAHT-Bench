"""The offline training :class:`Dataset`: a collected
:class:`~oaht_bench.dataset.schema.EpisodeBatch` processed into fixed-length
:class:`Windows`.

Every trajectory-view baseline consumes the same tensors — an ego stream to
predict from and a teammate stream to model — so the split happens once here
rather than inside each method. What differs between methods is what they *do*
with the teammate stream: LIAM reconstructs it from the ego embeddings, TAO
encodes it into a policy embedding and cross-attends.

JAX has no ``Dataset`` base class like PyTorch's ``torch.utils.data.Dataset`` --
the ecosystem loads data with separate libraries and keeps model code purely
functional -- so :class:`Dataset` here is a small container of our own: it holds
the loaded batch, does the windowing once, and exposes what training reads.

Return-to-go is computed rather than stored, because it is a function of the
rewards already in the artifact and storing it would let the two disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oaht_bench.dataset.schema import EpisodeBatch
from oaht_bench.dataset.vault import read_vault


@dataclass(frozen=True)
class Normalization:
    """The transform applied to a dataset, so evaluation can repeat it.

    Stored rather than recomputed: a policy trained on standardised
    observations must see standardised observations at rollout, and a target
    return expressed in raw units has to be divided by the same scale it was
    trained under.
    """

    obs_mean: np.ndarray
    obs_std: np.ndarray
    rtg_scale: float

    def apply_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean) / self.obs_std

    def apply_rtg(self, rtg):
        return rtg / self.rtg_scale


@dataclass(frozen=True)
class Windows:
    """Fixed-length windows over the ego and teammate streams.

    Leading axis is the window. ``T`` is the context length: TAO and TAGET both
    train on fixed-length fragments rather than whole episodes, and
    :mod:`oaht_bench.models.backbone` needs a static shape to jit.
    """

    #: (N, T, obs_dim) — the learner's observations.
    ego_obs: np.ndarray
    #: (N, T) — the learner's actions, the behaviour-cloning target.
    ego_actions: np.ndarray
    #: (N, T, num_actions) — which actions the environment permitted. Collection
    #: already masks with these (``collect.py`` passes them to every seat's
    #: ``get_action``), so a recorded action is always legal; carrying them here
    #: is what lets the learned policy be held to the same constraint.
    ego_avail: np.ndarray
    #: (N, T) — return-to-go for the learner, the DT conditioning signal.
    ego_rtg: np.ndarray
    #: (N, T, obs_dim) — teammate observations, LIAM's reconstruction target and
    #: the ancillary decoder's input.
    mate_obs: np.ndarray
    #: (N, T, obs_dim) — teammate observations shifted one step forward. TAO's
    #: encoder fuses ``(a_t, r_t, o_{t+1})``: the reference realises the paper's
    #: ``(a_{t-1}, r_{t-1}, o_t)`` by feeding next-observations at the same index
    #: rather than shifting the action and reward streams.
    mate_next_obs: np.ndarray
    #: (N, T) — teammate actions.
    mate_actions: np.ndarray
    #: (N, T, num_actions) — the teammate's action mask, for the reconstruction
    #: and ancillary heads that predict teammate actions.
    mate_avail: np.ndarray
    #: (N, T) — teammate rewards, fused into TAO's encoder tokens.
    mate_rewards: np.ndarray
    #: (N, T) — timestep within the episode, for the positional encoding.
    timesteps: np.ndarray
    #: (N, T) — False where the window ran past the end of its episode.
    mask: np.ndarray
    #: (N,) — which episode the fragment came from. TAO's generative term
    #: conditions on a *different trajectory* of the same teammate, which is an
    #: episode-level notion: two overlapping windows of one episode are not two
    #: trajectories.
    episode_id: np.ndarray
    #: (N,) — the ego's total return over its *whole* source episode (not the
    #: window's return-to-go, which is truncated to the window and, when
    #: ``normalize``, rescaled). Constant across every window cut from the same
    #: episode. BC's return filter needs an episode-level quantity;
    #: nothing else in this module did before it.
    episode_return: np.ndarray
    #: (N,) — which population member was the teammate. TAO's InfoNCE positives
    #: are defined by this label; it is the field §4.2 anticipated.
    teammate_id: np.ndarray
    #: The transform already applied, or ``None`` if the arrays are raw.
    norm: Normalization | None = None

    def __len__(self) -> int:
        return int(self.ego_obs.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.ego_obs.shape[-1])

    @property
    def context_length(self) -> int:
        return int(self.ego_obs.shape[1])


@dataclass(frozen=True)
class TeammateIndex:
    """Window indices grouped by teammate, and by episode within teammate.

    Built once per dataset (a :class:`Dataset` owns one). ``by_teammate`` answers
    "which windows show this teammate"; ``episodes`` answers "which episodes did
    they come from", which is what makes "a *different* trajectory" expressible --
    two overlapping windows of one episode are not two trajectories.
    """

    by_teammate: dict[int, np.ndarray]
    episodes: dict[int, np.ndarray]

    @classmethod
    def build(cls, windows: Windows) -> TeammateIndex:
        by_teammate, episodes = {}, {}
        for t in np.unique(windows.teammate_id):
            idx = np.flatnonzero(windows.teammate_id == t)
            by_teammate[int(t)] = idx
            episodes[int(t)] = windows.episode_id[idx]
        return cls(by_teammate=by_teammate, episodes=episodes)

    @property
    def teammates(self) -> list[int]:
        return sorted(self.by_teammate)

    def cross_trajectory(self, teammate: int, episode: int, rng: np.random.Generator) -> int:
        """A window of the same teammate from a *different* episode.

        Falls back to the same episode when the teammate appears in only one,
        which the reference also permits -- its ``index_gen`` can select the
        anchor itself. Recorded rather than raised because with 1-4 episodes per
        teammate the single-episode case is common, and dropping those teammates
        would bias the batch toward the well-covered ones.
        """
        pool = self.by_teammate[teammate]
        other = pool[self.episodes[teammate] != episode]
        return int(rng.choice(other if other.size else pool))


class Dataset:
    """The offline training dataset: load and process only, in the spirit of
    PyTorch's ``Dataset`` (data, not sampling).

    Constructed from the path to the data, it reads the episodes off disk, windows
    them, applies the training transforms (return-to-go, observation
    normalisation), and builds the :class:`TeammateIndex` the samplers need -- so a
    baseline runner hands over a path and gets everything training reads. It holds
    the episode-level view (``batch``: metadata, action space, the population to
    evaluate against), the windowed view (``windows``), and their ``index``.

    Drawing batches is deliberately *not* its job: the two-stage baselines read the
    whole ``windows`` pool through the samplers in
    :mod:`oaht_bench.dataset.sampler`, which need cross-window structure (a
    different trajectory of the same teammate, several positives per anchor) that a
    per-item ``__getitem__`` cannot express -- so there is no ``DataLoader`` here,
    just this container and those samplers.
    """

    def __init__(
        self,
        vault_dir: str | Path,
        *,
        context_length: int,
        stride: int = 1,
        teammate_index: int | None = None,
        normalize: bool = True,
        variant: str | None = None,
        streaming: bool = False,
    ):
        """Read the Flashbax Vault at ``vault_dir``, window it, and transform.

        ``variant`` selects the sub-directory; if omitted and the vault holds
        exactly one, that one is used (see :func:`~oaht_bench.dataset.vault.read_vault`).

        ``streaming`` swaps the eagerly-materialized :class:`Windows` for
        :class:`LazyWindows`, which builds each window on demand from the compact
        ragged episodes. Every window's ``(T, obs_dim)`` slice is ``context_length``
        copies of the raw data, so the eager path reaches tens of GiB on a real
        Hanabi dataset; the lazy path is O(dataset) and only builds the fields a
        baseline reads (BC never touches the teammate streams). Default off, so the
        existing path -- and its numerics -- are unchanged.
        """
        self.batch = read_vault(vault_dir, variant=variant)
        builder = LazyWindows if streaming else _build_windows
        self.windows = builder(
            self.batch,
            context_length=context_length,
            stride=stride,
            teammate_index=teammate_index,
            normalize=normalize,
        )
        self.index = TeammateIndex.build(self.windows)

    @property
    def obs_dim(self) -> int:
        return self.windows.obs_dim

    @property
    def context_length(self) -> int:
        return self.windows.context_length

    @property
    def norm(self) -> Normalization | None:
        return self.windows.norm

    @property
    def action_dim(self) -> int:
        """Size of the action space, from the collected availability masks."""
        return int(self.batch.episodes[0].avail_actions.shape[-1])

    @property
    def meta(self) -> dict:
        return self.batch.meta


def return_to_go(rewards: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Reverse-cumulative reward, zeroed past the end of the episode.

    ``rewards`` and ``valid`` are ``(episode, T)``. Padding contributes nothing,
    which matters because episodes are padded to a common length and a nonzero
    tail would inflate every earlier target.
    """
    r = np.asarray(rewards) * np.asarray(valid)
    return np.flip(np.cumsum(np.flip(r, axis=-1), axis=-1), axis=-1)


def _build_windows(
    batch: EpisodeBatch,
    *,
    context_length: int,
    stride: int = 1,
    teammate_index: int | None = None,
    normalize: bool = True,
) -> Windows:
    """Slice every episode into overlapping windows of ``context_length``.

    Args:
        batch: A collected dataset.
        context_length: Timesteps per window. AD found in-context RL only emerges
            with multi-episode context; for behaviour-cloning baselines like LIAM
            a within-episode window is what the specification asks for.
        stride: Step between window starts.
        normalize: Standardise observations and rescale return-to-go. The
            reference normalises observations per opponent (``OBS_NORMALIZE``)
            and divides returns by a per-environment ``REWARD_SCALE``; neither
            was applied here, which left LBF observations at std 3.3 against a
            return-to-go at std 0.155. Since the three modality embeddings are
            summed, the return token — the Decision Transformer's whole control
            signal — carried about a twentieth of an observation feature's
            magnitude.
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

    # The whole-episode ego return, the canonical form (schema.py's own
    # ``describe()`` uses it) rather than re-derived from ``rtg``.
    episode_returns = batch.episode_returns()[:, ego]
    T = context_length
    ego_o, ego_a, ego_g = [], [], []
    mate_o, mate_no, mate_a, mate_r = [], [], [], []
    ego_av, mate_av = [], []
    steps, masks, ids, eps, ep_rets = [], [], [], [], []

    def pad(arr, width=T, fill=0):
        """Left-pad, the Decision Transformer convention the reference inherits:
        the most recent timestep is always last, so a short window and a full one
        agree on where "now" is."""
        if arr.shape[0] == width:
            return arr
        head = np.full((width - arr.shape[0], *arr.shape[1:]), fill, dtype=arr.dtype)
        return np.concatenate([head, arr], axis=0)

    for ep, episode in enumerate(batch.episodes):
        ego_obs = episode.obs[ego]  # (T_ep, obs_dim); episodes are already ragged
        mate_obs = episode.obs[teammate_index]
        length = ego_obs.shape[0]
        if length == 0:
            continue
        # No valid mask: T_ep is the real length. Return-to-go over the whole
        # episode, and teammate observations shifted one step (the final step
        # repeats, which the window mask covers since a window never ends past the
        # episode).
        rtg = return_to_go(episode.rewards[ego][None], np.ones((1, length), dtype=bool))[0]
        next_obs = np.concatenate([mate_obs[1:], mate_obs[-1:]], axis=0)
        ego_act = episode.actions[ego]
        mate_act = episode.actions[teammate_index]
        mate_rew = episode.rewards[teammate_index]
        ego_avail = episode.avail_actions[ego]
        mate_avail = episode.avail_actions[teammate_index]

        for start in range(0, max(1, length - T + 1), stride):
            sl = slice(start, start + T)
            n = min(T, length - start)
            if n <= 0:
                continue
            ego_o.append(pad(ego_obs[sl][:n]))
            # Reference pads actions with -10, an out-of-range sentinel, so the
            # embedding of a padded action cannot be confused with action 0.
            ego_a.append(pad(ego_act[sl][:n], fill=-10))
            ego_g.append(pad(rtg[sl][:n]))
            mate_o.append(pad(mate_obs[sl][:n]))
            mate_no.append(pad(next_obs[sl][:n]))
            mate_a.append(pad(mate_act[sl][:n], fill=-10))
            mate_r.append(pad(mate_rew[sl][:n]))
            # 1-indexed, 0 reserved for padding (reference utils.py:115-118).
            # Pad the mask with ones: a padded step has no legal action either
            # way, and zeros would make the masked logits all -1e10.
            ego_av.append(pad(ego_avail[sl][:n], fill=1))
            mate_av.append(pad(mate_avail[sl][:n], fill=1))
            steps.append(pad(np.arange(start + 1, start + n + 1)))
            m = np.zeros(T, dtype=bool)
            m[T - n :] = True
            masks.append(m)
            ids.append(batch.member_ids[ep, teammate_index])
            eps.append(ep)
            ep_rets.append(episode_returns[ep])

    stacked_ego_obs = np.stack(ego_o).astype(np.float32)
    stacked_mate_obs = np.stack(mate_o).astype(np.float32)
    stacked_mate_next = np.stack(mate_no).astype(np.float32)
    stacked_rtg = np.stack(ego_g).astype(np.float32)
    stacked_mask = np.stack(masks)

    norm = None
    if normalize:
        valid = stacked_ego_obs[stacked_mask]
        obs_mean = valid.mean(axis=0)
        # Guard constant features: LBF observations include dimensions that
        # never vary, and dividing by their zero std produces NaN.
        obs_std = np.maximum(valid.std(axis=0), 1e-6)
        rtg_valid = stacked_rtg[stacked_mask]
        rtg_scale = float(max(rtg_valid.std(), 1e-6))
        norm = Normalization(obs_mean=obs_mean, obs_std=obs_std, rtg_scale=rtg_scale)
        stacked_ego_obs = norm.apply_obs(stacked_ego_obs)
        stacked_mate_obs = norm.apply_obs(stacked_mate_obs)
        stacked_mate_next = norm.apply_obs(stacked_mate_next)
        stacked_rtg = norm.apply_rtg(stacked_rtg)

    return Windows(
        norm=norm,
        ego_obs=stacked_ego_obs,
        ego_actions=np.stack(ego_a).astype(np.int32),
        ego_avail=np.stack(ego_av).astype(np.float32),
        ego_rtg=stacked_rtg,
        mate_obs=stacked_mate_obs,
        mate_next_obs=stacked_mate_next,
        mate_actions=np.stack(mate_a).astype(np.int32),
        mate_avail=np.stack(mate_av).astype(np.float32),
        mate_rewards=np.stack(mate_r).astype(np.float32),
        timesteps=np.stack(steps).astype(np.int32),
        mask=stacked_mask,
        episode_id=np.asarray(eps, dtype=np.int32),
        episode_return=np.asarray(ep_rets, dtype=np.float32),
        teammate_id=np.asarray(ids, dtype=np.int32),
    )


class _LazyField:
    """A window field materialized only for the indices asked for.

    Mirrors ``ndarray[idx]`` fancy-indexing (0-D, 1-D, and the 2-D context indices
    stage 2 uses) but calls ``build`` to construct just those windows.
    """

    __slots__ = ("_build",)

    def __init__(self, build):
        self._build = build  # build(flat_idx: 1-D int array) -> (M, T, ...) ndarray

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        out = self._build(idx.reshape(-1))
        return out.reshape(*idx.shape, *out.shape[1:])


class LazyWindows:
    """A :class:`Windows` work-alike that builds each window on demand.

    :func:`_build_windows` stacks every window's ``(T, obs_dim)`` slice up front --
    ``context_length`` copies of the raw data -- which reaches tens of GiB on a real
    Hanabi dataset (25k episodes x 80 x 658 floats, thrice, is ~16 GiB of
    observations alone). This keeps the compact ragged episodes and constructs only
    the windows a minibatch indexes, so host memory is O(dataset) rather than
    O(dataset x context). Fields a baseline never reads are never built -- BC touches
    only the ego streams, so the teammate windows cost nothing.

    The field API matches :class:`Windows`: per-window metadata (``episode_id``,
    ``teammate_id``, ``episode_return``) are plain arrays the samplers index
    element-wise; per-position fields (``ego_obs`` ...) are :class:`_LazyField`
    proxies supporting the same ``[idx]`` indexing.

    Normalization is computed streaming over the raw ego transitions. That equals
    :func:`_build_windows`'s per-window-position statistics *exactly* when every
    transition belongs to a single window -- episodes no longer than the context,
    which is the regime this path exists for (long context on a partially-observable
    task). With overlapping windows the two differ only by the overlap weighting.
    """

    def __init__(self, batch, *, context_length, stride=1, teammate_index=None, normalize=True):
        ego = batch.ego_index
        if teammate_index is None:
            others = [i for i in range(batch.num_agents) if i != ego]
            if len(others) != 1:
                raise ValueError(
                    f"{batch.num_agents} agents, so 'the teammate' is ambiguous; pass "
                    f"teammate_index explicitly."
                )
            teammate_index = others[0]
        self._ego, self._mate = ego, teammate_index
        self._T = int(context_length)
        self._episodes = batch.episodes

        # Return-to-go per episode, computed once (the one derived per-step stream).
        self._rtg = []
        for e in self._episodes:
            length = e.obs[ego].shape[0]
            self._rtg.append(
                return_to_go(e.rewards[ego][None], np.ones((1, length), dtype=bool))[0]
                if length
                else np.zeros((0,), np.float32)
            )

        # Enumerate windows exactly as _build_windows does.
        win_ep, win_start, win_n, ids, team, ep_ret = [], [], [], [], [], []
        episode_returns = batch.episode_returns()[:, ego]
        T = self._T
        for ep, e in enumerate(self._episodes):
            length = e.obs[ego].shape[0]
            if length == 0:
                continue
            for start in range(0, max(1, length - T + 1), stride):
                n = min(T, length - start)
                if n <= 0:
                    continue
                win_ep.append(ep)
                win_start.append(start)
                win_n.append(n)
                ids.append(ep)
                team.append(int(batch.member_ids[ep, teammate_index]))
                ep_ret.append(episode_returns[ep])
        self._win_ep = np.asarray(win_ep, np.int32)
        self._win_start = np.asarray(win_start, np.int32)
        self._win_n = np.asarray(win_n, np.int32)
        self.episode_id = np.asarray(ids, np.int32)
        self.teammate_id = np.asarray(team, np.int32)
        self.episode_return = np.asarray(ep_ret, np.float32)
        self._obs_dim = int(self._episodes[0].obs[ego].shape[-1]) if len(self._episodes) else 0

        self.norm = self._streaming_norm() if normalize else None

    def _streaming_norm(self):
        ego = self._ego
        d = self._obs_dim
        s = np.zeros(d, np.float64)
        ss = np.zeros(d, np.float64)
        count = 0
        rtgs = []
        for ep, e in enumerate(self._episodes):
            o = np.asarray(e.obs[ego], np.float64)
            if o.shape[0] == 0:
                continue
            s += o.sum(0)
            ss += (o * o).sum(0)
            count += o.shape[0]
            rtgs.append(self._rtg[ep])
        count = max(count, 1)
        mean = s / count
        std = np.maximum(np.sqrt(np.maximum(ss / count - mean * mean, 0.0)), 1e-6)
        rtg_all = np.concatenate(rtgs) if rtgs else np.zeros((1,), np.float32)
        return Normalization(
            obs_mean=mean.astype(np.float32),
            obs_std=std.astype(np.float32),
            rtg_scale=float(max(rtg_all.std(), 1e-6)),
        )

    def __len__(self):
        return int(self._win_ep.shape[0])

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def context_length(self):
        return self._T

    def _pad(self, arr, fill):
        n = arr.shape[0]
        if n == self._T:
            return arr
        head = np.full((self._T - n, *arr.shape[1:]), fill, dtype=arr.dtype)
        return np.concatenate([head, arr], axis=0)

    def _gather(self, flat_idx, one):
        return np.stack(
            [
                one(int(self._win_ep[w]), int(self._win_start[w]), int(self._win_n[w]))
                for w in flat_idx
            ],
            axis=0,
        )

    def _obs_field(self, seat, *, next_obs=False):
        def one(ep, s, n):
            o = np.asarray(self._episodes[ep].obs[seat], np.float32)
            if next_obs:
                o = np.concatenate([o[1:], o[-1:]], axis=0)
            p = self._pad(o[s : s + n], 0.0)
            return (p - self.norm.obs_mean) / self.norm.obs_std if self.norm else p

        return _LazyField(lambda fi: self._gather(fi, one))

    def _int_field(self, stream_of, fill):
        def one(ep, s, n):
            return self._pad(np.asarray(stream_of(ep))[s : s + n], fill)

        return _LazyField(lambda fi: self._gather(fi, one))

    @property
    def ego_obs(self):
        return self._obs_field(self._ego)

    @property
    def mate_obs(self):
        return self._obs_field(self._mate)

    @property
    def mate_next_obs(self):
        return self._obs_field(self._mate, next_obs=True)

    @property
    def ego_rtg(self):
        def one(ep, s, n):
            p = self._pad(self._rtg[ep][s : s + n], 0.0)
            return p / self.norm.rtg_scale if self.norm else p

        return _LazyField(lambda fi: self._gather(fi, one))

    @property
    def ego_actions(self):
        return self._int_field(lambda ep: self._episodes[ep].actions[self._ego], -10)

    @property
    def mate_actions(self):
        return self._int_field(lambda ep: self._episodes[ep].actions[self._mate], -10)

    @property
    def ego_avail(self):
        return self._int_field(lambda ep: self._episodes[ep].avail_actions[self._ego], 1)

    @property
    def mate_avail(self):
        return self._int_field(lambda ep: self._episodes[ep].avail_actions[self._mate], 1)

    @property
    def mate_rewards(self):
        return self._int_field(lambda ep: self._episodes[ep].rewards[self._mate], 0)

    @property
    def timesteps(self):
        def one(ep, s, n):
            return self._pad(np.arange(s + 1, s + n + 1, dtype=np.int32), 0)

        return _LazyField(lambda fi: self._gather(fi, one))

    @property
    def mask(self):
        def one(ep, s, n):
            m = np.zeros(self._T, dtype=bool)
            m[self._T - n :] = True
            return m

        return _LazyField(lambda fi: self._gather(fi, one))
