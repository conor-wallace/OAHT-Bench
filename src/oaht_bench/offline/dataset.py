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
    :mod:`oaht_bench.offline.backbone` needs a static shape to jit.
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

    rtg = return_to_go(batch.rewards[:, ego], batch.valid)
    T = context_length
    ego_o, ego_a, ego_g = [], [], []
    mate_o, mate_no, mate_a, mate_r = [], [], [], []
    ego_av, mate_av = [], []
    steps, masks, ids, eps = [], [], [], []
    # Teammate observations shifted one step; the final step repeats, which the
    # mask covers since a window never ends on a step the episode did not take.
    next_obs = np.concatenate(
        [batch.obs[:, teammate_index, 1:], batch.obs[:, teammate_index, -1:]], axis=1
    )

    for ep in range(batch.num_episodes):
        length = int(batch.valid[ep].sum())
        if length == 0:
            continue
        for start in range(0, max(1, length - T + 1), stride):
            sl = slice(start, start + T)
            n = min(T, length - start)
            if n <= 0:
                continue

            def pad(arr, width=T, fill=0):
                """Left-pad, the Decision Transformer convention the reference
                inherits: the most recent timestep is always last, so a short
                window and a full one agree on where "now" is."""
                if arr.shape[0] == width:
                    return arr
                head = np.full((width - arr.shape[0], *arr.shape[1:]), fill, dtype=arr.dtype)
                return np.concatenate([head, arr], axis=0)

            ego_o.append(pad(batch.obs[ep, ego][sl][:n]))
            # Reference pads actions with -10, an out-of-range sentinel, so the
            # embedding of a padded action cannot be confused with action 0.
            ego_a.append(pad(batch.actions[ep, ego][sl][:n], fill=-10))
            ego_g.append(pad(rtg[ep][sl][:n]))
            mate_o.append(pad(batch.obs[ep, teammate_index][sl][:n]))
            mate_no.append(pad(next_obs[ep][sl][:n]))
            mate_a.append(pad(batch.actions[ep, teammate_index][sl][:n], fill=-10))
            mate_r.append(pad(batch.rewards[ep, teammate_index][sl][:n]))
            # 1-indexed, 0 reserved for padding (reference utils.py:115-118).
            # Pad the mask with ones: a padded step has no legal action either
            # way, and zeros would make the masked logits all -1e10.
            ego_av.append(pad(batch.avail_actions[ep, ego][sl][:n], fill=1))
            mate_av.append(pad(batch.avail_actions[ep, teammate_index][sl][:n], fill=1))
            steps.append(pad(np.arange(start + 1, start + n + 1)))
            m = np.zeros(T, dtype=bool)
            m[T - n :] = True
            masks.append(m)
            ids.append(batch.member_ids[ep, teammate_index])
            eps.append(ep)

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
        teammate_id=np.asarray(ids, dtype=np.int32),
    )
