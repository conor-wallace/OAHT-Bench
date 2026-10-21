"""Cross-episode in-context deployment eval for opponent-trajectory baselines.

TAO and OMIS both adapt to a teammate by accumulating its trajectories
*across episodes* into an Opponent Context Window (OCW) and re-conditioning on it --
unlike BC/LIAM/MeLIBA, whose context is within-episode and resets each episode. So
these two need a different eval shape: a teammate's episodes run **sequentially**, an
OCW of the teammate's last ``C`` trajectories grows after each episode, and the ego's
context is re-encoded from it before the next one (TAO: OPE -> z^-1; OMIS: D_epi's raw
state/action pairs, no separate encoder). We report the **per-episode returns** (the
adaptation curve) and their mean, so a within-episode baseline reads as a flat line and
TAO/OMIS should climb as the OCW fills.

Faithful to TAO's OOM assumption (paper App. J): the opponent's full trajectory is
observable after each episode. The OCW holds the teammate's own ``(o^-1, a^-1, r^-1)``
stream, collected from the rollout (the eval harness controls both seats). OMIS's
``D_epi`` reuses this exact OCW (:meth:`~oaht_bench.models.omis_agent.OmisAgent.encode_ocw`);
its within-episode ``D_step`` is additive, accumulated here in
:func:`rollout_one_episode` (not the OCW -- D_step resets every episode, D_epi does
not) and threaded through :meth:`~oaht_bench.models.omis_agent.OmisAgent.get_action_with_mate_history`.
Decision-time search stays out of scope (``omis_search``, :mod:`oaht_bench.offline.omis`).

The OCW logic lives entirely here: :class:`OpponentContextWindow` accumulates the
teammate stream, and the eval re-encodes it via the agent's ``encode_ocw``, merging the
result into ``params["stage2"]`` (:func:`_context_params`). TAO's ``act`` reads
``context``/``context_mask`` straight off ``stage2``, so its forward is unchanged by
this; OMIS's ``act`` takes the equivalent ``epi_state``/``epi_mate_actions``/``epi_mask``
as explicit kwargs instead (see that method's own docstring for why), so
``get_action_with_mate_history`` reads them back out of ``stage2`` itself before
calling ``act``.
"""

from __future__ import annotations

import numpy as np


class OpponentContextWindow:
    """A rolling buffer of the last ``C`` teammate trajectory fragments, padded to ``T``.

    Each appended fragment is one episode's teammate stream: ``mate_next_obs`` ``(L, d)``,
    ``mate_actions`` ``(L,)``, ``mate_rewards`` ``(L,)``, ``timesteps`` ``(L,)`` (``L`` <=
    ``T``). :meth:`arrays` stacks the most recent ``C`` into fixed ``(C, T, ...)`` tensors
    for the encoder, front-padding with empty (all-``mask``-zero) slots so the shape is
    constant regardless of how full the OCW is. An empty OCW (episode 1 vs a teammate)
    yields all-zero masks, so the encoder/cross-attention sees no valid context.
    """

    def __init__(self, capacity: int, horizon: int, obs_dim: int):
        self.C = int(capacity)
        self.T = int(horizon)
        self.obs_dim = int(obs_dim)
        self._frags: list[dict] = []

    def reset(self) -> None:
        """Empty the window -- called when the teammate (opponent) switches."""
        self._frags = []

    def append(self, mate_next_obs, mate_actions, mate_rewards, timesteps) -> None:
        n = int(np.asarray(mate_actions).shape[0])
        self._frags.append(
            {
                "mate_next_obs": np.asarray(mate_next_obs, np.float32)[: self.T],
                "mate_actions": np.asarray(mate_actions, np.int32)[: self.T],
                "mate_rewards": np.asarray(mate_rewards, np.float32)[: self.T],
                "timesteps": np.asarray(timesteps, np.int32)[: self.T],
                "len": min(n, self.T),
            }
        )
        self._frags = self._frags[-self.C :]

    def __len__(self) -> int:
        return len(self._frags)

    def arrays(self):
        """``(mate_next_obs, mate_actions, mate_rewards, timesteps, mask)`` of shape
        ``(C, T, ...)`` -- the encoder input. Slots are filled from the back so the most
        recent trajectory is last; unfilled slots stay zero with ``mask == 0``."""
        C, T, d = self.C, self.T, self.obs_dim
        no = np.zeros((C, T, d), np.float32)
        ac = np.zeros((C, T), np.int32)
        rw = np.zeros((C, T), np.float32)
        ts = np.zeros((C, T), np.int32)
        mk = np.zeros((C, T), np.float32)
        frags = self._frags[-C:]
        base = C - len(frags)  # front-pad empty slots
        for i, f in enumerate(frags):
            slot = base + i
            length = f["len"]
            no[slot, :length] = f["mate_next_obs"][:length]
            ac[slot, :length] = f["mate_actions"][:length]
            rw[slot, :length] = f["mate_rewards"][:length]
            ts[slot, :length] = f["timesteps"][:length]
            mk[slot, :length] = 1.0
        return no, ac, rw, ts, mk


def _context_params(agent, params, ocw: OpponentContextWindow):
    """A copy of ``params`` with the OCW's re-encoding merged into ``stage2``.

    ``encode_ocw`` returns a dict rather than a fixed tuple because what it
    produces differs by agent: TAO's is a pre-encoded ``(context,
    context_mask)`` pair the policy cross-attends to; OMIS's ``D_epi`` has no
    separate encoder (Sec. 4.1/Fig. 1 is one shared backbone), so it is the
    three raw ``(epi_state, epi_mate_actions, epi_mask)`` arrays the backbone
    reads directly. Either way, swapping the result into ``stage2`` is how a
    static-context method becomes an in-context one without touching the
    forward pass -- the agent's ``act`` just reads whatever sits there.
    """
    no, ac, rw, ts, mk = ocw.arrays()
    stage2 = {**params["stage2"], **agent.encode_ocw(params, no, ac, rw, mk, ts)}
    return {**params, "stage2": stage2}


def rollout_one_episode(
    rng, env, ego_agent, params_ep, mate_params, mate_policy, *, max_episode_steps, ego_index=0
):
    """One episode, Python-orchestrated, returning ``(ego_return, teammate_stream, mate_avail)``.

    Mirrors :func:`~oaht_bench.common.run_episodes.run_single_episode`'s per-step
    contract (the ``aux_obs`` triple, ``hstate`` threading, turn-based action masking,
    ``LogWrapper``'s per-agent episode return) but runs in a Python loop so it can also
    collect the teammate's ``(o⁻¹, a⁻¹, r⁻¹)`` stream, which the scanned rollout does not
    expose. ``o⁻¹`` is the teammate's observation *after* the step, matching the OPE's
    ``mate_next_obs`` convention. Un-jitted stepping is fine at eval scale. ``mate_avail``
    is the teammate's per-step ``avail_actions``, ``(L, action_dim)`` -- not part of
    ``teammate_stream`` (which gets unpacked positionally into
    :meth:`OpponentContextWindow.append`) because it is only for
    :func:`evaluate_incontext`'s ancillary-decoder probe, not the OCW.
    """
    import jax
    import jax.numpy as jnp

    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    done = {k: jnp.zeros((1,), dtype=bool) for k in list(env.agents) + ["__all__"]}
    act_onehot = {k: jnp.zeros(env.action_space(env.agents[i]).n) for i, k in enumerate(env.agents)}
    reward = {k: jnp.zeros(1) for k in env.agents}
    hstate_0 = ego_agent.init_hstate(1, aux_info={"agent_id": 0})
    hstate_1 = mate_policy.init_hstate(1, aux_info={"agent_id": 1})
    # OMIS's D_step: the opponent's own action history, within this episode only
    # (resets every episode, unlike D_epi's cross-episode OCW) -- additive to the
    # shared ContextWindow rather than a field on it (see
    # OmisAgent.get_action_with_mate_history's own docstring for why).
    uses_d_step = hasattr(ego_agent, "get_action_with_mate_history")

    ego_return = 0.0
    mate_no, mate_a, mate_r, mate_ts, mate_avail = [], [], [], [], []
    for t in range(int(max_episode_steps)):
        if bool(done["__all__"]):
            break
        avail = jax.lax.stop_gradient(env.get_avail_actions(env_state))
        avail_0 = avail["agent_0"].astype(jnp.float32)
        avail_1 = avail["agent_1"].astype(jnp.float32)
        joint_act_onehot = jnp.concatenate(
            (act_onehot["agent_0"].reshape(1, 1, -1), act_onehot["agent_1"].reshape(1, 1, -1)),
            axis=-1,
        )
        rng, a0, a1, sr = jax.random.split(rng, 4)
        if uses_d_step:
            # a^{-1}_{t-1}, ..., left-padded to K -- mate_a holds exactly the
            # opponent's actions through step t-1 at this point (this step's is
            # appended below, after both agents have acted).
            K = ego_agent.context_length
            history = mate_a[-K:]
            # -10: the same "no action yet" sentinel training's `_shift_right`
            # uses (offline/omis.py) -- not 0, which is a real action id.
            window = [-10] * (K - len(history)) + history
            act_0, hstate_0 = ego_agent.get_action_with_mate_history(
                params=params_ep,
                obs=obs["agent_0"].reshape(1, 1, -1),
                done=done["agent_0"].reshape(1, 1),
                avail_actions=avail_0,
                hstate=hstate_0,
                rng=a0,
                mate_action_window=window,
                reward=reward["agent_0"].reshape(1, 1, -1),
            )
        else:
            act_0, hstate_0 = ego_agent.get_action(
                params=params_ep,
                obs=obs["agent_0"].reshape(1, 1, -1),
                done=done["agent_0"].reshape(1, 1),
                avail_actions=avail_0,
                hstate=hstate_0,
                rng=a0,
                aux_obs=(
                    act_onehot["agent_0"].reshape(1, 1, -1),
                    joint_act_onehot,
                    reward["agent_0"].reshape(1, 1, -1),
                ),
                env_state=env_state,
                test_mode=False,
                reward=reward["agent_0"].reshape(1, 1, -1),
            )
        act_1, hstate_1 = mate_policy.get_action(
            params=mate_params,
            obs=obs["agent_1"].reshape(1, 1, -1),
            done=done["agent_1"].reshape(1, 1),
            avail_actions=avail_1,
            hstate=hstate_1,
            rng=a1,
            env_state=env_state,
            test_mode=False,
        )
        env_act = {"agent_0": act_0.squeeze(), "agent_1": act_1.squeeze()}
        act_onehot = {k: jax.nn.one_hot(env_act[k], env.action_space(k).n) for k in env.agents}
        obs, env_state, reward, done, info = env.step(sr, env_state, env_act)
        mate_no.append(np.asarray(obs["agent_1"]).reshape(-1))
        mate_a.append(int(np.asarray(env_act["agent_1"])))
        mate_r.append(float(np.asarray(reward["agent_1"]).reshape(-1)[0]))
        mate_ts.append(t)
        mate_avail.append(np.asarray(avail_1).reshape(-1))
        if bool(done["__all__"]):
            ego_return = float(np.asarray(info["returned_episode_returns"]).reshape(-1)[ego_index])
    teammate_stream = (
        np.stack(mate_no) if mate_no else np.zeros((0, 0)),
        np.asarray(mate_a),
        np.asarray(mate_r),
        np.asarray(mate_ts),
    )
    mate_avail_arr = np.stack(mate_avail) if mate_avail else np.zeros((0, 0))
    return ego_return, teammate_stream, mate_avail_arr


def _ancillary_mate_action_acc(agent, params, context, mate_obs, mate_actions, mate_avail):
    """TAO's ancillary decoder, scored against this episode's REAL teammate actions.

    A DIFFERENT information set from LIAM/MeLIBA/OMIS's ``mate_action_logits``
    (docs/tuning_record.md): TAO already assumes access to the opponent's own
    trajectory (its whole premise), so this checks whether the pooled OCW
    summary ``z̄⁻¹`` -- built from episodes *before* this one, the same quantity
    the ego's cross-attention conditions on -- captures enough of the teammate to
    explain a NEW episode of its behaviour. That is exactly Stage 1's
    cross-episode ``generative_accuracy`` task (``embedding_loss``,
    ``offline/tao.py``), just measured online instead of on dataset windows.
    Returns ``(pred, true)`` int arrays, or ``(None, None)`` for an empty episode.
    """
    import jax.numpy as jnp

    from oaht_bench.models.masking import mask_logits
    from oaht_bench.models.tao_agent import OpponentPolicyEncoder

    if mate_obs.shape[0] == 0:
        return None, None
    z_bar = OpponentPolicyEncoder.pool(context)
    logits = agent.decoder.apply(params["stage1"]["decoder"], jnp.asarray(mate_obs)[None], z_bar)
    logits = mask_logits(logits, jnp.asarray(mate_avail)[None])
    pred = np.asarray(jnp.argmax(logits[0], axis=-1))
    return pred, mate_actions


def evaluate_incontext(
    agent,
    params,
    env,
    teammates,
    *,
    max_episode_steps: int,
    num_episodes: int,
    ocw_size: int,
    obs_dim: int,
    rng,
    target_returns: dict,
    ego_index: int = 0,
):
    """Sequential-episode, OCW-accumulating eval for an opponent-trajectory agent.

    For each teammate: reset the OCW, then for each episode re-encode the ego's context
    from the current OCW (swapped into ``stage2["context"]``), roll one episode, and
    append the teammate's stream. Returns ``(per_teammate_mean, per_teammate_curve,
    per_teammate_ancillary_acc, ancillary_floor, per_teammate_ancillary_curve)`` -- the
    last two ``None``/empty unless ``agent`` has a decoder (TAO; see
    :func:`_ancillary_mate_action_acc`).

    ``per_teammate_ancillary_curve`` is the direct diagnostic for "is in-context
    adaptation doing anything": per-episode accuracy of the ancillary decoder reading
    the OCW's pooled summary, in the same episode order as ``per_teammate_curve``.
    An empty OCW (episode 1 against a teammate) still produces a prediction -- the
    pooled summary of an all-zero, all-masked context, not a skipped computation --
    so episode 1 is a real (likely poor, context-blind) number, not ``nan``; a
    working mechanism should trend upward from there as the OCW fills across
    episodes 2+. A flat or noisy curve with no trend means the encoder isn't using
    additional opponent trajectories, independent of what the aggregate return says.
    (``nan`` only appears for a literally zero-length episode -- see
    :func:`_ancillary_mate_action_acc`, not an empty context.)

    ``target_returns`` is ``{label: target}`` (see
    :func:`~oaht_bench.offline.evaluate.resolve_target_returns`) -- set via
    ``agent.set_target_return`` once per teammate, before its OCW resets and
    episodes start; every episode against one teammate shares that teammate's
    target, only the identity varies across the outer loop.
    """
    import jax

    probe = hasattr(agent, "decoder")
    per_mean: dict[str, float] = {}
    per_curve: dict[str, list[float]] = {}
    per_ancillary: dict[str, float] = {} if probe else None
    per_ancillary_curve: dict[str, list[float]] = {} if probe else None
    pooled_true = []
    for label, mate_params, mate_policy in teammates:
        agent.set_target_return(target_returns[label])
        ocw = OpponentContextWindow(ocw_size, max_episode_steps, obs_dim)
        curve: list[float] = []
        acc_curve: list[float] = []
        ep_pred, ep_true = [], []
        for _ in range(num_episodes):
            rng, ep_rng = jax.random.split(rng)
            params_ep = _context_params(agent, params, ocw)
            ego_return, mate_stream, mate_avail = rollout_one_episode(
                ep_rng,
                env,
                agent,
                params_ep,
                mate_params,
                mate_policy,
                max_episode_steps=max_episode_steps,
                ego_index=ego_index,
            )
            curve.append(float(ego_return))
            if probe:
                pred, true = _ancillary_mate_action_acc(
                    agent,
                    params_ep,
                    params_ep["stage2"]["context"],
                    mate_stream[0],
                    mate_stream[1],
                    mate_avail,
                )
                if pred is not None:
                    ep_pred.append(pred)
                    ep_true.append(true)
                    acc_curve.append(float((pred == true).mean()))
                else:
                    acc_curve.append(float("nan"))
            ocw.append(*mate_stream)
        if probe and ep_pred:
            p, t = np.concatenate(ep_pred), np.concatenate(ep_true)
            per_ancillary[label] = float((p == t).mean())
            per_ancillary_curve[label] = acc_curve
            pooled_true.append(t)
        per_curve[label] = curve
        per_mean[label] = float(np.mean(curve))

    ancillary_floor = None
    if probe and pooled_true:
        pooled = np.concatenate(pooled_true)
        ancillary_floor = float(np.bincount(pooled, minlength=agent.action_dim).max() / pooled.size)
    return per_mean, per_curve, per_ancillary, ancillary_floor, per_ancillary_curve
