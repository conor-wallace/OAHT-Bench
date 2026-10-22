"""LiamAgent as an AgentPolicy: its acting step must reproduce the reference
deployment loop exactly.

``LiamAgent.get_action`` moved the rolling-window / return-to-go bookkeeping that
used to live inline in ``offline.evaluate._rollout`` into the agent, so the
shared ``run_episodes`` loop can drive it. A full end-to-end diff against
``_rollout`` is impossible -- ``run_episodes`` threads its RNG through a
``lax.scan`` while ``_rollout`` is a Python loop, so the *sampled* actions
diverge even for identical policies. What must be identical is everything
deterministic: given the same observation/reward sequence and greedy action
selection, the window the agent assembles at every step, the return-to-go it
conditions on, and the action it takes must match a hand-rolled copy of
``_rollout``'s window logic. That is what this test pins.
"""

import jax
import jax.numpy as jnp
import numpy as np

from oaht_bench.configs.job import OfflineTrainingConfig
from oaht_bench.dataset.dataset import Normalization
from oaht_bench.models.agent_interface import AgentPolicy
from oaht_bench.models.liam_agent import LiamAgent
from oaht_bench.models.masking import mask_logits

OBS_DIM = 6
ACTION_DIM = 4
K = 5  # context length
T = 9  # steps, > K so the window fully cycles


def _agent_and_params(normalization, target_return):
    cfg = OfflineTrainingConfig.model_validate(
        {"network": {"architecture": "liam", "obs_dim": OBS_DIM, "action_dim": ACTION_DIM}}
    )
    agent = LiamAgent(
        cfg,
        context_length=K,
        target_return=target_return,
        normalization=normalization,
    )
    agent.build_model()

    # Minimal eval-shaped params: an encoder over a (1, K) window and a policy
    # conditioned on its embedding. The decoder is not needed to act.
    rng = jax.random.PRNGKey(0)
    k_enc, k_net = jax.random.split(rng)
    dummy = dict(
        rtg=jnp.zeros((1, K)),
        obs=jnp.zeros((1, K, OBS_DIM)),
        actions=jnp.zeros((1, K), dtype=jnp.int32),
        timesteps=jnp.zeros((1, K), dtype=jnp.int32),
        mask=jnp.ones((1, K), dtype=bool),
    )
    enc_params = agent.encoder.init(
        k_enc,
        dummy["rtg"],
        dummy["obs"],
        dummy["actions"],
        timesteps=dummy["timesteps"],
        mask=dummy["mask"],
    )
    z = agent.encoder.apply(
        enc_params,
        dummy["rtg"],
        dummy["obs"],
        dummy["actions"],
        timesteps=dummy["timesteps"],
        mask=dummy["mask"],
    )
    net_params = agent.network.init(
        k_net,
        dummy["rtg"],
        dummy["obs"],
        dummy["actions"],
        timesteps=dummy["timesteps"],
        embedding=z,
        mask=dummy["mask"],
    )
    params = {"stage1": {"encoder": enc_params}, "stage2": net_params}
    return agent, params


def _reference_step(agent, params, ref, obs_t, prev_r, avail_t, *, normalization, step):
    """One step of ``_rollout``'s window logic, in numpy (offline/evaluate.py)."""
    scale = 1.0 if normalization is None else normalization.rtg_scale
    ref["rtg"] = ref["rtg"] - prev_r / scale

    ref["obs"] = np.roll(ref["obs"], -1, axis=0)
    ref["act"] = np.roll(ref["act"], -1)
    ref["rtg_ctx"] = np.roll(ref["rtg_ctx"], -1)
    ref["t"] = np.roll(ref["t"], -1)
    ref["mask"] = np.roll(ref["mask"], -1)

    ref["obs"][-1] = obs_t if normalization is None else normalization.apply_obs(obs_t)
    ref["act"][-1] = -10
    ref["rtg_ctx"][-1] = ref["rtg"]
    ref["t"][-1] = min(step + 1, K * 64)
    ref["mask"][-1] = True

    logits = np.asarray(
        agent.act(
            params,
            jnp.asarray(ref["rtg_ctx"])[None],
            jnp.asarray(ref["obs"])[None],
            jnp.asarray(ref["act"])[None],
            timesteps=jnp.asarray(ref["t"])[None],
            mask=jnp.asarray(ref["mask"])[None],
        )
    )
    masked = np.asarray(mask_logits(jnp.asarray(logits[0, -1]), jnp.asarray(avail_t)))
    action = int(np.argmax(masked))
    ref["act"][-1] = action
    return action


def _fresh_ref():
    return dict(
        obs=np.zeros((K, OBS_DIM), dtype=np.float32),
        act=np.full(K, -10, dtype=np.int32),
        rtg_ctx=np.zeros(K, dtype=np.float32),
        t=np.zeros(K, dtype=np.int32),
        mask=np.zeros(K, dtype=bool),
    )


def _run_equivalence(normalization):
    rng_np = np.random.default_rng(1)
    obs_seq = rng_np.normal(size=(T, OBS_DIM)).astype(np.float32)
    reward_seq = rng_np.uniform(0.0, 1.0, size=T).astype(np.float32)
    # A couple of masked actions per step, so the argmax genuinely exercises masking.
    avail_seq = np.ones((T, ACTION_DIM), dtype=np.float32)
    avail_seq[np.arange(T) % 2 == 0, 0] = 0.0
    avail_seq[np.arange(T) % 3 == 0, ACTION_DIM - 1] = 0.0

    # The already-conditioned target the runner would pass (units the agent
    # decrements in); the reference uses the identical value.
    target_return = 2.5
    agent, params = _agent_and_params(normalization, target_return)
    assert isinstance(agent, AgentPolicy)

    ref = _fresh_ref()
    ref["rtg"] = float(target_return)

    hstate = agent.init_hstate(1, aux_info={"agent_id": 0})
    dummy_done = jnp.zeros((1, 1), dtype=bool)
    key = jax.random.PRNGKey(7)  # unused under test_mode=True, but the signature wants one

    for step in range(T):
        prev_r = 0.0 if step == 0 else float(reward_seq[step - 1])
        ref_action = _reference_step(
            agent,
            params,
            ref,
            obs_seq[step],
            prev_r,
            avail_seq[step],
            normalization=normalization,
            step=step,
        )
        action, hstate = agent.get_action(
            params,
            jnp.asarray(obs_seq[step]).reshape(1, 1, -1),
            dummy_done,
            jnp.asarray(avail_seq[step]),
            hstate,
            key,
            test_mode=True,
            reward=jnp.asarray(prev_r).reshape(1, 1, 1),
        )

        assert int(action) == ref_action, f"action mismatch at step {step}"
        np.testing.assert_allclose(np.asarray(hstate.ctx_obs), ref["obs"], rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(np.asarray(hstate.ctx_act), ref["act"])
        np.testing.assert_allclose(np.asarray(hstate.ctx_rtg), ref["rtg_ctx"], rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(np.asarray(hstate.ctx_t), ref["t"])
        np.testing.assert_array_equal(np.asarray(hstate.ctx_mask), ref["mask"])
        np.testing.assert_allclose(float(hstate.rtg), ref["rtg"], rtol=1e-5, atol=1e-5)


def test_get_action_matches_rollout_window_unnormalised():
    _run_equivalence(normalization=None)


def test_get_action_matches_rollout_window_normalised():
    norm = Normalization(
        obs_mean=np.arange(OBS_DIM, dtype=np.float32) * 0.1,
        obs_std=np.linspace(0.5, 2.0, OBS_DIM).astype(np.float32),
        rtg_scale=3.0,
    )
    _run_equivalence(normalization=norm)
