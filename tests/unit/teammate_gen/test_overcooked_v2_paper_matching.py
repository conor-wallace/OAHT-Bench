"""Unit coverage for the Overcooked-v2 paper-matching pieces (Gessler et al.,
ICLR 2025): annealed reward shaping, the CNN+GRU network, and the LR
warmup+cosine schedule. Configs are built in-memory; no shipped-file paths.

These pin the non-regression contract: at ``reward_shaping_horizon=0`` /
``lr_warmup=0`` the shaping and schedule are exactly what they were before, so
LBF/Hanabi/Overcooked-v1 are untouched.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oaht_bench.configs import get_preset
from oaht_bench.configs.teammate_gen import PpoHyperparams
from oaht_bench.envs import make_env
from oaht_bench.envs.log_wrapper import LogWrapper
from oaht_bench.teammate_gen.marl.lr_schedule import make_lr_schedule
from oaht_bench.teammate_gen.marl.reward_shaping import add_shaped_reward, shaping_coef


# --------------------------------------------------------------------------
# Part 1 -- annealed reward shaping
# --------------------------------------------------------------------------
def test_shaping_coef_anneals_one_to_zero():
    assert float(shaping_coef(100.0, 0)) == pytest.approx(1.0)
    assert float(shaping_coef(100.0, 50)) == pytest.approx(0.5)
    assert float(shaping_coef(100.0, 100)) == pytest.approx(0.0)
    # clamped: never negative past the horizon
    assert float(shaping_coef(100.0, 200)) == pytest.approx(0.0)


def test_add_shaped_reward_is_identity_when_disabled():
    reward = {"agent_0": jnp.ones(3), "agent_1": jnp.ones(3)}
    info = {"shaped_reward": jnp.full((3, 2), 5.0)}
    # horizon 0 -> untouched (LBF/Hanabi/v1 path)
    out = add_shaped_reward(reward, info, ["agent_0", "agent_1"], horizon=0.0, global_env_step=0)
    assert out is reward
    # env without a shaped reward -> untouched even with a horizon set
    out2 = add_shaped_reward(reward, {}, ["agent_0", "agent_1"], horizon=1e6, global_env_step=0)
    assert out2 is reward


def test_add_shaped_reward_folds_per_agent_with_coef():
    reward = {"agent_0": jnp.zeros(3), "agent_1": jnp.zeros(3)}
    info = {"shaped_reward": jnp.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])}
    # step 0 of a horizon-100 anneal: coef 1.0, full shaped reward added per agent
    out = add_shaped_reward(reward, info, ["agent_0", "agent_1"], horizon=100.0, global_env_step=0)
    assert np.allclose(out["agent_0"], 1.0)
    assert np.allclose(out["agent_1"], 2.0)
    # halfway: coef 0.5
    out_h = add_shaped_reward(reward, info, ["agent_0", "agent_1"], horizon=100.0, global_env_step=50)
    assert np.allclose(out_h["agent_0"], 0.5)
    assert np.allclose(out_h["agent_1"], 1.0)


def test_reward_shaping_horizon_defaults_off():
    assert PpoHyperparams().reward_shaping_horizon == 0.0


# --------------------------------------------------------------------------
# Part 3 -- LR warmup + cosine, with the no-warmup path unchanged
# --------------------------------------------------------------------------
def test_lr_schedule_constant_when_no_warmup_no_anneal():
    lr = make_lr_schedule(PpoHyperparams(learning_rate=5e-4), num_updates=100)
    assert isinstance(lr, float) and lr == pytest.approx(5e-4)


def test_lr_schedule_linear_when_annealed_matches_old_formula():
    ppo = PpoHyperparams(learning_rate=5e-4, anneal_lr=True, num_minibatches=4, update_epochs=2)
    lr = make_lr_schedule(ppo, num_updates=100)
    assert callable(lr)
    steps_per_update = 4 * 2
    assert float(lr(0)) == pytest.approx(5e-4)
    assert float(lr(99 * steps_per_update)) == pytest.approx(5e-4 * (1 - 99 / 100))


def test_lr_schedule_warmup_cosine_shape():
    ppo = PpoHyperparams(learning_rate=5e-4, lr_warmup=0.1, num_minibatches=4, update_epochs=2)
    lr = make_lr_schedule(ppo, num_updates=100)
    total = 100 * 4 * 2
    warm = int(0.1 * total)
    assert float(lr(0)) == pytest.approx(0.0, abs=1e-9)  # warmup starts at 0
    assert float(lr(warm)) == pytest.approx(5e-4, rel=1e-3)  # peak at end of warmup
    assert float(lr(total)) == pytest.approx(0.0, abs=1e-6)  # cosine decays to 0


def test_lr_warmup_defaults_off():
    assert PpoHyperparams().lr_warmup == 0.0


# --------------------------------------------------------------------------
# Part 2 -- CNN+GRU network forward pass, params init from env.obs_shape
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def v2_env():
    cfg = get_preset("overcooked_v2_counter_circuit")
    return LogWrapper(make_env(cfg.env_name, cfg.env_kwargs()))


def test_cnn_actor_critic_forward_shapes(v2_env):
    from oaht_bench.models.cnn_rnn_actor_critic_agent import CNNRNNActorCriticPolicy
    from oaht_bench.models.initialize_agents import _unwrap_obs_shape

    obs_shape = _unwrap_obs_shape(v2_env)
    flat = v2_env.observation_space(v2_env.agents[0]).shape[0]
    assert int(np.prod(obs_shape)) == flat  # the reshape the encoder relies on

    adim = v2_env.action_space(v2_env.agents[0]).n
    policy = CNNRNNActorCriticPolicy(action_dim=adim, obs_dim=flat, obs_shape=obs_shape)
    params = policy.init_params(jax.random.PRNGKey(0))

    seq, batch = 3, 5
    obs = jnp.zeros((seq, batch, flat))
    done = jnp.zeros((seq, batch))
    avail = jnp.ones((seq, batch, adim))
    action, val, pi, hstate = policy.get_action_value_policy(
        params, obs, done, avail, policy.init_hstate(batch), jax.random.PRNGKey(1)
    )
    assert action.shape == (seq, batch)
    assert val.shape == (seq, batch)
    assert pi.logits.shape == (seq, batch, adim)
    assert hstate.shape == (1, batch, 128)


def test_cnn_conditional_critic_forward_shapes(v2_env):
    from oaht_bench.models.cnn_rnn_actor_critic_agent import CNNRNNActorWithConditionalCriticPolicy
    from oaht_bench.models.initialize_agents import _unwrap_obs_shape

    obs_shape = _unwrap_obs_shape(v2_env)
    flat = v2_env.observation_space(v2_env.agents[0]).shape[0]
    adim = v2_env.action_space(v2_env.agents[0]).n
    pop = 4
    policy = CNNRNNActorWithConditionalCriticPolicy(
        action_dim=adim, obs_dim=flat, obs_shape=obs_shape, pop_size=pop
    )
    params = policy.init_params(jax.random.PRNGKey(2))

    seq, batch = 3, 5
    obs = jnp.zeros((seq, batch, flat))
    done = jnp.zeros((seq, batch))
    avail = jnp.ones((seq, batch, adim))
    aux = jnp.zeros((seq, batch, pop))
    action, val, pi, hstate = policy.get_action_value_policy(
        params, obs, done, avail, policy.init_hstate(batch), jax.random.PRNGKey(3), aux_obs=aux
    )
    assert action.shape == (seq, batch)
    assert val.shape == (seq, batch)
    assert hstate.shape == (1, batch, 128)


def test_unwrap_obs_shape_raises_without_grid():
    """LBF has no obs_shape; the CNN initializers must fail loudly, not silently
    build a wrong architecture."""
    from oaht_bench.models.initialize_agents import _unwrap_obs_shape

    lbf = LogWrapper(make_env(get_preset("lbf_12x12").env_name, get_preset("lbf_12x12").env_kwargs()))
    with pytest.raises(ValueError, match="obs_shape"):
        _unwrap_obs_shape(lbf)


# --------------------------------------------------------------------------
# Part 4 -- generated configs carry the paper backbone; a paired runtime builds
# --------------------------------------------------------------------------
def test_generated_v2_configs_use_cnn_and_table4():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gtc", "scripts/gen_teammate_configs.py")
    gtc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gtc)

    expected_actor = {
        "fcp": "cnn_rnn",
        "comedi": "cnn_rnn_actor_with_conditional_critic",
        "brdiv": "cnn_rnn_actor_with_conditional_critic",
        "lbrdiv": "cnn_rnn_actor_with_conditional_critic",
    }
    for gen_name, actor in expected_actor.items():
        c = gtc.build(gen_name, "overcooked_v2_counter_circuit")
        assert c.actor_type == actor
        assert c.ppo.learning_rate == pytest.approx(5e-4)
        assert c.ppo.lr_warmup == pytest.approx(0.05)
        assert c.ppo.reward_shaping_horizon == pytest.approx(5e6)
        assert c.ppo.clip_eps == pytest.approx(0.2)
        assert c.ppo.max_grad_norm == pytest.approx(0.25)
        assert c.network.hidden_dim == 128
        assert c.network.activation == "relu"
        # the paper's num_envs (H100 target); num_minibatches=64 -> 4 envs/minibatch
        assert c.num_envs == 256
        assert c.ppo.num_minibatches == 64

    # non-regression: LBF stays on the MLP backbone with shaping/warmup off
    lbf = gtc.build("brdiv", "lbf_12x12")
    assert lbf.actor_type == "actor_with_conditional_critic"
    assert lbf.ppo.reward_shaping_horizon == 0.0
    assert lbf.ppo.lr_warmup == 0.0
    assert lbf.network.hidden_dim == 64
