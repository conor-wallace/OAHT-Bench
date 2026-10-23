"""``ActorCritic``'s configurable depth (``num_layers``).

OMIS's own PPO actor/critic for LBF is 5 hidden layers wide at hidden_dim,
not the absorbed code's prior fixed 2 (confirmed directly from
OMIS/pretraining/ppo.py's PolicyNet/ValueNet, not the paper's own ambiguous
table -- see ``configs/network.py::MlpNetwork.num_layers``'s docstring).
``num_layers=2`` stays the default so every existing config is unaffected.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from oaht_bench.models.mlp_actor_critic import ActorCritic


def _dense_kernel_count(params) -> int:
    return sum(
        1 for path, _ in jax.tree_util.tree_leaves_with_path(params) if "kernel" in path[-1].key
    )


@pytest.mark.parametrize("num_layers", [2, 5])
def test_actor_critic_param_count_scales_with_num_layers(num_layers):
    action_dim, obs_dim = 4, 6
    net = ActorCritic(action_dim=action_dim, fc_hidden_dim=8, num_layers=num_layers)
    obs = jnp.zeros((obs_dim,))
    avail = jnp.ones((action_dim,))
    params = net.init(jax.random.PRNGKey(0), (obs, avail))["params"]

    # num_layers hidden Dense(+activation) layers plus one output Dense, mirrored
    # on actor and critic -- (num_layers + 1) * 2 Dense kernels total.
    assert _dense_kernel_count(params) == (num_layers + 1) * 2


def test_actor_critic_output_shapes_are_unaffected_by_depth():
    action_dim, obs_dim = 4, 6
    obs = jnp.zeros((obs_dim,))
    avail = jnp.ones((action_dim,))
    for num_layers in (2, 5):
        net = ActorCritic(action_dim=action_dim, fc_hidden_dim=8, num_layers=num_layers)
        params = net.init(jax.random.PRNGKey(0), (obs, avail))
        pi, value = net.apply(params, (obs, avail))
        assert pi.logits.shape == (action_dim,)
        assert value.shape == ()


def test_default_num_layers_is_two_dense_kernel_count_unchanged():
    """Regression guard: the default must match the pre-num_layers behaviour
    exactly -- 2 hidden layers, not some other depth."""
    action_dim, obs_dim = 4, 6
    net = ActorCritic(action_dim=action_dim, fc_hidden_dim=8)  # num_layers unset -> default
    obs = jnp.zeros((obs_dim,))
    avail = jnp.ones((action_dim,))
    params = net.init(jax.random.PRNGKey(0), (obs, avail))["params"]
    assert _dense_kernel_count(params) == 6  # (2 + 1) * 2
