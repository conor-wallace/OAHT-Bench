import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp


@dataclass
class TeammatePolicy:
    """Unified interface for teammate policies.

    This is a dataclass wrapper providing a consistent interface for both
    heuristic and RL teammates.

    Attributes:
        name: Human-readable name (e.g., "assembly_line[default]" or "brdiv[ckpt_run1]")
        init: Callable (rng) -> carry, initializes policy state
        act: Callable (carry, obs, done, rng, env_state, avail_actions) -> (new_carry, action)
        _policy: The underlying policy object (for advanced use)
        _params: Policy parameters (only for RL policies, None for heuristics)
    """
    name: str
    init: Callable[[chex.PRNGKey], Any]
    act: Callable[..., Tuple[Any, jnp.ndarray]]
    _policy: Any = None
    _params: Any = None


def _make_rl_teammate(
    spec: RLTeammateSpec,
    env,
    teammate_agent_id: str,
) -> TeammatePolicy:
    """Create an RL teammate policy from spec.

    Args:
        spec: The RL teammate specification
        env: The environment instance
        teammate_agent_id: The agent ID for this teammate

    Returns:
        TeammatePolicy with unified interface

    Note:
        If the checkpoint file doesn't exist, this will raise FileNotFoundError.
        Use the dry_run mode in tests to skip actual loading.
    """
    # Import here to avoid circular imports and allow lazy loading
    from teammate_wrapper.rl_wrappers import make_rl_policy_wrapper

    # Get checkpoint basename for name
    ckpt_basename = os.path.basename(spec.ckpt_path.rstrip('/\\'))
    name = f"{spec.algo}[{ckpt_basename}]"

    # Create the RL policy wrapper
    policy, params = make_rl_policy_wrapper(
        algo=spec.algo,
        ckpt_path=spec.ckpt_path,
        env=env,
        use_log_wrapper=spec.use_log_wrapper,
        extra=spec.extra,
    )

    # Parse agent_id to int
    if isinstance(teammate_agent_id, str) and teammate_agent_id.startswith("agent_"):
        agent_idx = int(teammate_agent_id.split("_")[1])
    else:
        agent_idx = int(teammate_agent_id)

    # Get extra config
    test_mode = spec.extra.get("test_mode", True)

    # Create init function
    def init_fn(rng: chex.PRNGKey) -> Any:
        """Initialize the RL policy carry state (hidden state if RNN)."""
        return policy.init_hstate(batch_size=1, aux_info={"agent_id": agent_idx})

    # Create act function
    def act_fn(
        carry: Any,
        obs: jnp.ndarray,
        done: jnp.ndarray,
        rng: Optional[chex.PRNGKey] = None,
        env_state: Any = None,
        avail_actions: Optional[jnp.ndarray] = None,
    ) -> Tuple[Any, jnp.ndarray]:
        """Get action from the RL policy.

        Args:
            carry: The policy carry state (hidden state for RNN, None for MLP)
            obs: Observation
            done: Done flag
            rng: JAX random key for stochastic action selection
            env_state: Environment state (unused for RL policies)
            avail_actions: Available actions mask

        Returns:
            Tuple of (new_carry, action)
        """
        # Ensure done is a JAX array
        done = jnp.asarray(done)
        if done.ndim == 0:
            done = done.reshape(1)

        # Get available actions if not provided
        if avail_actions is None:
            avail_actions = jnp.ones((6,), dtype=jnp.float32)

        # Check if this is a CNN+RNN policy (has obs_shape attribute)
        is_cnn_rnn = hasattr(policy, 'obs_shape')

        # For RNN/CNN+RNN policies, need to handle sequence dimension
        if hasattr(policy, 'gru_hidden_dim'):
            if is_cnn_rnn:
                # CNN+RNN policy expects (seq_len, batch, H, W, C) shape
                # Add sequence and batch dimensions
                obs_seq = obs.reshape(1, 1, *obs.shape)  # (1, 1, H, W, C)
            else:
                # Standard RNN policy expects (seq_len, batch, obs_dim) shape
                obs_seq = obs.reshape(1, 1, -1)  # (1, 1, obs_dim)

            done_seq = done.reshape(1, 1)  # (1, 1)
            avail_seq = avail_actions.reshape(1, 1, -1)  # (1, 1, action_dim)

            action, new_carry = policy.get_action(
                params=params,
                obs=obs_seq,
                done=done_seq,
                avail_actions=avail_seq,
                hstate=carry,
                rng=rng,
                test_mode=test_mode,
            )
            # Remove sequence dimension from action
            action = action.squeeze(0)
        else:
            # MLP/S5 policy
            action, new_carry = policy.get_action(
                params=params,
                obs=obs,
                done=done,
                avail_actions=avail_actions,
                hstate=carry,
                rng=rng,
                test_mode=test_mode,
            )

        # Reset carry on done (for RNN policies with hidden state)
        if carry is not None and hasattr(policy, 'init_hstate'):
            new_carry = jax.lax.cond(
                done.squeeze().astype(bool),
                lambda: policy.init_hstate(batch_size=1, aux_info={"agent_id": agent_idx}),
                lambda: new_carry,
            )

        return new_carry, action

    return TeammatePolicy(
        name=name,
        init=init_fn,
        act=act_fn,
        _policy=policy,
        _params=params,
    )


def make_teammate(
    spec: TeammateSpec,
    env,
    teammate_agent_id: str,
) -> TeammatePolicy:
    """Create a teammate policy from a specification.

    This is the main factory function for the teammate registry. It takes a
    specification (heuristic or RL) and returns a unified TeammatePolicy interface.

    Args:
        spec: TeammateSpec (HeuristicTeammateSpec or RLTeammateSpec)
        env: The environment instance (used for layout and obs/action dims)
        teammate_agent_id: The agent ID string (e.g., "agent_0", "agent_1")

    Returns:
        TeammatePolicy with:
            - name: Human-readable identifier
            - init(rng) -> carry: Initialize policy state
            - act(carry, obs, done, rng, env_state, avail_actions) -> (new_carry, action)

    Raises:
        ValueError: If the spec is invalid
        TypeError: If the spec type is unknown
        FileNotFoundError: If an RL checkpoint doesn't exist

    Example:
        >>> from teammate_wrapper import make_teammate, HeuristicTeammateSpec
        >>> spec = HeuristicTeammateSpec(family="assembly_line", theta="default")
        >>> teammate = make_teammate(spec, env, "agent_1")
        >>> carry = teammate.init(jax.random.PRNGKey(0))
        >>> new_carry, action = teammate.act(carry, obs, done, rng, env_state=state)
    """
    # Validate spec first
    validate_spec(spec)

    if isinstance(spec, HeuristicTeammateSpec):
        return _make_heuristic_teammate(spec, env, teammate_agent_id)
    elif isinstance(spec, RLTeammateSpec):
        return _make_rl_teammate(spec, env, teammate_agent_id)
    else:
        raise TypeError(
            f"Unknown spec type '{type(spec).__name__}'. "
            f"Expected HeuristicTeammateSpec or RLTeammateSpec."
        )