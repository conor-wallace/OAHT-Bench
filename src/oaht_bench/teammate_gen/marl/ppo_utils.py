import jax
import jax.numpy as jnp
from typing import NamedTuple

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_info(x: dict, agent_list, num_actors):
    '''Handle special case that info has both per-agent and global information'''
    x = jnp.stack([x[a] for a in x if a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape((num_agents, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _is_checkpoint_step(update_steps, num_updates, num_checkpoints):
    """True on update steps whose params should be snapshotted: ``num_checkpoints``
    evenly spaced steps across training, plus always the final step (which may
    not land exactly on that spacing)."""
    ckpt_interval = num_updates // max(1, num_checkpoints - 1)
    return jnp.logical_or(
        jnp.equal(jnp.mod(update_steps - 1, ckpt_interval), 0),
        jnp.equal(update_steps, num_updates),
    )


def _maybe_store_checkpoint(checkpoint_array, ckpt_idx, params, should_store):
    """Write ``params`` into ``checkpoint_array[ckpt_idx]`` and advance ``ckpt_idx``
    if ``should_store``, else leave both unchanged."""

    def store(args):
        _checkpoint_array, _ckpt_idx, _params = args
        new_checkpoint_array = jax.tree.map(
            lambda c_arr, p: c_arr.at[_ckpt_idx].set(p), _checkpoint_array, _params
        )
        return new_checkpoint_array, _ckpt_idx + 1

    def skip(args):
        _checkpoint_array, _ckpt_idx, _ = args
        return _checkpoint_array, _ckpt_idx

    return jax.lax.cond(should_store, store, skip, (checkpoint_array, ckpt_idx, params))


def _create_minibatches(traj_batch, advantages, targets, init_hstate, num_actors, num_minibatches, perm_rng):
    """Create minibatches for PPO updates, where each leaf has shape 
        (num_minibatches, rollout_len, num_actors / num_minibatches, ...) 
    This function ensures that the rollout (time) dimension is kept separate from the minibatch and num_actors 
    dimensions, so that the minibatches are compatible with recurrent ActorCritics.
    """
    # Create batch containing trajectory, advantages, and targets
    batch = (
        init_hstate, # shape (1, num_actors, hidden_dim)
        traj_batch, # pytree: obs is shape (rollout_len, num_actors, feat_shape)
        advantages, # shape (rollout_len, num_actors)
        targets # shape (rollout_len, num_actors)
            )

    permutation = jax.random.permutation(perm_rng, num_actors)

    # each leaf of shuffled batch has shape (rollout_len, num_actors, feat_shape)
    # except for init_hstate which has shape (1, num_actors, hidden_dim)
    shuffled_batch = jax.tree.map(
        lambda x: jnp.take(x, permutation, axis=1), batch
    )
    # each leaf has shape (num_minibatches, rollout_len, num_actors/num_minibatches, feat_shape)
    # except for init_hstate which has shape (num_minibatches, 1, num_actors/num_minibatches, hidden_dim)
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(
            jnp.reshape(
                x,
                [x.shape[0], num_minibatches, -1] 
                + list(x.shape[2:]),
        ), 1, 0,),
        shuffled_batch,
    )

    return minibatches