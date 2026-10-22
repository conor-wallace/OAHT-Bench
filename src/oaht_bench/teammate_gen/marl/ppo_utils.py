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


def _is_checkpoint_step(update_steps, num_updates, num_checkpoints, evenly_spaced=False):
    """True on update steps whose params should be snapshotted: ``num_checkpoints``
    evenly spaced steps across training, plus always the final step (which may
    not land exactly on that spacing).

    ``evenly_spaced=True`` (default ``False``, reproducing the formula above
    exactly) instead snapshots at ``num_checkpoints`` targets computed
    directly from the already-known ``num_updates`` -- e.g. steps 30/60/90
    for 3 checkpoints over 90 updates -- rather than ``num_checkpoints - 1``
    equal-width segments measured from training start, which puts the first
    checkpoint one update after random init: a near-identical snapshot every
    member's checkpoint 0 otherwise shares regardless of
    generator/population/coefficient (observed directly: a released
    population's minimum pairwise KL divergence was 0.0000 with the default
    spacing). Only MEP's own checkpoint call site may opt in;
    ``population/loading.py``'s ``_checkpoint_update_steps`` independently
    duplicates the formula below to recover each stored checkpoint's update
    step for ``ppo_br``'s checkpoint-ratio selection without re-running
    training, so that call site must never see ``evenly_spaced=True``."""
    if evenly_spaced:
        target_steps = jnp.asarray(
            [(k * num_updates) // num_checkpoints for k in range(1, num_checkpoints + 1)]
        )
        return jnp.any(jnp.equal(update_steps, target_steps))
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


def _maybe_store_best(best_params, best_score, params, score):
    """Overwrite ``best_params``/``best_score`` with ``params``/``score`` iff
    ``score > best_score``, else leave both unchanged.

    Always the same slot -- unlike ``_maybe_store_checkpoint``'s advancing
    index, this is a running max, not a sequence of snapshots, so a plain
    ``jnp.where`` select suffices (no second side effect like index
    advancement to branch a ``lax.cond`` on).
    """
    is_new_best = score > best_score
    new_best_params = jax.tree.map(
        lambda b, p: jnp.where(is_new_best, p, b), best_params, params
    )
    return new_best_params, jnp.where(is_new_best, score, best_score)


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