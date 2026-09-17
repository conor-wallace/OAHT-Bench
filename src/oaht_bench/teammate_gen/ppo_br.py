"""`ppo_br` — a dedicated best-response ego per fixed teammate.

The offline dataset needs, for each teammate, the trajectories of a *single*
competent ego that best-responds to it (TAO/OMIS build the offline set exactly this
way; ICRL4AHT fixes the generated teammates and trains an ego PPO best-response
against each). We had been *reusing* population policies as the ego, which the
teammate-id oracle showed caps the dataset at ~45% of competence
(``docs/tuning_record.md``). This trains the real thing.

Structure: one trainable ego in seat 0, one **frozen** teammate in seat 1. It is a
two-policy PPO — the same rollout/GAE/clipped-PPO machinery as
:func:`~oaht_bench.teammate_gen.marl.ippo.make_train`, but the seats carry *different*
params: only the ego has a ``TrainState`` and gradients; the teammate acts with fixed
params (``jax.lax.stop_gradient``) and is part of the environment. The ego is
**warm-started** from an already-competent policy (the paired ``br`` where one exists,
else the self-play member itself), so PPO fine-tunes rather than learns from scratch.

``jax.vmap`` over a leading member axis trains one BR per teammate in a single run
(the ``teammate_params`` and ``ego_init_params`` pytrees carry that axis, exactly as
:mod:`~oaht_bench.teammate_gen.brdiv` vmaps over ``population_size``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from oaht_bench.teammate_gen.marl.lr_schedule import make_lr_schedule
from oaht_bench.teammate_gen.marl.ppo_utils import (
    Transition,
    _create_minibatches,
)
from oaht_bench.teammate_gen.marl.reward_shaping import add_shaped_reward


def make_br_train(runtime, env, ego_policy, teammate_policy, logger=None, progress_callback=None):
    """Build the vmappable BR-training function.

    Returns ``train(rng, teammate_params, ego_init_params)`` — vmap it over a leading
    member axis on ``teammate_params``/``ego_init_params`` to train one best-response
    per teammate at once. The ego occupies ``env.agents[0]``; the teammate fills the
    rest with its fixed params.
    """
    config = runtime
    n_envs = config.num_envs
    ego_name = env.agents[0]
    mate_names = env.agents[1:]
    lr = make_lr_schedule(config.ppo, config.num_updates)

    def _aux(ref, ego_aux_id):
        # A conditional-critic ego conditions its value head on a fixed id (its own
        # member id, as BRDiv's forward_pass_br passed); broadcast that constant to the
        # obs's leading dims. None for plain actors (no aux head).
        if ego_aux_id is None:
            return None
        return jnp.broadcast_to(ego_aux_id, (*ref.shape[:-1], ego_aux_id.shape[-1]))

    def _ego_forward(params, obs_d, done_d, avail_d, hstate, rng, ego_aux_id):
        obs = obs_d[ego_name].reshape(1, n_envs, -1)
        act, val, pi, new_h = ego_policy.get_action_value_policy(
            params=params,
            obs=obs,
            done=done_d[ego_name].reshape(1, n_envs),
            avail_actions=avail_d[ego_name].reshape(1, n_envs, -1),
            hstate=hstate,
            rng=rng,
            aux_obs=_aux(obs, ego_aux_id),
        )
        return act, val, pi, new_h

    def _mate_act(params, name, obs_d, done_d, avail_d, hstate, rng):
        # Teammate is frozen: only its action matters (no value/grad), so use
        # get_action, which needs no aux_obs even for conditional-critic policies.
        act, new_h = teammate_policy.get_action(
            params=jax.lax.stop_gradient(params),
            obs=obs_d[name].reshape(1, n_envs, -1),
            done=done_d[name].reshape(1, n_envs),
            avail_actions=avail_d[name].astype(jnp.float32).reshape(1, n_envs, -1),
            hstate=hstate,
            rng=rng,
            test_mode=False,
        )
        return act, new_h

    def train(rng, teammate_params, ego_init_params, ego_aux_id=None):
        tx = optax.chain(
            optax.clip_by_global_norm(config.ppo.max_grad_norm),
            optax.adam(learning_rate=lr, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=ego_policy.network.apply, params=ego_init_params, tx=tx
        )

        rng, reset_rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset)(jax.random.split(reset_rng, n_envs))

        ego_h0 = ego_policy.init_hstate(n_envs)
        mate_h0 = {name: teammate_policy.init_hstate(n_envs) for name in mate_names}

        def _update_step(runner_state, unused):
            def _env_step(carry, unused):
                train_state, env_state, last_obs, last_done, ego_h, mate_h, upd, rng = carry
                rng, ego_rng, step_rng, *mate_rngs = jax.random.split(rng, 3 + len(mate_names))

                avail = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail = jax.lax.stop_gradient(avail)

                action, value, pi, new_ego_h = _ego_forward(
                    train_state.params, last_obs, last_done, avail, ego_h, ego_rng, ego_aux_id
                )
                log_prob = pi.log_prob(action).squeeze()
                ego_action = action.squeeze()

                env_act = {ego_name: ego_action}
                new_mate_h = {}
                for name, mr in zip(mate_names, mate_rngs, strict=True):
                    m_act, m_h = _mate_act(
                        teammate_params, name, last_obs, last_done, avail, mate_h[name], mr
                    )
                    env_act[name] = m_act.squeeze()
                    new_mate_h[name] = m_h

                new_obs, new_env_state, reward, new_done, info = jax.vmap(env.step)(
                    jax.random.split(step_rng, n_envs), env_state, env_act
                )
                reward = add_shaped_reward(
                    reward, info, env.agents,
                    horizon=config.ppo.reward_shaping_horizon,
                    global_env_step=upd * config.rollout_length * n_envs,
                )
                # Keep the ego's slice of each per-agent info leaf; leaves already
                # per-env are untouched. The metric below reads the ego return.
                info = jax.tree.map(lambda x: x[:, 0] if x.ndim >= 2 else x, info)

                transition = Transition(
                    new_done[ego_name],
                    ego_action,
                    value.squeeze(),
                    reward[ego_name],
                    log_prob,
                    last_obs[ego_name],
                    info,
                    avail[ego_name].astype(jnp.float32),
                )
                carry = (train_state, new_env_state, new_obs, new_done, new_ego_h, new_mate_h, upd, rng)
                return carry, transition

            (train_state, env_state, last_obs, last_done, ego_h, mate_h, upd, rng) = runner_state
            carry, traj = jax.lax.scan(
                _env_step,
                (train_state, env_state, last_obs, last_done, ego_h, mate_h, upd, rng),
                None,
                config.rollout_length,
            )
            (train_state, env_state, last_obs, last_done, ego_h, mate_h, upd, rng) = carry

            avail = jax.lax.stop_gradient(jax.vmap(env.get_avail_actions)(env_state.env_state))
            _, last_val, _, _ = _ego_forward(
                train_state.params, last_obs, last_done, avail, ego_h, jax.random.PRNGKey(0), ego_aux_id
            )
            last_val = last_val.squeeze()

            def _gae(traj, last_val):
                def _adv(carry, t):
                    gae, next_value = carry
                    delta = t.reward + config.ppo.gamma * next_value * (1 - t.done) - t.value
                    gae = delta + config.ppo.gamma * config.ppo.gae_lambda * (1 - t.done) * gae
                    return (gae, t.value), gae

                _, adv = jax.lax.scan(
                    _adv, (jnp.zeros_like(last_val), last_val), traj, reverse=True, unroll=16
                )
                return adv, adv + traj.value

            advantages, targets = _gae(traj, last_val)

            def _update_epoch(update_state, unused):
                def _minbatch(ts, batch):
                    init_h, mb_traj, mb_adv, mb_tgt = batch

                    def _loss(params):
                        _, value, pi, _ = ego_policy.get_action_value_policy(
                            params=params,
                            obs=mb_traj.obs,
                            done=mb_traj.done,
                            avail_actions=mb_traj.avail_actions,
                            hstate=init_h,
                            rng=jax.random.PRNGKey(0),
                            aux_obs=_aux(mb_traj.obs, ego_aux_id),
                        )
                        log_prob = pi.log_prob(mb_traj.action)
                        v_clipped = mb_traj.value + (value - mb_traj.value).clip(
                            -config.ppo.clip_eps, config.ppo.clip_eps
                        )
                        value_loss = jnp.maximum(
                            jnp.square(value - mb_tgt), jnp.square(v_clipped - mb_tgt)
                        ).mean()
                        ratio = jnp.exp(log_prob - mb_traj.log_prob)
                        gae = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                        actor_loss = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1.0 - config.ppo.clip_eps, 1.0 + config.ppo.clip_eps) * gae,
                        ).mean()
                        entropy = pi.entropy().mean()
                        return (
                            actor_loss
                            + config.ppo.value_coef * value_loss
                            - config.ppo.entropy_coef * entropy
                        ), (value_loss, actor_loss, entropy)

                    (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(ts.params)
                    return ts.apply_gradients(grads=grads), (loss, *aux)

                train_state, init_h, traj, adv, tgt, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(
                    traj, adv, tgt, init_h, n_envs, config.ppo.num_minibatches, perm_rng
                )
                train_state, losses = jax.lax.scan(_minbatch, train_state, minibatches)
                return (train_state, init_h, traj, adv, tgt, rng), losses

            init_h = ego_policy.init_hstate(n_envs)
            update_state = (train_state, init_h, traj, advantages, targets, rng)
            update_state, _ = jax.lax.scan(
                _update_epoch, update_state, None, config.ppo.update_epochs
            )
            train_state = update_state[0]

            def _mean(x, mask):
                return jnp.where(mask, x, 0).sum() / jnp.maximum(1, mask.sum())

            mask = traj.info.get("returned_episode", jnp.ones_like(traj.reward))
            metric = {"ego_return": _mean(traj.info.get("returned_episode_returns", traj.reward), mask)}
            # Tick the host-side progress bar once per update (fires once per update even
            # under the member vmap -- io_callback batches the lane axis), the same
            # mechanism ippo uses to stream from inside its device scan. The callback must
            # return nothing (result_shape=None): tqdm's update() returns a truthy value,
            # so wrap it to discard the return, or the GPU callback path raises
            # "Mismatched number of outputs from callback".
            if progress_callback is not None:

                def _tick(_u):
                    progress_callback()

                jax.experimental.io_callback(_tick, None, upd)
            runner_state = (train_state, env_state, last_obs, last_done, ego_h, mate_h, upd + 1, rng)
            return runner_state, metric

        init_done = {name: jnp.zeros((n_envs,), dtype=bool) for name in env.agents + ["__all__"]}
        runner_state = (train_state, env_state, obsv, init_done, ego_h0, mate_h0, jnp.asarray(0), rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )
        return {"final_params": runner_state[0].params, "metrics": metrics}

    return train


def _tree_slice(tree, sl):
    return jax.tree.map(lambda x: x[sl], tree)


def _vmap_train_chunked(train, rngs, teammate_params, ego_init_params, ego_aux_ids, chunk):
    """vmap ``train`` over the member axis, in chunks of ``chunk`` (0 = all at once).

    Chunking bounds peak VRAM to ``chunk`` concurrent lanes -- the lever that lets a
    pooled run fit a 6GB GPU (small chunk) or saturate an H100 (chunk=0). Each chunk's
    outputs are concatenated back along the member axis, so the result is identical to
    an unchunked vmap.
    """
    n = int(rngs.shape[0])
    in_axes = (0, 0, 0, None if ego_aux_ids is None else 0)
    vtrain = jax.jit(jax.vmap(train, in_axes=in_axes))
    if chunk <= 0 or chunk >= n:
        return vtrain(rngs, teammate_params, ego_init_params, ego_aux_ids)
    outs = []
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        aux = None if ego_aux_ids is None else ego_aux_ids[sl]
        outs.append(
            vtrain(rngs[sl], _tree_slice(teammate_params, sl), _tree_slice(ego_init_params, sl), aux)
        )
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *outs)


def _train_source(src_path, job, env, base, source_index, logger):
    """Train one best-response per teammate in a single released population.

    Architecture (``actor_type``/``network``/``pop_size``) is taken from the *source's*
    own job -- so the warm-start params load and each population keeps its own actor --
    while the BR training budget (``ppo``/``num_envs``/``total_timesteps``) comes from
    this ``ppo_br`` job. Returns ``(br_params, entries, policy_cls)`` where ``br_params``
    carries a leading member axis in ``entries`` order.
    """
    from pathlib import Path

    from oaht_bench.common.save_load_utils import load_train_run
    from oaht_bench.configs import load_job
    from oaht_bench.population.loading import artifact_dir, population_from_run
    from oaht_bench.population.members import get_member_params, released_members
    from oaht_bench.teammate_gen.runtime import PpoRuntime

    gen = job.generator
    src = Path(src_path)
    src_run = src.parent.parent if src.name == "saved_train_run" else src
    src_job = load_job(src_run / "job.json")
    loaded = population_from_run(src_job, load_train_run(str(artifact_dir(src_run))), env)
    members = [int(m) for m in released_members(src_job, loaded.pop_size)]

    teammate_list, ego_list, entries = [], [], []
    for pos, m in enumerate(members):
        if loaded.paired:
            teammate_list.append(get_member_params(loaded.params, m))  # confederate (fixed)
            ego_list.append(get_member_params(loaded.partner_params, m))  # its br (warm start)
            role = "conf"
        else:
            p = get_member_params(loaded.params, m)
            teammate_list.append(p)  # self member (fixed)
            ego_list.append(p)  # warm-start from itself
            role = "self"
        entries.append(
            {
                "generator": loaded.generator,
                "member": m,
                "role": role,
                "source_index": source_index,
                "pos": pos,
                "source_population_path": str(src_path),
            }
        )
    teammate_params = jax.tree.map(lambda *xs: jnp.stack(xs), *teammate_list)
    ego_init_params = jax.tree.map(lambda *xs: jnp.stack(xs), *ego_list)

    ego_aux_ids = None
    if "conditional_critic" in src_job.generator.actor_type:
        eye = jnp.eye(loaded.pop_size)
        ego_aux_ids = jnp.stack([eye[m] for m in members])  # (members, pop_size) onehot

    rt = PpoRuntime.from_config(
        ppo=gen.ppo,
        network=src_job.generator.network,
        actor_type=src_job.generator.actor_type,
        rollout_length=job.env.rollout_length,
        num_envs=gen.num_envs,
        total_timesteps=gen.total_timesteps,
        num_checkpoints=1,
        num_agents=base.num_agents,
        pop_size=loaded.pop_size,
    )
    from tqdm import tqdm

    chunk = int(gen.members_per_chunk)
    n = len(members)
    n_chunks = 1 if chunk <= 0 or chunk >= n else -(-n // chunk)
    bar = tqdm(
        total=rt.num_updates * n_chunks,
        desc=f"ppo_br {loaded.generator} ({n} BRs)",
        unit="update",
        dynamic_ncols=True,
    )
    train = make_br_train(
        rt, env, loaded.policy_cls, loaded.policy_cls, logger=logger,
        progress_callback=lambda: bar.update(1),
    )
    rngs = jax.random.split(jax.random.PRNGKey(gen.train_seed + source_index), len(members))
    try:
        result = _vmap_train_chunked(
            train, rngs, teammate_params, ego_init_params, ego_aux_ids, chunk
        )
    finally:
        bar.close()
    return result["final_params"], entries, loaded.policy_cls, result["metrics"]


def run_ppo_br(job, logger):
    """Train a best-response ego for every teammate in one or more released populations.

    ``source_population_path`` may be a single run dir or a *list* (pooled mode: the whole
    released roster across generators). Each population's ``self``/``conf`` members become
    fixed teammates; each BR ego is warm-started from the paired ``br`` (paired) or the
    member itself (homogeneous). Populations are trained per-source (they carry different
    actor architectures, which cannot share one ``vmap``), member-chunked for bounded
    VRAM, and saved as one combined artifact: ``br_by_source`` (params per source) plus a
    ``br_manifest.json`` mapping every teammate ``(generator, member, role)`` to its BR.
    The runner skips the diversity cross-play eval for ``ppo_br`` (a BR is scored against
    its own teammate, which this logs).
    """
    import json
    from pathlib import Path

    import numpy as np

    from oaht_bench.common.logging import nonfatal
    from oaht_bench.common.save_load_utils import save_train_run
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.models.population_interface import AgentPopulation

    gen = job.generator
    base = make_env(job.env.env_name, job.env.env_kwargs())
    env = LogWrapper(base)

    sources = gen.source_population_path
    if isinstance(sources, str):
        sources = [sources]

    br_by_source, manifest, all_returns, first_cls = {}, [], [], None
    for i, src_path in enumerate(sources):
        br_params, entries, policy_cls, metrics = _train_source(src_path, job, env, base, i, logger)
        br_by_source[str(i)] = br_params
        manifest.extend(entries)
        all_returns.append((entries, np.asarray(metrics["ego_return"])[:, -1]))
        first_cls = first_cls or policy_cls

    with nonfatal("ppo_br reporting"):
        means = []
        for entries, rets in all_returns:
            for entry, r in zip(entries, rets.tolist(), strict=True):
                logger.log_item(
                    f"BR/return_{entry['generator']}:{entry['member']}:{entry['role']}", float(r)
                )
                means.append(float(r))
        logger.log_item("BR/mean_return", float(np.mean(means)))
        logger.commit()

    save_train_run({"br_by_source": br_by_source}, job.run_dir(), savename="saved_train_run")
    (Path(job.run_dir()) / "br_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    population = AgentPopulation(pop_size=len(manifest), policy_cls=first_cls)
    return br_by_source, population
