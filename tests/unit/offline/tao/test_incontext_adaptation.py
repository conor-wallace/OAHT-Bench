"""Eq. 6, deployment-stage in-context opponent adapting -- behavioral check.

Every other test in this package checks shapes, masks, and losses; none of
them can tell you whether in-context learning is *doing anything*. This one
trains a tiny, real TAO (stage 1 and stage 2, no mocking) on a dataset
deliberately built so the only way to pick the correct ego action is to
recognise which of two opponents is present: teammate 0 always plays action 0,
teammate 1 always plays action 1, the "near-optimal" ego label is to mirror
whichever teammate is present, and the ego's own observations are pure noise
carrying no signal at all. If TAO's in-context mechanism (OpponentContextWindow
-> encode_ocw -> cross-attention -> act, the production path, not a
reimplementation of it) works, feeding it a context of teammate 0's trajectories
versus teammate 1's must produce different predicted actions. If it doesn't,
the model has nothing else to go on and both predictions collapse to the same
thing.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from oaht_bench.configs.job import OfflineTrainingConfig
from oaht_bench.configs.network import TaoNetworkConfig
from oaht_bench.dataset.dataset import TeammateIndex, _build_windows
from oaht_bench.offline.incontext_eval import OpponentContextWindow
from oaht_bench.offline.tao import TaoTrainer


class _NullLogger:
    def log_item(self, *a, **k):
        pass

    def commit(self):
        pass


def _mirror_opponent_windows(obs_dim=3, action_dim=2, T=6, n_ep_per_teammate=6, seed=0):
    """Two opponents, deterministic and trivially distinguishable *by their
    actions only* -- observations are noise, uncorrelated with teammate
    identity or the ego label, so there is no shortcut around using the
    context.
    """
    from oaht_bench.dataset.schema import Episode, EpisodeBatch

    rng = np.random.default_rng(seed)
    episodes, member_rows = [], []
    for teammate in (0, 1):
        for _ in range(n_ep_per_teammate):
            mate_action = np.full(T, teammate, dtype=np.int64)
            ego_action = mate_action.copy()  # the "near-optimal" label: mirror the opponent
            episodes.append(
                Episode(
                    obs=rng.normal(size=(2, T, obs_dim)).astype(np.float32),
                    actions=np.stack([ego_action, mate_action]),
                    rewards=np.zeros((2, T), dtype=np.float32),
                    avail_actions=np.ones((2, T, action_dim), dtype=np.float32),
                    dones=np.zeros(T, dtype=bool),
                )
            )
            member_rows.append([0, teammate])
    return _build_windows(
        EpisodeBatch(episodes=episodes, member_ids=np.array(member_rows), ego_index=0, meta={}),
        context_length=T,
        stride=T,
    )


def _train_tiny_tao(w, *, obs_dim, action_dim, hidden_dim=16, context_trajectories=3, steps=300):
    cfg = OfflineTrainingConfig(
        network=TaoNetworkConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim),
        context_length=w.context_length,
        stage1_steps=steps,
        stage2_steps=steps,
        batch_size=8,
        context_trajectories=context_trajectories,
        stage1_learning_rate=1e-2,
        stage2_learning_rate=1e-2,
        normalize_observations=False,  # synthetic obs are already unit noise; irrelevant to the label anyway
    )
    trainer = TaoTrainer(cfg)
    trainer.build_model()
    dataset = SimpleNamespace(windows=w, index=TeammateIndex.build(w))
    trainer.prepare(
        dataset,
        _NullLogger(),
        rng=jax.random.PRNGKey(0),
        np_rng=np.random.default_rng(0),
        num_seeds=1,
    )
    stage1_params = trainer.train_stage_1()
    stage2_out = trainer.train_stage_2(stage1_params)
    return trainer, {"stage1": stage1_params, "stage2": stage2_out}


def _predicted_action(agent, params, ocw, *, T, obs_dim):
    """Swap ``ocw``'s encoding into the context, the same way
    ``incontext_eval._context_params`` does at real evaluation time, then read
    the ego's predicted action off the last ("now") position.
    """
    no, ac, rw, ts, mk = ocw.arrays()
    context, context_mask = agent.encode_ocw(params, no, ac, rw, mk, ts)
    stage2 = {**params["stage2"], "context": context, "context_mask": context_mask}
    params_ep = {**params, "stage2": stage2}

    ego_obs = jnp.zeros(
        (1, T, obs_dim)
    )  # noise-level input either way -- carries no signal by construction
    ego_rtg = jnp.zeros((1, T))
    ego_actions = jnp.full((1, T), -10, dtype=jnp.int32)
    ego_mask = jnp.ones((1, T), dtype=bool)
    ego_timesteps = jnp.arange(1, T + 1)[None]
    logits = agent.act(
        params_ep, ego_rtg, ego_obs, ego_actions, timesteps=ego_timesteps, mask=ego_mask
    )
    return int(jnp.argmax(logits[0, -1]))


def test_in_context_adaptation_shifts_the_predicted_action_toward_the_observed_opponent():
    obs_dim, action_dim, T, C = 3, 2, 6, 3
    w = _mirror_opponent_windows(obs_dim=obs_dim, action_dim=action_dim, T=T)
    trainer, params = _train_tiny_tao(
        w, obs_dim=obs_dim, action_dim=action_dim, context_trajectories=C
    )
    agent = trainer.agent

    rng = np.random.default_rng(1)
    ocw_opponent_0 = OpponentContextWindow(capacity=C, horizon=T, obs_dim=obs_dim)
    ocw_opponent_1 = OpponentContextWindow(capacity=C, horizon=T, obs_dim=obs_dim)
    for _ in range(C):
        ocw_opponent_0.append(
            rng.normal(size=(T, obs_dim)),
            np.zeros(T, dtype=np.int64),
            np.zeros(T),
            np.arange(1, T + 1),
        )
        ocw_opponent_1.append(
            rng.normal(size=(T, obs_dim)),
            np.ones(T, dtype=np.int64),
            np.zeros(T),
            np.arange(1, T + 1),
        )

    pred_0 = _predicted_action(agent, params, ocw_opponent_0, T=T, obs_dim=obs_dim)
    pred_1 = _predicted_action(agent, params, ocw_opponent_1, T=T, obs_dim=obs_dim)

    assert pred_0 == 0, "observing opponent 0's trajectories should predict action 0 (mirror)"
    assert pred_1 == 1, "observing opponent 1's trajectories should predict action 1 (mirror)"


class _FixedActionPolicy:
    """Always plays the same action, ignoring env state -- a synthetic opponent for
    exercising evaluate_incontext's real env-rollout path end to end."""

    def __init__(self, action):
        self.action = action

    def init_hstate(self, batch_size, aux_info=None):
        return None

    def get_action(
        self, *, params, obs, done, avail_actions, hstate, rng, env_state=None, test_mode=False
    ):
        return jnp.array([self.action], dtype=jnp.int32), hstate


def test_evaluate_incontext_produces_a_real_per_episode_curve_not_nan():
    """Exercises evaluate_incontext itself (not just encode_ocw/act in isolation) --
    the one piece of Eq. 6 the rest of this package deliberately bypasses, since it
    needs a real env. Pins the 5-tuple return shape added when the per-episode
    ancillary-accuracy curve was added, and that episode 1 (empty OCW) is a real
    number, not nan -- nan only appears for a literally zero-length episode, a
    mistake this test caught: an empty context still produces a prediction, just
    likely a poor, context-blind one.
    """
    from oaht_bench.configs import get_preset
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.models.tao_agent import TaoAgent
    from oaht_bench.offline.incontext_eval import evaluate_incontext

    env_cfg = get_preset("lbf_12x12").model_copy(update={"rollout_length": 8})
    env = LogWrapper(make_env(env_cfg.env_name, env_cfg.env_kwargs()))
    obs_dim = int(env.observation_space(env.agents[1]).shape[0])
    action_dim = int(env.action_space(env.agents[1]).n)
    T, C, num_episodes = 6, 3, 4

    w = _mirror_opponent_windows(obs_dim=obs_dim, action_dim=action_dim, T=T)
    trainer, params = _train_tiny_tao(
        w, obs_dim=obs_dim, action_dim=action_dim, context_trajectories=C, steps=150
    )
    # A deployment-configured agent sharing trainer's architecture (trainer.agent,
    # built for training only, has no context_length/normalization for act()).
    agent = TaoAgent(trainer.config, context_length=T, target_return=0.0, normalization=None)
    agent.build_model()

    teammates = [
        ("opponent_0", None, _FixedActionPolicy(0)),
        ("opponent_1", None, _FixedActionPolicy(1)),
    ]
    target_returns = {"opponent_0": 0.0, "opponent_1": 0.0}

    mean, curve, ancillary, ancillary_floor, ancillary_curve = evaluate_incontext(
        agent,
        params,
        env,
        teammates,
        max_episode_steps=env_cfg.rollout_length,
        num_episodes=num_episodes,
        ocw_size=C,
        obs_dim=obs_dim,
        rng=jax.random.PRNGKey(0),
        target_returns=target_returns,
    )

    assert set(mean) == set(curve) == set(ancillary_curve) == {"opponent_0", "opponent_1"}
    for label in ("opponent_0", "opponent_1"):
        assert len(curve[label]) == num_episodes
        assert len(ancillary_curve[label]) == num_episodes
        # episode 1 (empty OCW) is a real prediction, not a skipped/nan one
        assert not np.isnan(ancillary_curve[label][0])
    assert 0.0 <= ancillary_floor <= 1.0
