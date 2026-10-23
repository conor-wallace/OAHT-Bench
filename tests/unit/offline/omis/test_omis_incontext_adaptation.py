"""D_step/D_epi, deployment-stage -- behavioral check that OMIS's restored
opponent-conditioning actually changes what it predicts, not just that it runs.

Same structure, same synthetic "mirror the opponent" dataset, and the same
``encode_ocw`` production call site as
``tests/unit/offline/tao/test_incontext_adaptation.py`` -- the scenario those
tests use is deliberately hard without context (the ego's own observations
are pure noise, carrying no signal about which opponent is present), so a
predicted action that tracks the opponent only via D_epi/D_step is direct
evidence the mechanism is live, not a reimplementation of it.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from oaht_bench.configs.job import OfflineTrainingConfig
from oaht_bench.configs.network import OmisNetworkConfig
from oaht_bench.offline.incontext_eval import OpponentContextWindow
from oaht_bench.offline.omis import OmisTrainer, _shift_right


class _NullLogger:
    def log_item(self, *a, **k):
        pass

    def commit(self):
        pass


def _mirror_opponent_batch(obs_dim=3, action_dim=2, T=6, n_ep_per_teammate=6, seed=0):
    """Two opponents, deterministic and trivially distinguishable *by their
    actions only* -- observations are noise, uncorrelated with teammate
    identity or the ego label, so there is no shortcut around using the
    context. Identical to
    ``tests/unit/offline/tao/test_incontext_adaptation.py``'s helper of the
    same name (duplicated rather than imported: this package has no
    ``__init__.py``, so cross-test-file imports aren't how this repo's test
    suite is structured).

    ``member_rows`` labels every episode ``[teammate, teammate]`` -- matched,
    not ``[0, teammate]`` -- because ``OmisTrainer.prepare``'s decoder pool
    now filters to identity-matched episodes (``member_ids[:,ego] ==
    member_ids[:,mate]``); this fixture tests in-context adaptation via
    D_epi/D_step, not the mismatch filter itself, so every episode must
    survive that filter or half the decoder training signal silently vanishes.
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
            member_rows.append([teammate, teammate])
    return EpisodeBatch(episodes=episodes, member_ids=np.array(member_rows), ego_index=0, meta={})


def _identity_by_observation_batch(
    obs_dim=3, action_dim=2, T=6, n_ep_per_teammate=8, seed=0, signature=3.0
):
    """Two opponents distinguishable *only by their observation stream*
    (``mate_obs``, what D_epi's ``epi_state`` reads) -- unlike
    ``_mirror_opponent_batch``, the opponent's *action* is uniform noise,
    independent of identity, so D_step (which only ever sees opponent
    actions) cannot shortcut this task the way it can mirror-the-opponent.
    The ego label is tied directly to identity (not to the opponent's
    noisy current-episode action), so the only usable signal is D_epi.

    ``member_rows`` labels every episode ``[teammate, teammate]`` -- matched
    -- for the same reason ``_mirror_opponent_batch`` does: ``OmisTrainer.
    prepare``'s decoder pool now filters to identity-matched episodes, and
    this fixture is testing D_epi's signal, not that filter.
    """
    from oaht_bench.dataset.schema import Episode, EpisodeBatch

    rng = np.random.default_rng(seed)
    episodes, member_rows = [], []
    for teammate in (0, 1):
        offset = signature if teammate else -signature
        for _ in range(n_ep_per_teammate):
            mate_obs = rng.normal(loc=offset, scale=0.1, size=(T, obs_dim)).astype(np.float32)
            ego_obs = rng.normal(size=(T, obs_dim)).astype(np.float32)  # no identity shortcut here
            mate_action = rng.integers(0, action_dim, size=T)  # noise: uncorrelated with identity
            ego_action = np.full(T, teammate, dtype=np.int64)  # the label: identity, not mimicry
            episodes.append(
                Episode(
                    obs=np.stack([ego_obs, mate_obs]),
                    actions=np.stack([ego_action, mate_action]),
                    rewards=np.zeros((2, T), dtype=np.float32),
                    avail_actions=np.ones((2, T, action_dim), dtype=np.float32),
                    dones=np.zeros(T, dtype=bool),
                )
            )
            member_rows.append([teammate, teammate])
    return EpisodeBatch(episodes=episodes, member_ids=np.array(member_rows), ego_index=0, meta={})


def _train_tiny_omis(
    batch,
    *,
    T,
    obs_dim,
    action_dim,
    hidden_dim=16,
    context_trajectories=3,
    epi_length=6,
    steps=300,
    vf_coef=0.5,
    oppo_pi_coef=0.8,
):
    cfg = OfflineTrainingConfig(
        network=OmisNetworkConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim),
        context_length=T,
        # stage1_steps/stage1_learning_rate are unused -- OmisTrainer.train_stage_1
        # is a no-op (see its own docstring); everything real runs in stage 2.
        stage2_steps=steps,
        batch_size=8,
        context_trajectories=context_trajectories,
        epi_length=epi_length,
        stride=T,
        stage2_learning_rate=1e-2,
        normalize_observations=False,
        vf_coef=vf_coef,
        oppo_pi_coef=oppo_pi_coef,
    )
    trainer = OmisTrainer(cfg)
    trainer.build_model()
    dataset = SimpleNamespace(batch=batch, ego=0, mate=1, norm=None)
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
    """``encode_ocw`` is the production call site
    (:func:`~oaht_bench.offline.incontext_eval._context_params` does exactly
    this merge at real evaluation time), but unlike TAO's ``act`` -- which
    reads ``context``/``context_mask`` straight off ``params["stage2"]`` --
    OMIS's ``act`` takes ``epi_state``/``epi_mate_actions``/``epi_mask`` as
    explicit kwargs (see :meth:`OmisAgent.act`'s own docstring), so they have
    to be pulled back out of ``stage2`` here, the same way
    :meth:`OmisAgent.get_action_with_mate_history` does for every real step.

    D_step (``mate_actions``) is passed as all-"unknown" ("no history
    observed yet", training's own convention for a fresh episode -- see
    ``offline/omis._shift_right``), not omitted: the model is trained with
    D_step's token group always present (never a shorter, group-of-3
    sequence), so dropping the group entirely here would evaluate an input
    shape the model never saw rather than isolating D_epi's contribution.
    Only D_epi (the OCW) can distinguish the two opponents under this
    all-unknown D_step.
    """
    no, ac, rw, ts, mk = ocw.arrays()
    epi = agent.encode_ocw(params, no, ac, rw, mk, ts)

    ego_obs = jnp.zeros((1, T, obs_dim))
    ego_rtg = jnp.zeros((1, T))
    ego_actions = jnp.full((1, T), -10, dtype=jnp.int32)
    ego_mask = jnp.ones((1, T), dtype=bool)
    ego_timesteps = jnp.arange(1, T + 1)[None]
    mate_actions = jnp.full((1, T), -10, dtype=jnp.int32)
    logits = agent.act(
        params,
        ego_rtg,
        ego_obs,
        ego_actions,
        timesteps=ego_timesteps,
        mask=ego_mask,
        mate_actions=mate_actions,
        epi_state=epi["epi_state"],
        epi_mate_actions=epi["epi_mate_actions"],
        epi_mask=epi["epi_mask"],
    )
    return int(jnp.argmax(logits[0, -1]))


def test_in_context_adaptation_shifts_the_predicted_action_toward_the_observed_opponent():
    """D_epi, isolated: D_step carries no usable signal in this dataset (the
    opponent's current-episode action is uniform noise), so a correct,
    identity-tracking prediction can only come from D_epi reading the OCW's
    opponent *observations*. ``vf_coef``/``oppo_pi_coef`` are zeroed for this
    training run only -- the imitator's label here is pure noise by
    construction (there is nothing to imitate), and its gradient through the
    shared backbone would otherwise inject noise that swamps the actor's
    much weaker D_epi-dependent signal, which is what this test isolates.
    """
    obs_dim, action_dim, T, C = 3, 2, 6, 3
    batch = _identity_by_observation_batch(obs_dim=obs_dim, action_dim=action_dim, T=T)
    trainer, params = _train_tiny_omis(
        batch,
        T=T,
        obs_dim=obs_dim,
        action_dim=action_dim,
        context_trajectories=C,
        # Matches the OCW's own C*T below -- D_epi's H is independent of
        # context_length in general (Appendix C; see offline/omis.py's
        # build_d_epi), but this test's signal is in the OCW's per-trajectory
        # horizon T, so H is set here to reproduce exactly that at train time.
        epi_length=C * T,
        steps=400,
        vf_coef=0.0,
        oppo_pi_coef=0.0,
    )
    agent = trainer.agent

    rng = np.random.default_rng(1)
    ocw_opponent_0 = OpponentContextWindow(capacity=C, horizon=T, obs_dim=obs_dim)
    ocw_opponent_1 = OpponentContextWindow(capacity=C, horizon=T, obs_dim=obs_dim)
    for _ in range(C):
        noise_actions = rng.integers(0, action_dim, size=T)
        ocw_opponent_0.append(
            rng.normal(loc=-3.0, scale=0.1, size=(T, obs_dim)),
            noise_actions,
            np.zeros(T),
            np.arange(1, T + 1),
        )
        ocw_opponent_1.append(
            rng.normal(loc=3.0, scale=0.1, size=(T, obs_dim)),
            noise_actions,
            np.zeros(T),
            np.arange(1, T + 1),
        )

    pred_0 = _predicted_action(agent, params, ocw_opponent_0, T=T, obs_dim=obs_dim)
    pred_1 = _predicted_action(agent, params, ocw_opponent_1, T=T, obs_dim=obs_dim)

    assert pred_0 == 0, "observing opponent 0's OCW (D_epi) should predict action 0 (identity)"
    assert pred_1 == 1, "observing opponent 1's OCW (D_epi) should predict action 1 (identity)"


def test_shift_right_never_lets_the_label_leak_into_its_own_input():
    """D_step's conditioning input at position t must be the opponent's action
    at t-1, never t -- otherwise the imitator's label and the backbone's input
    are the same tensor slice and the loss is trivially satisfiable.
    """
    mate_actions = jnp.asarray([[1, 2, 3, 4]])
    shifted = _shift_right(mate_actions, fill=-10)
    assert shifted.shape == mate_actions.shape
    assert shifted[0].tolist() == [-10, 1, 2, 3]
    # the label at every position is still the original, unshifted stream
    assert mate_actions[0].tolist() == [1, 2, 3, 4]


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
    """Exercises evaluate_incontext (and OmisAgent.get_action_with_mate_history
    within it) against a real env, the same check TAO's own test runs -- pins
    the return shape and that episode 1 (empty OCW, empty D_step) is a real
    number rather than crashing or nan.
    """
    from oaht_bench.configs import get_preset
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.models.omis_agent import OmisAgent
    from oaht_bench.offline.incontext_eval import evaluate_incontext

    env_cfg = get_preset("lbf_12x12").model_copy(update={"rollout_length": 8})
    env = LogWrapper(make_env(env_cfg.env_name, env_cfg.env_kwargs()))
    obs_dim = int(env.observation_space(env.agents[1]).shape[0])
    action_dim = int(env.action_space(env.agents[1]).n)
    T, C, num_episodes = 6, 3, 4

    batch = _mirror_opponent_batch(obs_dim=obs_dim, action_dim=action_dim, T=T)
    trainer, params = _train_tiny_omis(
        batch, T=T, obs_dim=obs_dim, action_dim=action_dim, context_trajectories=C, steps=150
    )
    agent = OmisAgent(trainer.config, context_length=T, target_return=0.0, normalization=None)
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

    assert set(mean) == set(curve) == {"opponent_0", "opponent_1"}
    assert (
        ancillary is None and ancillary_curve is None
    )  # OMIS has no decoder-based ancillary probe
    for label in ("opponent_0", "opponent_1"):
        assert len(curve[label]) == num_episodes
        assert not any(np.isnan(v) for v in curve[label])


def _prepared_pool(member_rows, *, T=6, obs_dim=3, action_dim=2):
    """Build an OmisTrainer against a hand-built EpisodeBatch with the given
    (ego, teammate) member_ids rows and return its prepared self._pool,
    without running any training steps."""
    from oaht_bench.dataset.schema import Episode, EpisodeBatch

    episodes = [
        Episode(
            obs=np.zeros((2, T, obs_dim), dtype=np.float32),
            actions=np.zeros((2, T), dtype=np.int64),
            rewards=np.zeros((2, T), dtype=np.float32),
            avail_actions=np.ones((2, T, action_dim), dtype=np.float32),
            dones=np.zeros(T, dtype=bool),
        )
        for _ in member_rows
    ]
    batch = EpisodeBatch(
        episodes=episodes, member_ids=np.array(member_rows), ego_index=0, meta={}
    )
    dataset = SimpleNamespace(batch=batch, ego=0, mate=1, norm=None)

    cfg = OfflineTrainingConfig(
        network=OmisNetworkConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=8),
        context_length=T,
        stride=T,
        batch_size=4,
        context_trajectories=2,
        epi_length=4,
    )
    trainer = OmisTrainer(cfg)
    trainer.build_model()
    trainer.prepare(
        dataset, _NullLogger(), rng=jax.random.PRNGKey(0), np_rng=np.random.default_rng(0)
    )
    return trainer


def test_omis_pool_only_contains_identity_matched_episodes():
    """OmisTrainer._pool must exclude any episode where the ego seat is not
    teammate k's own dedicated best response -- D_step's BC target would
    otherwise be polluted by a mismatched partner's sub-optimal actions.
    """
    # rows: matched (0,0), (1,1); mismatched (0,1), (1,0).
    trainer = _prepared_pool([[0, 0], [1, 1], [0, 1], [1, 0]])
    episode_ids, _ = trainer._pool
    assert set(episode_ids.tolist()) == {0, 1}


def test_omis_pool_is_unchanged_when_every_episode_is_matched():
    """Regression guard: every dataset collected before this change is
    100% best-response-matched, so the identity filter must be a no-op on
    that data -- the full pool, not a subset of it."""
    trainer = _prepared_pool([[0, 0], [1, 1], [2, 2]])
    episode_ids, _ = trainer._pool
    assert set(episode_ids.tolist()) == {0, 1, 2}


def test_omis_pool_is_empty_and_fails_loudly_when_every_episode_is_mismatched():
    import pytest

    trainer = _prepared_pool([[0, 1], [1, 0]])
    episode_ids, _ = trainer._pool
    assert len(episode_ids) == 0
    with pytest.raises(ValueError):
        trainer._sample_batch(0)
