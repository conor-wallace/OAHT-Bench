"""TaoTrainer._pool must exclude mismatched episodes.

Stage 2's decoder window (``_stage2_batch``, Eq. 5) reads the ego's actions
from the decoder window's own episode as the BC imitation target -- sound
only when that episode's ego is the mate's own dedicated best response.
Mirrors ``tests/unit/offline/omis/test_omis_incontext_adaptation.py``'s
``_prepared_pool`` tests for OMIS's identical ``self._pool`` filter; no
existing TAO test file calls ``.prepare(...)``, so this is a new file rather
than an addition to one of them (each builds its batch standalone via
``SimpleNamespace``, never a real ``TaoTrainer``).
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from oaht_bench.configs.job import OfflineTrainingConfig
from oaht_bench.configs.network import TaoNetworkConfig
from oaht_bench.offline.tao import TaoTrainer


class _NullLogger:
    def log_item(self, *a, **k):
        pass

    def commit(self):
        pass


def _prepared_pool(member_rows, *, T=6, obs_dim=3, action_dim=2):
    """Build a TaoTrainer against a hand-built EpisodeBatch with the given
    (ego, teammate) member_ids rows and return it, prepared, without running
    any training steps."""
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
        network=TaoNetworkConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=8),
        context_length=T,
        stride=T,
        batch_size=4,
        context_trajectories=2,
    )
    trainer = TaoTrainer(cfg)
    trainer.build_model()
    trainer.prepare(
        dataset, _NullLogger(), rng=jax.random.PRNGKey(0), np_rng=np.random.default_rng(0)
    )
    return trainer


def test_tao_pool_only_contains_identity_matched_episodes():
    """TaoTrainer._pool must exclude any episode where the ego seat is not
    the mate's own dedicated best response -- stage 2's BC target would
    otherwise be polluted by a mismatched partner's sub-optimal actions.
    """
    # rows: matched (0,0), (1,1); mismatched (0,1), (1,0).
    trainer = _prepared_pool([[0, 0], [1, 1], [0, 1], [1, 0]])
    episode_ids, _ = trainer._pool
    assert set(episode_ids.tolist()) == {0, 1}


def test_tao_pool_is_unchanged_when_every_episode_is_matched():
    """Regression guard: every dataset collected before this change is 100%
    best-response-matched, so the identity filter must be a no-op on that
    data -- the full pool, not a subset of it."""
    trainer = _prepared_pool([[0, 0], [1, 1], [2, 2]])
    episode_ids, _ = trainer._pool
    assert set(episode_ids.tolist()) == {0, 1, 2}


def test_tao_pool_is_empty_and_fails_loudly_when_every_episode_is_mismatched():
    trainer = _prepared_pool([[0, 1], [1, 0]])
    episode_ids, _ = trainer._pool
    assert len(episode_ids) == 0
    with pytest.raises(ValueError):
        trainer._stage2_batch(0)
