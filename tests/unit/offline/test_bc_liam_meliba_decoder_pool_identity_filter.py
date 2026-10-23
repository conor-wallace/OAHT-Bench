"""BC/LIAM/MeLIBA's decoder-window pool must exclude mismatched episodes.

Each of these clones the ego's own actions as a "near-optimal response"
target (BC and LIAM's/MeLIBA's stage-2 policy), the same soundness
requirement as OMIS's D_step and TAO's stage 2 -- see
``tests/unit/offline/omis/test_omis_incontext_adaptation.py``'s
``_prepared_pool`` and ``tests/unit/offline/tao/test_decoder_pool_identity_filter.py``
for the OMIS/TAO equivalents this mirrors. No existing LIAM/MeLIBA/BC test
file calls ``.prepare(...)`` against a real dataset, so this is a new file.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from oaht_bench.configs.job import OfflineTrainingConfig
from oaht_bench.configs.network import BcNetworkConfig, LiamNetworkConfig, MelibaNetworkConfig
from oaht_bench.offline.bc import BcTrainer
from oaht_bench.offline.liam import LiamTrainer
from oaht_bench.offline.meliba import MelibaTrainer


class _NullLogger:
    def log_item(self, *a, **k):
        pass

    def commit(self):
        pass


def _batch(member_rows, *, T=6, obs_dim=3, action_dim=2):
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
    return EpisodeBatch(episodes=episodes, member_ids=np.array(member_rows), ego_index=0, meta={})


def _prepared_pool(
    trainer_cls, network_cfg, member_rows, *, T=6, obs_dim=3, action_dim=2, batch_size=1, **extra_cfg
):
    batch = _batch(member_rows, T=T, obs_dim=obs_dim, action_dim=action_dim)
    dataset = SimpleNamespace(batch=batch, ego=0, mate=1, norm=None)

    cfg = OfflineTrainingConfig(
        network=network_cfg, context_length=T, stride=T, batch_size=batch_size, **extra_cfg
    )
    trainer = trainer_cls(cfg)
    trainer.build_model()
    trainer.prepare(
        dataset, _NullLogger(), rng=jax.random.PRNGKey(0), np_rng=np.random.default_rng(0)
    )
    return trainer


# rows: matched (0,0), (1,1); mismatched (0,1), (1,0).
_MIXED_ROWS = [[0, 0], [1, 1], [0, 1], [1, 0]]


@pytest.mark.parametrize(
    "trainer_cls,network_cfg_factory,pool_attr",
    [
        (LiamTrainer, lambda: LiamNetworkConfig(obs_dim=3, action_dim=2, hidden_dim=8), "_pool"),
        (MelibaTrainer, lambda: MelibaNetworkConfig(obs_dim=3, action_dim=2, hidden_dim=8), "_pool"),
    ],
)
def test_pool_only_contains_identity_matched_episodes(trainer_cls, network_cfg_factory, pool_attr):
    trainer = _prepared_pool(trainer_cls, network_cfg_factory(), _MIXED_ROWS)
    episode_ids, _ = getattr(trainer, pool_attr)
    assert set(episode_ids.tolist()) == {0, 1}


@pytest.mark.parametrize(
    "trainer_cls,network_cfg_factory,pool_attr",
    [
        (LiamTrainer, lambda: LiamNetworkConfig(obs_dim=3, action_dim=2, hidden_dim=8), "_pool"),
        (MelibaTrainer, lambda: MelibaNetworkConfig(obs_dim=3, action_dim=2, hidden_dim=8), "_pool"),
    ],
)
def test_pool_is_unchanged_when_every_episode_is_matched(trainer_cls, network_cfg_factory, pool_attr):
    trainer = _prepared_pool(trainer_cls, network_cfg_factory(), [[0, 0], [1, 1], [2, 2]])
    episode_ids, _ = getattr(trainer, pool_attr)
    assert set(episode_ids.tolist()) == {0, 1, 2}


@pytest.mark.parametrize(
    "trainer_cls,network_cfg_factory",
    [
        (LiamTrainer, lambda: LiamNetworkConfig(obs_dim=3, action_dim=2, hidden_dim=8)),
        (MelibaTrainer, lambda: MelibaNetworkConfig(obs_dim=3, action_dim=2, hidden_dim=8)),
    ],
)
def test_pool_fails_loudly_when_too_small_for_the_batch_size(trainer_cls, network_cfg_factory):
    with pytest.raises(ValueError, match="identity-matched filter"):
        _prepared_pool(trainer_cls, network_cfg_factory(), _MIXED_ROWS, batch_size=4)


def test_bc_pool_only_contains_identity_matched_episodes():
    """BC intersects the identity filter with its own top_return_quantile
    filter -- at the default quantile=1.0 (keep every episode by return),
    the identity filter is the only thing excluding the mismatched rows."""
    trainer = _prepared_pool(
        BcTrainer, BcNetworkConfig(obs_dim=3, action_dim=2, hidden_dim=8), _MIXED_ROWS, batch_size=2
    )
    assert set(trainer._episode_ids.tolist()) == {0, 1}


def test_bc_pool_is_empty_and_fails_loudly_when_every_episode_is_mismatched():
    with pytest.raises(ValueError, match="identity-matched filter"):
        _prepared_pool(BcTrainer, BcNetworkConfig(obs_dim=3, action_dim=2, hidden_dim=8), [[0, 1], [1, 0]])
