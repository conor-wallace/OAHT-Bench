"""Per-teammate RTG evaluation targets (§8) -- the crossplay-matrix path and
its dataset-wide fallback, and the agent-side seam that lets a target change
between teammates.
"""

from __future__ import annotations

import numpy as np
import pytest

from oaht_bench.configs.job import OfflineTrainingConfig
from oaht_bench.dataset.dataset import Normalization
from oaht_bench.models.bc_agent import BcAgent
from oaht_bench.offline.evaluate import crossplay_target_returns, resolve_target_returns


def _matrix_path(tmp_path):
    from oaht_bench.population.pooled_crossplay import RosterEntry, save_pooled

    matrix = np.array(
        [
            [10.0, 1.0, 2.0],
            [3.0, 20.0, 4.0],
            [5.0, 6.0, 30.0],
        ]
    )
    roster = [
        RosterEntry("fcp", 0, "self", params=None, policy_cls=None),
        RosterEntry("comedi", 1, "self", params=None, policy_cls=None),
        RosterEntry("brdiv", 2, "conf", params=None, policy_cls=None),
    ]
    return save_pooled(matrix, roster, tmp_path / "pooled_crossplay.npz", meta={"env": "test"})


def _teammates(labels):
    return [(label, None, None) for label in labels]


def test_crossplay_target_returns_reads_the_column_max(tmp_path):
    path = str(_matrix_path(tmp_path))
    teammates = _teammates(["fcp:0:self", "comedi:1:self", "brdiv:2:conf"])
    out = crossplay_target_returns(path, teammates)
    # column 0 max(10,3,5)=10; column 1 max(1,20,6)=20; column 2 max(2,4,30)=30
    assert out == pytest.approx({"fcp:0:self": 10.0, "comedi:1:self": 20.0, "brdiv:2:conf": 30.0})


def test_crossplay_target_returns_raises_on_unknown_teammate(tmp_path):
    path = str(_matrix_path(tmp_path))
    with pytest.raises(ValueError, match="not a column"):
        crossplay_target_returns(path, _teammates(["fcp:99:self"]))


class _FakeBatch:
    """Just enough of EpisodeBatch/VaultReader for dataset_target_return."""

    def __init__(self, returns):
        self._returns = np.asarray(returns)
        self.ego_index = 0

    def episode_returns(self):
        return self._returns


def test_resolve_target_returns_prefers_the_crossplay_matrix(tmp_path):
    path = str(_matrix_path(tmp_path))
    teammates = _teammates(["fcp:0:self", "comedi:1:self"])
    out = resolve_target_returns({"pooled_matrix_path": path}, teammates)
    assert out == pytest.approx({"fcp:0:self": 10.0, "comedi:1:self": 20.0})


def test_resolve_target_returns_falls_back_without_a_matrix():
    batch = _FakeBatch([[1.0, 0], [4.0, 0], [2.0, 0]])
    teammates = _teammates(["a", "b"])
    out = resolve_target_returns({}, teammates, fallback_batch=batch)
    # dataset-wide max (4.0), broadcast to every teammate -- today's behaviour.
    assert out == {"a": 4.0, "b": 4.0}


def test_resolve_target_returns_requires_a_fallback_batch_when_no_matrix():
    with pytest.raises(ValueError, match="fallback_batch"):
        resolve_target_returns({}, _teammates(["a"]))


def test_resolve_target_returns_applies_the_normalisation(tmp_path):
    path = str(_matrix_path(tmp_path))
    norm = Normalization(obs_mean=np.zeros(1), obs_std=np.ones(1), rtg_scale=2.0)
    out = resolve_target_returns(
        {"pooled_matrix_path": path}, _teammates(["fcp:0:self"]), norm=norm
    )
    assert out == pytest.approx({"fcp:0:self": 5.0})  # 10.0 / rtg_scale=2.0


def test_set_target_return_changes_what_init_hstate_seeds():
    config = OfflineTrainingConfig.model_validate(
        {"network": {"architecture": "bc", "obs_dim": 6, "action_dim": 6}}
    )
    agent = BcAgent(config, context_length=4, target_return=1.0)
    assert float(agent.init_hstate(1).rtg) == pytest.approx(1.0)

    agent.set_target_return(5.0)
    assert float(agent.init_hstate(1).rtg) == pytest.approx(5.0)
