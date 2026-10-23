"""``Dataset``'s normalisation construction -- in particular ``rtg_scale``,
the option to override the computed return-to-go scale (the dataset-wide RTG
standard deviation) with a fixed constant, matching references (OMIS App.
H.2's ``reward_scale_dict``) that divide by one environment-specific number
rather than a learned statistic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oaht_bench.dataset.dataset import Dataset
from oaht_bench.dataset.schema import Episode
from oaht_bench.dataset.vault import write_vault


def _vault(tmp_path: Path, *, lengths=(4, 6), obs_dim=3, num_actions=4) -> Path:
    rng = np.random.default_rng(0)
    episodes = [
        Episode(
            obs=rng.normal(size=(2, n, obs_dim)).astype(np.float32),
            actions=rng.integers(0, num_actions, size=(2, n)),
            rewards=rng.normal(size=(2, n)).astype(np.float32),
            avail_actions=np.ones((2, n, num_actions), dtype=np.float32),
            dones=np.arange(n) == (n - 1),
        )
        for n in lengths
    ]
    member_ids = np.zeros((len(lengths), 2), dtype=np.int32)
    return write_vault(
        episodes, member_ids, tmp_path / "ds.vlt", ego_index=0, meta={"generator": "test"}
    )


def test_rtg_scale_override_replaces_the_computed_std(tmp_path):
    vault_path = _vault(tmp_path)
    computed = Dataset(vault_path)
    fixed = Dataset(vault_path, rtg_scale=1.0)

    assert fixed.norm.rtg_scale == 1.0
    assert computed.norm.rtg_scale != 1.0  # the real computed std, not a coincidence
    # obs normalisation is untouched by the override -- OMIS's reward_scale_dict
    # only ever touches RTG, never observations.
    np.testing.assert_array_equal(fixed.norm.obs_mean, computed.norm.obs_mean)
    np.testing.assert_array_equal(fixed.norm.obs_std, computed.norm.obs_std)


def test_rtg_scale_default_keeps_todays_computed_value(tmp_path):
    vault_path = _vault(tmp_path)
    ds = Dataset(vault_path)
    assert ds.norm.rtg_scale == pytest.approx(float(ds.batch.norm_stats()[2]))


def test_rtg_scale_is_ignored_without_normalization(tmp_path):
    vault_path = _vault(tmp_path)
    ds = Dataset(vault_path, normalize=False, rtg_scale=1.0)
    assert ds.norm is None
