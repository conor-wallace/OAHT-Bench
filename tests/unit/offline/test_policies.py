"""The BaseAhtPolicy contract, per baseline.

End-to-end training and evaluation is covered by the parametrized runner tests in
``tests/test_offline_runner.py``; these pin the pure-config surface -- registry
resolution and model construction from a config alone -- for every baseline.
"""

import pytest

from oaht_bench.configs.job import OfflineTrainingConfig
from oaht_bench.offline import get_policy
from oaht_bench.offline.liam import LiamPolicy
from oaht_bench.offline.meliba import MelibaPolicy
from oaht_bench.offline.omis import OmisPolicy
from oaht_bench.offline.tao import TaoPolicy

CASES = [
    ("liam", LiamPolicy),
    ("meliba", MelibaPolicy),
    ("omis", OmisPolicy),
    ("tao", TaoPolicy),
]


def _config(architecture: str, *, resolved: bool = True) -> OfflineTrainingConfig:
    network = {"architecture": architecture}
    if resolved:
        # Stand in for what the runner resolves from the dataset.
        network |= {"obs_dim": 6, "action_dim": 6}
    return OfflineTrainingConfig.model_validate({"network": network})


@pytest.mark.parametrize("architecture,cls", CASES)
def test_registry_resolves_each_architecture(architecture, cls):
    policy = get_policy(_config(architecture))
    assert policy is cls
    assert cls(_config(architecture)).name == architecture


@pytest.mark.parametrize("architecture,cls", CASES)
def test_build_model_from_resolved_config(architecture, cls):
    # Pure-config construction: every baseline builds its flax modules from the
    # config alone, no environment.
    cls(_config(architecture)).build_model()


@pytest.mark.parametrize("architecture,cls", CASES)
def test_build_model_requires_resolved_dims(architecture, cls):
    with pytest.raises(ValueError, match="unresolved"):
        cls(_config(architecture, resolved=False)).build_model()
