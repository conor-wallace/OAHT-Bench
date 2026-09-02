import pytest

from oaht_bench.configs.job import OfflineTrainingConfig
from oaht_bench.offline import get_policy
from oaht_bench.offline.liam import LiamPolicy


@pytest.fixture
def liam_config() -> OfflineTrainingConfig:
    # obs_dim/action_dim stand in for what the runner resolves from the dataset.
    return OfflineTrainingConfig.model_validate(
        {"network": {"architecture": "liam", "obs_dim": 6, "action_dim": 6}}
    )


class TestLiamPolicy:
    def test_registry_resolves_liam(self, liam_config: OfflineTrainingConfig):
        assert get_policy(liam_config) is LiamPolicy

    def test_construct(self, liam_config: OfflineTrainingConfig):
        policy = LiamPolicy(liam_config)
        assert policy is not None
        assert policy.name == "liam"

    def test_build_model_from_resolved_config(self, liam_config: OfflineTrainingConfig):
        policy = LiamPolicy(liam_config)
        policy.build_model()
        # Pure-config construction: the composed LiamAgent's three flax modules
        # exist without an env.
        assert policy.agent.encoder is not None
        assert policy.agent.decoder is not None
        assert policy.agent.network is not None

    def test_build_model_requires_resolved_dims(self):
        # Without the runner's dim resolution, build_model must fail loudly rather
        # than construct a module with a None feature dimension.
        cfg = OfflineTrainingConfig.model_validate({"network": {"architecture": "liam"}})
        with pytest.raises(ValueError, match="unresolved"):
            LiamPolicy(cfg).build_model()
