"""The training runner's contracts: what it refuses, and what it writes.

Runs are tiny (a couple of gradient steps) -- these check the runner's
guarantees, not that the baselines learn, which the loss tests cover.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oaht_bench.configs import get_preset
from oaht_bench.configs.job import OfflineTrainingConfig, TrainingJob
from oaht_bench.data.schema import EpisodeBatch


def _dataset(tmp_path: Path, n_ep=8, T=14, obs_dim=6, teammates=(0, 1, 2, 3)) -> Path:
    rng = np.random.default_rng(0)
    member_ids = np.array([[0, teammates[i % len(teammates)]] for i in range(n_ep)])
    batch = EpisodeBatch(
        obs=rng.normal(size=(n_ep, 2, T, obs_dim)).astype(np.float32),
        actions=rng.integers(0, 6, size=(n_ep, 2, T)),
        rewards=rng.normal(size=(n_ep, 2, T)).astype(np.float32),
        dones=np.zeros((n_ep, T), dtype=bool),
        valid=np.ones((n_ep, T), dtype=bool),
        avail_actions=np.ones((n_ep, 2, T, 6), dtype=np.float32),
        member_ids=member_ids,
        ego_index=0,
        meta={"generator": "test"},
    )
    return batch.save(tmp_path / "dataset.npz")


def _job(tmp_path: Path, baseline: str, **overrides) -> TrainingJob:
    offline = OfflineTrainingConfig(
        context_length=6,
        stride=3,
        stage1_steps=2,
        stage2_steps=2,
        log_every=1,
        stage2_batch_size=8,
        teammates_per_batch=2,
        windows_per_teammate=4,
        context_trajectories=2,
    )
    return TrainingJob(
        label=f"t_{baseline}",
        env=get_preset("lbf_12x12"),
        dataset_path=str(_dataset(tmp_path)),
        baseline=baseline,
        offline=offline,
        output_dir=str(tmp_path / "out"),
        **overrides,
    )


@pytest.mark.parametrize("baseline", ["liam", "tao"])
def test_runner_trains_and_writes_parameters(tmp_path, baseline):
    """Both baselines complete and leave a loadable artifact plus metrics."""
    from oaht_bench.offline.runner import run

    run_dir = run(_job(tmp_path, baseline))
    assert (run_dir / "params.pkl").exists()
    assert (run_dir / "job.json").exists()

    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    tags = {k for r in records for k in r if "/" in k}
    assert any(t.startswith("Stage1/") for t in tags)
    assert any(t.startswith("Stage2/") for t in tags)

    summary = json.loads((run_dir / "training_summary.json").read_text())
    assert summary["baseline"] == baseline


def test_runner_refuses_an_unimplemented_baseline(tmp_path):
    """The roster in BaselineName is the plan, not what exists.

    Silently training something else would be worse than failing.
    """
    from oaht_bench.offline.runner import run

    with pytest.raises(NotImplementedError, match="has no runner yet"):
        run(_job(tmp_path, "meliba"))


def test_runner_refuses_to_overwrite_a_finished_run(tmp_path):
    """Same guard as teammate generation, on the artifact rather than the directory."""
    from oaht_bench.offline.runner import run

    job = _job(tmp_path, "liam")
    run(job)
    with pytest.raises(FileExistsError, match="already exists"):
        run(job)


def test_runner_saves_parameters_before_reporting():
    """The lesson from teammate generation, pinned rather than assumed.

    A reporting failure after a long run must not be able to discard it, so the
    parameter write has to precede anything that can raise.
    """
    import inspect

    from oaht_bench.offline import runner

    src = inspect.getsource(runner.run)
    assert src.index('params.pkl").open("wb")') < src.index("nonfatal(")
