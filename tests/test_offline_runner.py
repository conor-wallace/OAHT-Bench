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


@pytest.mark.parametrize("baseline", ["liam", "tao"])
def test_runner_logs_accuracies_and_evaluation_returns(tmp_path, baseline):
    """A falling loss is not evidence the policy plays.

    Three metric families have to appear: action-prediction accuracy on the ego
    (comparable across baselines in a way nats are not), teammate-action
    accuracy from the representation stage (whether the auxiliary task worked),
    and episode returns per teammate from real rollouts.
    """
    from oaht_bench.offline.runner import run

    job = _job(tmp_path, baseline)
    job = job.model_copy(update={"offline": job.offline.model_copy(update={"eval_episodes": 1})})
    run_dir = run(job)

    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    tags = {k for r in records for k in r if "/" in k}

    assert "Stage2/action_accuracy" in tags
    # the representation stage reports whether it can name the teammate's action
    assert any("accuracy" in t for t in tags if t.startswith("Stage1/"))

    # This fixture's dataset carries no population, so rollouts cannot run. That
    # must be recorded as a skip with a reason rather than left as a null
    # indistinguishable from a crash.
    summary = json.loads((run_dir / "training_summary.json").read_text())
    assert summary["eval"] is None
    assert "population_run" in summary["eval_skipped"]


def test_evaluation_target_return_comes_from_the_dataset(tmp_path):
    """The reference reads per-opponent targets from a config table; we have none.

    Conditioning on the dataset's best episode return is the Decision
    Transformer convention -- ask for the best behaviour the data contains.
    """
    from oaht_bench.data.schema import EpisodeBatch
    from oaht_bench.offline.evaluate import dataset_target_return

    batch = EpisodeBatch.load(_dataset(tmp_path))
    target = dataset_target_return(batch)
    ego_returns = batch.episode_returns()[:, batch.ego_index]
    assert target == pytest.approx(float(np.max(ego_returns)))
    assert dataset_target_return(batch, quantile=0.5) <= target


def test_eval_scores_report_the_worst_teammate_not_just_the_mean():
    """Ad-hoc teamwork is about the partners you did not train for."""
    from oaht_bench.offline.evaluate import EvalScores

    s = EvalScores(
        per_teammate={0: 1.0, 1: 0.0, 2: 0.5},
        per_teammate_stderr={0: 0.0, 1: 0.0, 2: 0.0},
        episodes_per_teammate=5,
        target_return=1.0,
    )
    assert s.mean_return == pytest.approx(0.5)
    assert s.worst_teammate_return == 0.0
    # equal weight per teammate, so coverage imbalance cannot tilt the mean
    assert "worst" in s.describe()
