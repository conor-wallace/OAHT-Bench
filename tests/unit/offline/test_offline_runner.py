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
from oaht_bench.dataset.schema import Episode, EpisodeBatch
from oaht_bench.dataset.vault import read_vault, write_vault


def _dataset(tmp_path: Path, n_ep=8, T=14, obs_dim=6, teammates=(0, 1, 2, 3)) -> Path:
    """Write a small vault from ragged episodes, the way collection does.

    ``member_ids`` labels every episode ``[teammate, teammate]`` -- matched --
    since every baseline's decoder/BC window now filters to identity-matched
    episodes (``identity_matched_episode_ids``); this fixture is testing the
    runner's contracts, not that filter, so every episode must survive it.
    """
    rng = np.random.default_rng(0)
    member_ids = np.array([[teammates[i % len(teammates)]] * 2 for i in range(n_ep)])
    episodes = [
        Episode(
            obs=rng.normal(size=(2, T, obs_dim)).astype(np.float32),
            actions=rng.integers(0, 6, size=(2, T)),
            rewards=rng.normal(size=(2, T)).astype(np.float32),
            avail_actions=np.ones((2, T, 6), dtype=np.float32),
            dones=np.zeros(T, dtype=bool),
        )
        for _ in range(n_ep)
    ]
    return write_vault(
        episodes, member_ids, tmp_path / "dataset.vlt", ego_index=0, meta={"generator": "test"}
    )


def _job(tmp_path: Path, baseline: str, **overrides) -> TrainingJob:
    # network.architecture selects the model; baseline drives runner dispatch. For
    # a baseline with no network config yet (e.g. the unimplemented "taget"), fall
    # back to a valid architecture so the *runner's* SUPPORTED gate is what refuses,
    # not config validation.
    architecture = baseline if baseline in ("liam", "meliba", "omis", "tao", "bc") else "liam"
    offline = OfflineTrainingConfig(
        network={"architecture": architecture},
        context_length=6,
        stride=3,
        stage1_steps=2,
        stage2_steps=2,
        log_every=1,
        batch_size=8,
        context_trajectories=2,
    )
    return TrainingJob(
        label=f"t_{baseline}",
        env=get_preset("lbf_12x12"),
        dataset_path=str(_dataset(tmp_path)),
        baseline=baseline,
        offline=offline,
        output_dir=str(tmp_path / "out"),
        num_seeds=overrides.pop("num_seeds", 1),
        **overrides,
    )


@pytest.mark.parametrize("baseline", ["liam", "meliba", "tao"])
def test_runner_trains_and_writes_parameters(tmp_path, baseline):
    """Both baselines complete and leave a loadable artifact plus metrics.

    OMIS is excluded from this parametrization for the same reason BC is
    (see ``test_bc_stage_one_trains_nothing`` / ``test_omis_stage_one_trains_nothing``
    below): both have a no-op ``train_stage_1`` and so never log a ``Stage1/``
    tag, which this test's generic assertion requires.
    """
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


def test_bc_stage_one_trains_nothing(tmp_path):
    """%BC's whole point is having no representation stage -- pinned directly
    rather than left as an inference from the loss tests.

    Stage 2 still has to look like every other baseline's: an accuracy metric,
    a loadable artifact.
    """
    from oaht_bench.offline.runner import run

    run_dir = run(_job(tmp_path, "bc"))
    assert (run_dir / "params.pkl").exists()

    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    tags = {k for r in records for k in r if "/" in k}
    assert not any(t.startswith("Stage1/") for t in tags)
    assert "Stage2/action_accuracy" in tags


def test_omis_stage_one_trains_nothing(tmp_path):
    """OMIS's ``train_stage_1`` is a no-op (see its own docstring and
    ``docs/tuning_record.md``'s record of the earlier, incorrect staged
    version) -- pinned directly, the same way BC's is above.

    Unlike BC, Stage 2 still has an imitator head, so both
    ``action_accuracy`` (the actor) and ``imitator_accuracy`` (``mu_phi``)
    must show up there.
    """
    from oaht_bench.offline.runner import run

    run_dir = run(_job(tmp_path, "omis"))
    assert (run_dir / "params.pkl").exists()

    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    tags = {k for r in records for k in r if "/" in k}
    assert not any(t.startswith("Stage1/") for t in tags)
    assert "Stage2/action_accuracy" in tags
    assert "Stage2/imitator_accuracy" in tags


def test_bc_filters_to_the_top_quantile_of_episodes_by_return(tmp_path):
    """The '%' in %BC: only episodes above the return threshold contribute
    windows, and filtering is by episode, not by window, so a long episode's
    overlapping fragments cannot dominate the quantile.
    """
    from oaht_bench.offline.bc import _filter_by_return

    n_ep, T, obs_dim = 4, 6, 3
    episodes = []
    for ep in range(n_ep):
        r = np.zeros((2, T), dtype=np.float32)
        r[0] = float(ep) / T  # episode ep's ego return sums to exactly ep
        episodes.append(
            Episode(
                obs=np.zeros((2, T, obs_dim), dtype=np.float32),
                actions=np.zeros((2, T), dtype=np.int32),
                rewards=r,
                avail_actions=np.ones((2, T, 4), dtype=np.float32),
                dones=np.zeros(T, dtype=bool),
            )
        )
    batch = EpisodeBatch(
        episodes=episodes,
        member_ids=np.zeros((n_ep, 2), dtype=np.int32),
        ego_index=0,
        meta={"generator": "test"},
    )

    kept_episodes = set(_filter_by_return(batch, ego=0, top_return_quantile=0.5).tolist())
    assert kept_episodes == {2, 3}, "top 50% of 4 episodes by return should be episodes 2 and 3"

    assert len(_filter_by_return(batch, ego=0, top_return_quantile=1.0)) == n_ep


def test_runner_refuses_an_unimplemented_baseline(tmp_path):
    """The roster in BaselineName is the plan, not what exists.

    Silently training something else would be worse than failing.
    """
    from oaht_bench.offline.runner import run

    with pytest.raises(NotImplementedError, match="has no runner yet"):
        run(_job(tmp_path, "taget"))


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


@pytest.mark.parametrize("baseline", ["liam", "meliba", "tao"])
def test_runner_logs_accuracies_and_evaluation_returns(tmp_path, baseline):
    """A falling loss is not evidence the policy plays.

    Three metric families have to appear: action-prediction accuracy on the ego
    (comparable across baselines in a way nats are not), teammate-action
    accuracy from the representation stage (whether the auxiliary task worked),
    and episode returns per teammate from real rollouts.

    OMIS is excluded here too (see ``test_omis_logs_accuracies_and_evaluation_returns``
    below) -- it has no representation stage to report a ``Stage1/`` accuracy from.
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


def test_omis_logs_accuracies_and_evaluation_returns(tmp_path):
    """OMIS's analog of ``test_runner_logs_accuracies_and_evaluation_returns``:
    the teammate-action accuracy (``mu_phi``) lands under ``Stage2/`` instead
    of ``Stage1/``, since there is no separate representation stage to report
    it from -- everything is one pass (see ``test_omis_stage_one_trains_nothing``).
    """
    from oaht_bench.offline.runner import run

    job = _job(tmp_path, "omis")
    job = job.model_copy(update={"offline": job.offline.model_copy(update={"eval_episodes": 1})})
    run_dir = run(job)

    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    tags = {k for r in records for k in r if "/" in k}

    assert "Stage2/action_accuracy" in tags
    assert "Stage2/imitator_accuracy" in tags
    assert not any(t.startswith("Stage1/") for t in tags)

    summary = json.loads((run_dir / "training_summary.json").read_text())
    assert summary["eval"] is None
    assert "population_run" in summary["eval_skipped"]


def test_evaluation_target_return_comes_from_the_dataset(tmp_path):
    """dataset_target_return is the fallback path (no pooled crossplay matrix):
    resolve_target_returns prefers a per-teammate value read off the matrix
    when one exists (offline/evaluate.py), mirroring the reference's own
    per-opponent target table.

    Conditioning on the dataset's best episode return is the Decision
    Transformer convention -- ask for the best behaviour the data contains.
    """
    from oaht_bench.offline.evaluate import dataset_target_return

    batch = read_vault(_dataset(tmp_path))
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
