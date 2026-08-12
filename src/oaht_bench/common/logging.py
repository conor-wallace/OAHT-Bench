"""Run logging, with Weights & Biases off by default.

The absorbed :class:`oaht_bench.common.wandb_visualizations.Logger` calls
``wandb.init`` unconditionally and reads ``config["logger"]["entity"]`` without a
default, so it fails on a machine with no wandb credentials and, worse, will
happily publish to whatever entity a config happens to carry. A benchmark should
run identically for someone who has never heard of wandb.

:class:`RunLogger` therefore defaults to local-only. Metrics still go somewhere —
a JSONL file under the run directory — so a disabled run is *quiet*, not
*silent*: the numbers survive for later analysis without depending on a hosted
service. Enabling wandb is an explicit opt-in through the job config.
"""

from __future__ import annotations

import json
import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@contextmanager
def nonfatal(what: str):
    """Run reporting code that must not be able to destroy a finished run.

    Post-training logging is the last thing a generator does, after hours of
    training, and it is the least important thing it does. A charting call that
    rejects a matrix width, or a NaN assertion in a loss curve, should surface as
    an error to read — not as a lost checkpoint.

    Callers must have already written their artifacts before entering this.
    """
    try:
        yield
    except Exception:
        log.exception(
            "%s failed. Training finished and the checkpoint is already written, "
            "so the run is intact and can be re-scored with `sweep.py rescore`; "
            "only this reporting step was lost.",
            what,
        )


def _is_wandb_object(value: Any) -> bool:
    """Whether a value only has meaning inside Weights & Biases.

    The training code builds charts and tables with ``wandb.plot.*`` and passes
    them through the same ``log_item`` used for scalars. They are presentation
    objects with no local equivalent, so with wandb disabled they are dropped
    rather than stringified into the metrics stream.
    """
    return type(value).__module__.split(".")[0] == "wandb"


def _to_jsonable(value: Any) -> Any:
    """Coerce a logged value into something JSON can represent.

    Never raises. A metric sink that can crash is a metric sink that takes a
    multi-hour training run down with it — which is exactly what happened when
    CoMeDi logged a ``wandb.plot.line_series`` chart through this path.
    """
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    try:
        if hasattr(value, "item") and getattr(value, "size", None) == 1:
            return value.item()
        arr = np.asarray(value)
        if arr.dtype == object:
            raise TypeError("object array")
        return arr.item() if arr.ndim == 0 else arr.tolist()
    except Exception:
        return f"<unserializable {type(value).__name__}>"


class RunLogger:
    """Metric sink matching the interface the absorbed training code expects.

    Implements ``log_item``, ``commit``, ``log_xp_matrix``, ``log_artifact`` and
    ``close`` — the five methods the teammate-generation algorithms actually call.

    Args:
        run_dir: Directory for this run's outputs. Created if absent.
        use_wandb: Opt in to Weights & Biases. Off by default.
        wandb_project / wandb_entity: Only consulted when ``use_wandb`` is set.
        config: Recorded verbatim as the run's provenance.
        verbose: Echo each logged item to stdout.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        config: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._metrics_file = self._metrics_path.open("a", encoding="utf-8")
        self._pending: dict[str, Any] = {}
        self._step = 0
        self.run = None

        if config is not None:
            (self.run_dir / "config.json").write_text(
                json.dumps(config, indent=2, sort_keys=True, default=str) + "\n"
            )

        if use_wandb:
            import wandb  # imported lazily; not needed for the default path

            if wandb_project is None:
                raise ValueError("use_wandb is set but wandb_project is None.")
            self.run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config=config,
                reinit=True,
            )
            wandb.define_metric("train_step")
            wandb.define_metric("checkpoint")
            for prefix in ("Train", "Losses", "Eval", "Returns"):
                wandb.define_metric(f"{prefix}/*", step_metric="train_step")

    # --- interface used by the absorbed training code ---------------------

    def log(self, data: dict[str, Any], step: int | None = None, commit: bool = False) -> None:
        # wandb charts have no local representation; keep them out of the JSONL.
        local = {k: v for k, v in data.items() if not _is_wandb_object(v)}
        self._pending.update(local)
        if self.run is not None:
            import wandb

            wandb.log(data, step=step, commit=commit)
        if commit:
            self.commit(step=step)

    def log_item(
        self, tag: str, val: Any, step: int | None = None, commit: bool = True, **kwargs
    ) -> None:
        self.log({tag: val, **kwargs}, step=step, commit=commit)
        if self.verbose:
            print(f"{tag}: {val}")

    def commit(self, step: int | None = None) -> None:
        """Flush pending metrics as one JSONL record.

        Swallows serialization failures. Logging is not worth losing a training
        run over, and ``close`` calls this from ``__exit__`` where raising would
        mask whatever actually went wrong.
        """
        if not self._pending:
            return
        record = {"step": self._step if step is None else step}
        record.update({k: _to_jsonable(v) for k, v in self._pending.items()})
        try:
            self._metrics_file.write(json.dumps(record) + "\n")
            self._metrics_file.flush()
        except (TypeError, ValueError, OSError) as e:  # pragma: no cover - defensive
            log.warning("dropping a metrics record that could not be written: %s", e)
        finally:
            self._pending.clear()
            self._step += 1
        if self.run is not None:
            import wandb

            wandb.log({}, commit=True)

    def log_xp_matrix(
        self,
        tag: str,
        mat,
        step: int | None = None,
        columns: list[str] | None = None,
        rows: list[str] | None = None,
        commit: bool = True,
        **kwargs,
    ) -> None:
        """Record a cross-play matrix as CSV alongside the metrics stream.

        Cross-play matrices are a headline diagnostic (§8), so they are written as
        a file rather than only into a metrics record — they need to be readable
        without re-running anything.
        """
        arr = np.asarray(mat)
        safe_tag = tag.replace("/", "_")
        path = self.run_dir / f"{safe_tag}.csv"
        header = ",".join(columns) if columns else ""
        np.savetxt(path, arr, delimiter=",", header=header, comments="")
        if self.run is None:
            return

        import wandb

        # wandb.Table defaults columns to ["Input", "Output", "Expected"] when
        # none are given, so a matrix of any width other than 3 is rejected. That
        # made this silently correct while populations were size 3 and a hard
        # failure the moment they were not. Derive them from the matrix instead.
        n_cols = arr.shape[1] if arr.ndim > 1 else arr.shape[0]
        cols = list(columns) if columns else [f"col {i}" for i in range(n_cols)]
        data = arr.tolist()
        if rows:
            # Row labels have to be a column; wandb.Table has no row index.
            cols = ["row", *cols]
            # strict: caller-supplied labels and matrix rows can genuinely
            # disagree, and silently truncating would mislabel the matrix.
            data = [[label, *row] for label, row in zip(rows, data, strict=True)]
        wandb.log({tag: wandb.Table(columns=cols, data=data)}, step=step, commit=commit, **kwargs)

    def log_artifact(self, name: str, path: str | Path, type_name: str) -> None:
        """Record an artifact. Locally this copies it under the run directory."""
        src = Path(path)
        if not src.exists():
            return
        # Artifacts written by the training code already land inside run_dir;
        # copying them again would triple-store multi-GB checkpoints.
        if self.run_dir.resolve() in src.resolve().parents:
            if self.run is None:
                return
        dst = self.run_dir / "artifacts" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if self.run_dir.resolve() not in src.resolve().parents:
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        if self.run is not None:
            import wandb

            artifact = wandb.Artifact(name, type=type_name)
            if src.is_dir():
                artifact.add_dir(str(src))
            else:
                artifact.add_file(str(src))
            self.run.log_artifact(artifact)

    def log_video(self, tag: str, path: str | Path, commit: bool = True) -> None:
        if self.run is not None:
            import wandb

            wandb.log({tag: wandb.Video(str(path))}, commit=commit)

    def close(self) -> None:
        self.commit()
        self._metrics_file.close()
        if self.run is not None:
            import wandb

            wandb.finish()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


#: Episode statistics logged for every generator, on top of whatever
#: ``get_metric_names`` reports for the environment. ``returned_episode_lengths``
#: is included because FCP and CoMeDi already emit it -- ``ippo``'s in-training
#: callback logs every statistic rather than only the named ones -- and a metric
#: present for two of four methods is not usable for comparison.
_ALWAYS_LOGGED = ("returned_episode_returns", "returned_episode_lengths")


def log_training_curves(
    logger: RunLogger,
    metrics: dict[str, Any],
    env_name: str,
    *,
    prefix: str = "Train",
) -> None:
    """Log per-update episode statistics under the same tags across generators.

    FCP and CoMeDi emit ``Train/<stat>`` from inside the PPO loop, through
    ``ippo``'s ``io_callback``. BRDiv and L-BRDiv have their own training loops
    and never call it, so they collected the same statistics but never logged
    them. This emits them post-hoc from the returned metrics so convergence is
    comparable across all four methods.

    Args:
        metrics: The training output's ``metrics`` dict. Episode statistics are
            expected with shape ``(num_seeds, num_updates)``.
        env_name: Selects the environment-specific statistics to include.

    Note:
        The tag is the same but the *measurement* is not identical across
        methods: BRDiv and L-BRDiv compute these over confederate trajectories,
        whereas FCP measures partner self-play. They are comparable as
        convergence signals -- is training progressing, has it plateaued -- and
        not as a like-for-like performance comparison between generators.
    """
    from oaht_bench.common.plot_utils import get_metric_names

    wanted = tuple(dict.fromkeys(tuple(get_metric_names(env_name)) + _ALWAYS_LOGGED))
    available = [k for k in wanted if k in metrics]
    if not available:
        return

    curves = {}
    for name in available:
        arr = np.asarray(metrics[name])
        if arr.ndim < 2:
            continue
        # Average over every axis but the update axis (axis 1).
        axes = tuple(i for i in range(arr.ndim) if i != 1)
        curves[name] = arr.mean(axis=axes)

    if not curves:
        return
    num_updates = len(next(iter(curves.values())))
    for step in range(num_updates):
        for name, series in curves.items():
            logger.log_item(f"{prefix}/{name}", series[step], train_step=step)
    logger.commit()


def log_update_metrics(
    metrics: dict[str, Any], logger: RunLogger, *, prefix: str = "Train"
) -> None:
    """Log one update step's statistics, from inside a jitted training loop.

    Called through ``jax.experimental.io_callback``, so it must stay a
    module-level function and must tolerate whatever the tracer hands it.

    Exists because BRDiv and L-BRDiv run their whole training inside a single
    ``jit(vmap(...))``: without a callback nothing escapes until the call
    returns, so a multi-hour run showed no progress at all until it finished.

    Non-scalar entries are skipped. The loss terms these generators record carry
    a population axis, and a partially-reduced array is not a meaningful scalar
    to plot against an update step; the per-pair losses are logged post-hoc where
    that axis can be handled properly.
    """
    stats = dict(metrics)
    step_val = stats.pop("update_steps", None)
    if step_val is None:
        return
    step = int(np.asarray(step_val).reshape(-1)[0])

    for name, value in stats.items():
        if name == "returned_episode":
            continue
        arr = np.asarray(value)
        if arr.size != 1:
            continue
        logger.log_item(f"{prefix}/{name}", float(arr.reshape(-1)[0]), train_step=step)
    logger.commit()
