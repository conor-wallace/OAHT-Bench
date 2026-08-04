"""Execute a :class:`~oaht_bench.configs.job.TeammateGenerationJob`.

Dispatches to the absorbed generator implementations. They take a plain nested
dict — jax-aht converted away from Hydra's ``DictConfig`` at its entry point and
we kept that boundary — so the job config projects straight onto them without
Hydra, YAML, or a subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from oaht_bench.common.logging import RunLogger
from oaht_bench.configs import save_job
from oaht_bench.configs.job import TeammateGenerationJob


def _generators() -> dict[str, Callable[..., Any]]:
    """Import the generators lazily so the CLI can validate without loading JAX."""
    from oaht_bench.teammate_gen.BRDiv import run_brdiv
    from oaht_bench.teammate_gen.CoMeDi import run_comedi
    from oaht_bench.teammate_gen.fcp import run_fcp
    from oaht_bench.teammate_gen.LBRDiv import run_lbrdiv

    return {"fcp": run_fcp, "comedi": run_comedi, "brdiv": run_brdiv, "lbrdiv": run_lbrdiv}


def run(job: TeammateGenerationJob) -> Path:
    """Train a teammate population and return the run directory.

    The config's content hash names the directory and the config is written into
    it, so a population can always be traced to the settings that produced it —
    the provenance §7.1 requires for released checkpoints.
    """
    run_dir = Path(job.run_dir())
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = job.to_jax_aht_cfg()
    alg = job.generator.generator

    runners = _generators()
    if alg not in runners:
        raise ValueError(f"No runner for generator {alg!r}. Known: {sorted(runners)}")

    save_job(job, run_dir / "job.json")
    (run_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True, default=str) + "\n"
    )

    with RunLogger(
        run_dir,
        use_wandb=job.logging.use_wandb,
        wandb_project=job.logging.wandb_project,
        wandb_entity=job.logging.wandb_entity,
        config=cfg,
        verbose=job.logging.verbose,
    ) as logger:
        # All four generators read the typed job directly.
        runners[alg](job, logger)

    return run_dir
