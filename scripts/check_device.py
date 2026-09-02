"""Preflight: confirm JAX sees the accelerator, and that a config will fit on it.

Two failures this catches, both of which otherwise only surface hours in.

**Silent CPU fallback.** ``pip install jax`` with no CUDA plugin, a driver too old
for the wheel, or a ``platform_system`` marker that missed this OS all leave a
working JAX that quietly runs on CPU. Training still starts, prints reasonable
metrics, and takes days instead of hours. ``jax.default_backend()`` is the whole
tell, and nothing in a training run prints it.

**Out of memory at the first update.** PPO holds a full rollout before it does
anything with it, so peak memory is set by ``num_envs x rollout_length x obs``
times the population, and the paired generators hold two of those. On a 6 GB
card the Hanabi configs are the ones to worry about.

Usage::

    uv run python scripts/check_device.py
    uv run python scripts/check_device.py configs/lbf_12x12/teammate_gen/brdiv.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: XLA preallocates a fraction of the device at first use, so "free" memory
#: reported by nvidia-smi after JAX starts is not what is actually available.
PREALLOC_VARS = (
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
)


def report_devices() -> bool:
    """Print the backend and verify a real computation lands on it."""
    import jax
    import jax.numpy as jnp

    backend = jax.default_backend()
    devices = jax.devices()
    print(f"jax          {jax.__version__}")
    print(f"backend      {backend}")
    print(f"devices      {devices}")

    # default_backend() reports what JAX would choose; this proves an actual
    # buffer was placed there, which is what a training run depends on.
    x = jnp.ones((512, 512)) @ jnp.ones((512, 512))
    placed = list(x.devices())
    print(f"placement    {placed}   <- where a real array actually lives")

    on_accel = backend != "cpu"
    if not on_accel:
        print("\n  RUNNING ON CPU.")
        print("  If this machine has a GPU, JAX is not using it. Check, in order:")
        print("    python -c 'import jaxlib; print(jaxlib.__file__)'   # cuda plugin present?")
        print("    nvidia-smi                                          # driver visible?")
        print("    uv pip list | grep -i jax                           # jax-cuda12-* installed?")
        print("  pyproject only installs jax[cuda12] on Linux; on Windows both markers")
        print("  miss and no jax is installed at all. CUDA on Windows means WSL2.")
    return on_accel


def report_memory() -> float | None:
    """Print device memory, returning the usable bytes if known."""
    import jax

    dev = jax.devices()[0]
    stats = getattr(dev, "memory_stats", lambda: None)()
    if not stats:
        print("\nmemory       not reported by this backend")
        return None

    limit = stats.get("bytes_limit")
    in_use = stats.get("bytes_in_use", 0)
    gb = 1024**3
    print(f"\nmemory       {limit / gb:.1f} GiB limit, {in_use / gb:.2f} GiB in use")
    set_vars = {v: os.environ[v] for v in PREALLOC_VARS if v in os.environ}
    if set_vars:
        for k, v in set_vars.items():
            print(f"             {k}={v}")
    else:
        print("             (XLA preallocates ~75% of the device by default; set")
        print("              XLA_PYTHON_CLIENT_MEM_FRACTION to change it)")
    return float(limit) if limit else None


def estimate_rollout_bytes(config: Path) -> tuple[str, float]:
    """Rough peak rollout-buffer size for one teammate-generation config.

    Counts the dominant term only — the stored observations — as
    ``num_envs x agents x rollout_length x obs_size x 4 bytes``, times the
    population for the vmapped generators and twice that for the paired ones,
    which hold a confederate *and* a best-response trajectory. Activations,
    gradients and optimizer state are on top; treat this as a floor, not a
    budget.
    """
    from oaht_bench.configs import load_job
    from oaht_bench.envs import make_env

    job = load_job(config)
    env = make_env(job.env.env_name, job.env.env_kwargs())
    obs_space = env.observation_space(env.agents[0])
    obs_size = int(__import__("numpy").prod(obs_space.shape))

    gen = job.generator
    n = gen.population_size
    per_traj = gen.num_envs * len(env.agents) * job.env.rollout_length * obs_size * 4
    paired = gen.generator in ("brdiv", "lbrdiv")
    total = per_traj * n * (2 if paired else 1)

    print(f"\nconfig       {config}")
    print(f"  generator  {gen.generator}, population {n}, num_envs {gen.num_envs}")
    print(f"  obs        {obs_space.shape} = {obs_size} floats")
    print(
        f"  rollout    {per_traj / 1024**3:.2f} GiB per trajectory set"
        f"{' x2 (conf + br)' if paired else ''} x {n} members"
    )
    print(f"  estimate   {total / 1024**3:.2f} GiB of observations at peak")
    print("             (a floor: activations, grads and Adam state are extra)")
    return gen.generator, float(total)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="Teammate-generation config to size against the device.",
    )
    args = ap.parse_args()

    on_accel = report_devices()
    limit = report_memory()

    if args.config:
        _, needed = estimate_rollout_bytes(args.config)
        if limit:
            frac = needed / limit
            verdict = (
                "fits comfortably"
                if frac < 0.25
                else "should fit"
                if frac < 0.5
                else "TIGHT -- expect OOM once activations are added"
                if frac < 0.8
                else "WILL NOT FIT"
            )
            print(f"  vs device  {frac:.0%} of the {limit / 1024**3:.1f} GiB limit -> {verdict}")

    if not on_accel:
        return 1
    print("\nOK: JAX is on an accelerator.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
