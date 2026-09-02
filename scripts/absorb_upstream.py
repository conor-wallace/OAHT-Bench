"""Copy upstream subtrees into ``src/oaht_bench/`` and rewrite their imports.

OAHT-Bench cannot depend on both jax-aht and ICRL4AHT as packages: they declare
the same top-level names (``agents``, ``common``, ``envs``, ``marl``,
``teammate_generation``), so installing both makes ``from envs import ...``
resolve to whichever landed second. ICRL4AHT hit the same wall and solved it the
same way — it ships a diverged copy of jax-aht's core with no dependency on it.

So upstream code is *absorbed* into our tree and owned, not vendored behind a
``third_party`` boundary. We modify nearly all of it (env-generic encoders,
offline conversions, the shared DT backbone), which makes a "sync from upstream"
contract dishonest.

This script exists so the absorption is reproducible and auditable rather than a
one-time manual copy:

  * it records the upstream commit for every absorbed subtree in PROVENANCE.md
  * it applies only mechanical import rewrites, so the first absorption commit is
    a pure transform and every later diff is genuinely *our* change
  * re-running it against a newer upstream shows what drifted

Usage::

    uv run python scripts/absorb_upstream.py --source ../jax-aht --plan
    uv run python scripts/absorb_upstream.py --source ../jax-aht --apply
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "src" / "oaht_bench"


@dataclass(frozen=True)
class Subtree:
    """One upstream directory (or selection of files) and where it lands."""

    upstream: str
    local: str
    note: str
    #: When set, absorb only these paths (relative to ``upstream``) rather than
    #: the whole directory. Used where a subtree mixes code we want with code we
    #: are replacing -- e.g. jax-aht's online ego trainers.
    only: tuple[str, ...] | None = None


#: jax-aht subtrees we take, and the module names they become. ``teammate_generation``
#: is renamed to ``teammate_gen`` to match our layout; the rewrite table below keeps
#: intra-package imports consistent with the rename.
JAX_AHT_SUBTREES = (
    Subtree("envs", "envs", "LBF, Overcooked-v1 and Hanabi wrappers over Jumanji/JaxMARL."),
    Subtree(
        "agents",
        "agents",
        "Policy architectures, population interfaces, scripted teammates (§7.6).",
    ),
    Subtree("teammate_generation", "teammate_gen", "FCP, CoMeDi, BRDiv, L-BRDiv (§7)."),
    Subtree(
        "marl",
        "teammate_gen/marl",
        "IPPO and PPO utilities; teammate generation is the only consumer.",
    ),
    Subtree("common", "common", "Rollout helpers, checkpoint save/load, plotting."),
)

#: Deliberately NOT absorbed, with the reason, so the omissions are auditable:
#:
#: evaluation/            -- we are writing our own protocol (§8); jax-aht's
#:                           held-out evaluator does not implement graded shift,
#:                           adaptation gain, or IQM aggregation.
#: ego_agent_training/    -- online PPO ego training, superseded by §3.1. Note
#:                           ppo_br.py will be needed later for per-teammate best
#:                           responses (§4.3 `expert`), which OMIS, TAO and
#:                           BR-Prox all depend on.
#: open_ended_training/   -- out of scope.

SOURCES = {"jax-aht": JAX_AHT_SUBTREES}


#: Upstream top-level module -> our dotted path. Order matters only in that longer
#: names must not be prefixes of shorter ones; these are all distinct.
def _rewrite_table(subtrees: tuple[Subtree, ...]) -> dict[str, str]:
    return {s.upstream: "oaht_bench." + s.local.replace("/", ".") for s in subtrees}


def rewrite_imports(text: str, table: dict[str, str]) -> tuple[str, int]:
    """Rewrite absolute imports of absorbed top-level modules.

    Handles ``from X import ...``, ``from X.y import ...``, ``import X`` and
    ``import X.y``. Deliberately conservative: it only touches statements at the
    start of a line (allowing indentation), so strings and comments mentioning a
    module name are left alone.
    """
    total = 0
    for old, new in table.items():
        patterns = (
            (rf"(?m)^(\s*)from {re.escape(old)}(\.|\s)", rf"\1from {new}\2"),
            (rf"(?m)^(\s*)import {re.escape(old)}(\.|\s|$)", rf"\1import {new}\2"),
        )
        for pat, repl in patterns:
            text, n = re.subn(pat, repl, text)
            total += n
    return text, total


def upstream_commit(source: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown (not a git checkout)"


@contextmanager
def clean_checkout(source: Path):
    """Yield a pristine export of the source's HEAD, ignoring local edits.

    Absorbing from a working tree would silently bake in whatever the developer
    happened to have uncommitted, and the recorded commit would then describe code
    we did not actually copy. Exporting HEAD keeps provenance exact.
    """
    is_git = (source / ".git").exists()
    if not is_git:
        yield source
        return
    with tempfile.TemporaryDirectory(prefix="absorb-") as tmp:
        tmpdir = Path(tmp)
        archive = subprocess.run(
            ["git", "-C", str(source), "archive", "HEAD"],
            capture_output=True,
            check=True,
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(tmpdir)], input=archive, check=True)
        yield tmpdir


def absorb(source: Path, subtrees: tuple[Subtree, ...], *, apply: bool) -> list[str]:
    table = _rewrite_table(subtrees)
    lines: list[str] = []
    for sub in subtrees:
        src_dir = source / sub.upstream
        dst_dir = PKG_ROOT / sub.local
        if not src_dir.is_dir():
            lines.append(f"  MISSING  {src_dir}")
            continue
        if sub.only is not None:
            py_files = [src_dir / f for f in sub.only if (src_dir / f).is_file()]
        else:
            py_files = [p for p in src_dir.rglob("*.py") if "__pycache__" not in p.parts]
        other = (
            []
            if sub.only is not None
            else [
                p
                for p in src_dir.rglob("*")
                if p.is_file()
                and p.suffix in {".yaml", ".yml", ".json", ".npy", ".safetensors", ".sh"}
                and "__pycache__" not in p.parts
            ]
        )
        rewrites = 0
        if apply:
            if sub.only is None and dst_dir.exists():
                shutil.rmtree(dst_dir)
            for p in py_files + other:
                rel = p.relative_to(src_dir)
                out = dst_dir / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                if p.suffix == ".py":
                    text = p.read_text(encoding="utf-8")
                    text, n = rewrite_imports(text, table)
                    rewrites += n
                    out.write_text(text, encoding="utf-8")
                else:
                    shutil.copy2(p, out)
        rename = "" if sub.upstream == sub.local else f"  (renamed from {sub.upstream})"
        lines.append(
            f"  {sub.upstream:22s} -> oaht_bench/{sub.local:16s} "
            f"{len(py_files):3d} py + {len(other):3d} data"
            + (f", {rewrites} imports rewritten" if apply else "")
            + rename
        )
    return lines


def write_provenance(source: Path, name: str, subtrees: tuple[Subtree, ...]) -> None:
    sha = upstream_commit(source)
    table = _rewrite_table(subtrees)
    path = REPO_ROOT / "PROVENANCE.md"
    body = [
        "# Provenance of absorbed upstream code",
        "",
        "Parts of `src/oaht_bench/` originate in other projects and were absorbed",
        "rather than depended on, because the upstreams claim colliding top-level",
        "package names (see `scripts/absorb_upstream.py` for the reasoning).",
        "",
        "Absorbed code is **owned and modified** here. To see our changes relative to",
        "upstream, diff against the recorded commit. Regenerate this file by re-running",
        "the absorption script.",
        "",
        f"## {name}",
        "",
        "- Upstream: `https://github.com/LARG/jax-aht`",
        f"- Commit: `{sha}`",
        "- License: MIT (see `LICENSES/jax-aht-LICENSE`)",
        "",
        "| upstream path | local path | contents |",
        "|---|---|---|",
    ]
    for s in subtrees:
        body.append(f"| `{s.upstream}/` | `src/oaht_bench/{s.local}/` | {s.note} |")
    body += [
        "",
        "Import rewrites applied:",
        "",
        "```",
        *[f"{old:22s} -> {new}" for old, new in table.items()],
        "```",
        "",
    ]
    path.write_text("\n".join(body))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="Path to the upstream checkout.")
    ap.add_argument("--name", default="jax-aht", choices=sorted(SOURCES))
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", action="store_true", help="Show what would happen.")
    group.add_argument("--apply", action="store_true", help="Perform the absorption.")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Absorb even though destinations already exist, discarding local edits.",
    )
    args = ap.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"error: no such source directory: {source}")

    subtrees = SOURCES[args.name]

    # Absorption is a one-time transform. Once absorbed, the code is ours and gets
    # modified -- re-running --apply would silently discard those modifications by
    # re-copying whole subtrees. Refuse unless explicitly forced.
    if args.apply and not args.force:
        occupied = [s.local for s in subtrees if s.only is None and (PKG_ROOT / s.local).exists()]
        if occupied:
            print(
                "refusing to absorb: these destinations already exist and would be\n"
                "overwritten, discarding any local modifications:\n"
                + "".join(f"  src/oaht_bench/{d}\n" for d in occupied)
                + "\nUse --plan to preview, or --force if you really mean to re-copy\n"
                "(commit first -- the overwrite is not recoverable from the worktree).",
                file=sys.stderr,
            )
            return 1
    print(f"{'Absorbing' if args.apply else 'Plan for'} {args.name} @ {source}")
    print(f"  upstream commit: {upstream_commit(source)}")
    with clean_checkout(source) as clean:
        if clean != source:
            print("  (exported HEAD; local uncommitted edits ignored)")
        for line in absorb(clean, subtrees, apply=args.apply):
            print(line)

    if args.apply:
        write_provenance(source, args.name, subtrees)
        print("  wrote PROVENANCE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
