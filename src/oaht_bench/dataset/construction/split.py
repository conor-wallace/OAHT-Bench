"""Train/test teammate split for ad-hoc teamwork (§8, held-out partners).

The benchmark's real objective is generalisation to *unseen* teammates: an offline
learner is trained on rollouts collected against **train** teammates and evaluated
online against **test** teammates it never saw during data collection. This module
derives that split and enforces its one hard invariant -- a test teammate is never
used in train collection, in either seat.

The unit of the split is a released population member, addressed by
``(generator, member)`` -- the same identity the pooled roster and cross-play matrix
use. ``released_members`` has already collapsed FCP to one converged checkpoint per
run, so there is no checkpoint-sibling leakage to reason about: every generator
contributes ``population_size`` distinct members, and ``holdout_per_generator`` of
each generator's members go to test. A paired generator's ``conf`` and ``br`` share a
``member``, so holding out a member removes *both* roles -- neither the confederate
nor its best response leaks.

The split is a deterministic function of ``(roster identities, split_seed,
holdout_per_generator)``, so every dataset type (expert / mixed / br_vs_worst)
collected with the same parameters gets the *same* held-out set -- one canonical
split per environment, by construction rather than by a shared file. The manifest a
collection writes records the roster fingerprint it was derived against, so a
re-released population that would silently change the split is caught.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

#: One roster entry's identity: ``(generator, member, role)``, index-aligned with a
#: :func:`~oaht_bench.population.pooled_crossplay.build_roster` roster.
Identity = tuple[str, int, str]


def roster_fingerprint(identities: list[Identity]) -> str:
    """Content hash of the roster, so a manifest is tied to one population release."""
    payload = json.dumps([[g, int(m), r] for g, m, r in identities], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class TeammateSplit:
    """A canonical train/test partition of a pooled roster.

    ``held_out`` maps each generator to the member ids assigned to test;
    ``train_indices`` / ``test_indices`` resolve those to roster positions (both
    roles of a held-out paired member land in ``test_indices``). ``manifest_hash``
    is a stable id for the split, derived only from the fields that define it.
    """

    env: str
    split_seed: int
    holdout_per_generator: int
    roster_fingerprint: str
    held_out: dict[str, list[int]]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

    @property
    def manifest_hash(self) -> str:
        payload = json.dumps(
            {
                "env": self.env,
                "split_seed": self.split_seed,
                "holdout_per_generator": self.holdout_per_generator,
                "roster_fingerprint": self.roster_fingerprint,
                "held_out": {g: sorted(ms) for g, ms in self.held_out.items()},
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "env": self.env,
            "split_seed": self.split_seed,
            "holdout_per_generator": self.holdout_per_generator,
            "roster_fingerprint": self.roster_fingerprint,
            "manifest_hash": self.manifest_hash,
            "held_out": {g: sorted(ms) for g, ms in self.held_out.items()},
            "train_indices": list(self.train_indices),
            "test_indices": list(self.test_indices),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TeammateSplit:
        return cls(
            env=d["env"],
            split_seed=int(d["split_seed"]),
            holdout_per_generator=int(d["holdout_per_generator"]),
            roster_fingerprint=d["roster_fingerprint"],
            held_out={g: [int(m) for m in ms] for g, ms in d["held_out"].items()},
            train_indices=tuple(int(i) for i in d["train_indices"]),
            test_indices=tuple(int(i) for i in d["test_indices"]),
        )


def derive_split(
    identities: list[Identity],
    *,
    env: str,
    split_seed: int,
    holdout_per_generator: int,
) -> TeammateSplit:
    """Partition a roster into train/test by holding out members per generator.

    ``identities`` is index-aligned with the roster (position ``i`` is roster entry
    ``i``). Generators are processed in sorted order under a single seeded RNG, so the
    split is fully determined by ``(identities, split_seed, holdout_per_generator)``.
    """
    members_by_gen: dict[str, list[int]] = OrderedDict()
    for gen, member, _role in identities:
        members_by_gen.setdefault(gen, [])
        if member not in members_by_gen[gen]:
            members_by_gen[gen].append(member)

    rng = np.random.default_rng(split_seed)
    held_out: dict[str, list[int]] = {}
    for gen in sorted(members_by_gen):
        members = sorted(members_by_gen[gen])
        if holdout_per_generator > len(members):
            raise ValueError(
                f"holdout_per_generator={holdout_per_generator} exceeds the "
                f"{len(members)} released members of generator {gen!r}; the whole "
                f"population would be held out, leaving nothing to train on."
            )
        perm = rng.permutation(len(members))
        held_out[gen] = sorted(members[i] for i in perm[:holdout_per_generator])

    held_set = {(g, m) for g, ms in held_out.items() for m in ms}
    test_indices = tuple(
        i for i, (g, m, _r) in enumerate(identities) if (g, m) in held_set
    )
    train_indices = tuple(
        i for i, (g, m, _r) in enumerate(identities) if (g, m) not in held_set
    )
    return TeammateSplit(
        env=env,
        split_seed=split_seed,
        holdout_per_generator=holdout_per_generator,
        roster_fingerprint=roster_fingerprint(identities),
        held_out=held_out,
        train_indices=train_indices,
        test_indices=test_indices,
    )
