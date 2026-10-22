"""The ad-hoc-teamwork train/test teammate split (§8).

Pins the one invariant the whole eval protocol rests on: a held-out (test) teammate
is never in the train partition, in either seat -- and, for paired generators, that
holding out a member removes *both* its roles.
"""

import numpy as np
import pytest

from oaht_bench.dataset.construction.split import derive_split, roster_fingerprint


def _pooled_roster():
    """A roster shaped like the real pooled one: paired generators contribute a
    conf and a br per member; homogeneous generators one self per member. 5 members
    each; FCP members are the converged-checkpoint indices released_members emits."""
    roster = []
    for m in range(5):
        roster.append(("brdiv", m, "conf"))
        roster.append(("brdiv", m, "br"))
        roster.append(("lbrdiv", m, "conf"))
        roster.append(("lbrdiv", m, "br"))
        roster.append(("comedi", m, "self"))
        roster.append(("fcp", m * 5 + 4, "self"))  # converged checkpoint per run
    return roster


def test_split_is_disjoint_and_covers_the_roster():
    roster = _pooled_roster()
    s = derive_split(roster, env="lbf_12x12", split_seed=0, holdout_per_generator=2)
    train, test = set(s.train_indices), set(s.test_indices)
    assert train.isdisjoint(test)
    assert train | test == set(range(len(roster)))
    assert test, "a nonzero holdout must produce a nonempty test set"


def test_holdout_count_per_generator():
    roster = _pooled_roster()
    s = derive_split(roster, env="lbf_12x12", split_seed=0, holdout_per_generator=2)
    for gen, held in s.held_out.items():
        assert len(held) == 2, f"{gen} held out {held}, expected 2"
    # every generator represented in test
    test_gens = {roster[i][0] for i in s.test_indices}
    assert test_gens == {"brdiv", "lbrdiv", "comedi", "fcp"}


def test_paired_member_holds_out_both_roles():
    """The 'whole member, both seats' decision: if a BRDiv member is test, both its
    conf and its br roster entries are test -- neither role leaks into train."""
    roster = _pooled_roster()
    s = derive_split(roster, env="lbf_12x12", split_seed=1, holdout_per_generator=2)
    test = set(s.test_indices)
    for gen in ("brdiv", "lbrdiv"):
        for m in s.held_out[gen]:
            idx = [i for i, (g, mm, _r) in enumerate(roster) if g == gen and mm == m]
            assert len(idx) == 2, "a paired member should have conf+br entries"
            assert all(i in test for i in idx), f"{gen} member {m} leaks a role into train"


def test_deterministic_in_seed():
    roster = _pooled_roster()
    a = derive_split(roster, env="lbf_12x12", split_seed=7, holdout_per_generator=2)
    b = derive_split(roster, env="lbf_12x12", split_seed=7, holdout_per_generator=2)
    assert a.test_indices == b.test_indices
    assert a.manifest_hash == b.manifest_hash
    # a different seed generally moves the held-out set (not guaranteed for every
    # seed pair, but must for at least one) and changes the manifest hash.
    c = derive_split(roster, env="lbf_12x12", split_seed=8, holdout_per_generator=2)
    assert c.manifest_hash != a.manifest_hash or c.held_out != a.held_out


def test_holdout_cannot_exceed_population():
    roster = _pooled_roster()
    with pytest.raises(ValueError, match="exceeds"):
        derive_split(roster, env="lbf_12x12", split_seed=0, holdout_per_generator=6)


def test_zero_holdout_keeps_everything():
    roster = _pooled_roster()
    s = derive_split(roster, env="lbf_12x12", split_seed=0, holdout_per_generator=0)
    assert s.test_indices == ()
    assert set(s.train_indices) == set(range(len(roster)))


def test_fingerprint_detects_roster_drift():
    a = roster_fingerprint(_pooled_roster())
    drifted = _pooled_roster() + [("rpg", 0, "self")]  # a re-released / extended pop
    assert roster_fingerprint(drifted) != a


def test_manifest_round_trips():
    from oaht_bench.dataset.construction.split import TeammateSplit

    roster = _pooled_roster()
    s = derive_split(roster, env="lbf_12x12", split_seed=3, holdout_per_generator=2)
    back = TeammateSplit.from_dict(s.to_dict())
    assert back == s
