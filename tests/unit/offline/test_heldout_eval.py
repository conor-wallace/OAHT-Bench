"""Held-out teammate selection for online evaluation (§8, step 2).

The eval must roll the trained ego against the *test* teammates a split held out --
across generators, teammate-role only (a br is a designed ego, never a partner) --
and never against a member that was in the training data.
"""


def _fake_roster():
    from oaht_bench.population.pooled_crossplay import RosterEntry

    return [
        RosterEntry("brdiv", 0, "conf", "p_b0c", "clsB"),  # train
        RosterEntry("brdiv", 0, "br", "p_b0b", "clsB"),  # train
        RosterEntry("brdiv", 2, "conf", "p_b2c", "clsB"),  # TEST teammate
        RosterEntry("brdiv", 2, "br", "p_b2b", "clsB"),  # TEST member but br -> not a teammate
        RosterEntry("comedi", 1, "self", "p_c1", "clsC"),  # TEST teammate
        RosterEntry("comedi", 3, "self", "p_c3", "clsC"),  # train
        RosterEntry("fcp", 14, "self", "p_f14", "clsF"),  # TEST teammate
        RosterEntry("fcp", 4, "self", "p_f4", "clsF"),  # train
    ]


class _Batch:
    def __init__(self, meta):
        self.meta = meta


def test_heldout_selection_picks_test_self_conf_across_generators(monkeypatch):
    import oaht_bench.population.pooled_crossplay as pc
    from oaht_bench.offline import runner

    roster = _fake_roster()
    monkeypatch.setattr(pc, "build_roster", lambda dirs, env: roster)

    batch = _Batch(
        {
            "held_out": {"brdiv": [2], "comedi": [1], "fcp": [14]},
            "populations": ["pops/brdiv", "pops/comedi", "pops/fcp"],
        }
    )
    teammates = runner._teammate_policies(batch, env=None)

    labels = sorted(t[0] for t in teammates)
    # held-out members only, teammate roles only -> brdiv:2 conf (not its br),
    # comedi:1 self, fcp:14 self. Nothing from train members, nothing with role br.
    assert labels == ["brdiv:2:conf", "comedi:1:self", "fcp:14:self"]
    # each tuple carries the teammate's own params + policy class (mixed generators)
    by_label = {t[0]: (t[1], t[2]) for t in teammates}
    assert by_label["brdiv:2:conf"] == ("p_b2c", "clsB")
    assert by_label["fcp:14:self"] == ("p_f14", "clsF")


def test_heldout_never_includes_a_train_member(monkeypatch):
    import oaht_bench.population.pooled_crossplay as pc
    from oaht_bench.offline import runner

    roster = _fake_roster()
    monkeypatch.setattr(pc, "build_roster", lambda dirs, env: roster)
    batch = _Batch(
        {"held_out": {"brdiv": [2], "comedi": [1], "fcp": [14]}, "populations": ["x"]}
    )
    members = {(lbl.split(":")[0], int(lbl.split(":")[1])) for lbl, *_ in runner._teammate_policies(batch, env=None)}
    # the train members (brdiv0, comedi3, fcp4) must never surface as eval teammates
    assert ("brdiv", 0) not in members
    assert ("comedi", 3) not in members
    assert ("fcp", 4) not in members


def test_single_population_falls_back_to_population_run(monkeypatch):
    """A split single-population dataset has no 'populations' list; the held-out
    branch must fall back to ['population_run']."""
    import oaht_bench.population.pooled_crossplay as pc
    from oaht_bench.offline import runner

    seen = {}

    def fake_build_roster(dirs, env):
        seen["dirs"] = [str(d) for d in dirs]
        from oaht_bench.population.pooled_crossplay import RosterEntry

        return [RosterEntry("brdiv", 2, "conf", "p", "cls")]

    monkeypatch.setattr(pc, "build_roster", fake_build_roster)
    batch = _Batch({"held_out": {"brdiv": [2]}, "population_run": "runs/brdiv_run"})
    teammates = runner._teammate_policies(batch, env=None)
    assert seen["dirs"] == ["runs/brdiv_run"]
    assert [t[0] for t in teammates] == ["brdiv:2:conf"]
