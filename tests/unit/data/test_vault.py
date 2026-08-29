"""Flashbax Vault round-trip: the flat store must reconstruct the exact
``EpisodeBatch`` it was given, so a consumer cannot tell npz from vault
(``docs/dataset_design.md`` §2).
"""

import numpy as np

from oaht_bench.data.schema import EpisodeBatch
from oaht_bench.data.vault import read_vault, to_flat, write_vault


def _batch(*, lengths=(3, 5, 2), num_agents=2, obs_dim=4, num_actions=5, meta=None):
    """A batch shaped as pad_and_stack makes them: padded to the max real length."""
    e = len(lengths)
    t = max(lengths)
    rng = np.random.default_rng(0)
    obs = np.zeros((e, num_agents, t, obs_dim), np.float32)
    actions = np.zeros((e, num_agents, t), np.int64)
    rewards = np.zeros((e, num_agents, t), np.float32)
    avail = np.zeros((e, num_agents, t, num_actions), np.float32)
    dones = np.zeros((e, t), bool)
    valid = np.zeros((e, t), bool)
    for i, n in enumerate(lengths):
        valid[i, :n] = True
        obs[i, :, :n] = rng.normal(size=(num_agents, n, obs_dim))
        actions[i, :, :n] = rng.integers(0, num_actions, size=(num_agents, n))
        rewards[i, :, :n] = rng.normal(size=(num_agents, n))
        avail[i, :, :n] = rng.integers(0, 2, size=(num_agents, n, num_actions))
        dones[i, n - 1] = True  # terminate on the last real step
    member_ids = rng.integers(0, 5, size=(e, num_agents))
    return EpisodeBatch(
        obs=obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        valid=valid,
        avail_actions=avail,
        member_ids=member_ids,
        ego_index=0,
        meta=meta or {"variant": "expert", "env": "lbf_12x12"},
    )


def _assert_equal(a: EpisodeBatch, b: EpisodeBatch):
    assert a.ego_index == b.ego_index
    for f in ("obs", "actions", "rewards", "dones", "valid", "avail_actions", "member_ids"):
        np.testing.assert_array_equal(
            np.asarray(getattr(a, f)), np.asarray(getattr(b, f)), err_msg=f
        )


def test_to_flat_drops_padding_and_marks_boundaries():
    batch = _batch(lengths=(3, 5, 2))
    exp, meta = to_flat(batch)
    n = 3 + 5 + 2
    assert exp["observations"].shape == (1, n, 2, 4)
    assert exp["actions"].shape == (1, n, 2)
    # episode_id names each transition's source episode, in order.
    assert list(exp["episode_id"][0]) == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
    assert meta["ego_index"] == 0
    assert "ego_response_quality" not in exp  # none in this batch's meta


def test_round_trip_reconstructs_the_batch(tmp_path):
    batch = _batch(lengths=(3, 5, 2))
    write_vault(batch, tmp_path / "dataset.vlt")
    back = read_vault(tmp_path / "dataset.vlt", variant="expert")
    _assert_equal(batch, back)
    assert back.meta["env"] == "lbf_12x12"
    assert back.meta["variant"] == "expert"


def test_round_trip_carries_pooled_epsilon_labels(tmp_path):
    # Pooled collections record a per-episode ego_response_quality; it must survive
    # as a broadcast flat field and be restored to the per-episode meta list.
    meta = {
        "variant": "br_vs_worst",
        "env": "lbf_12x12",
        "mode": "pooled",
        "ego_response_quality": [1.0, 0.0, 0.5],
        "roster": [{"generator": "fcp", "member": 0, "role": "self"}],
    }
    batch = _batch(lengths=(3, 5, 2), meta=meta)
    exp, _ = to_flat(batch)
    assert exp["ego_response_quality"].shape == (1, 10)
    # broadcast: episode 1 (5 transitions) all carry 0.0
    assert list(exp["ego_response_quality"][0]) == [1, 1, 1, 0, 0, 0, 0, 0, 0.5, 0.5]

    write_vault(batch, tmp_path / "dataset.vlt")
    back = read_vault(tmp_path / "dataset.vlt", variant="br_vs_worst")
    _assert_equal(batch, back)
    assert back.meta["ego_response_quality"] == [1.0, 0.0, 0.5]
    assert back.meta["roster"] == meta["roster"]


def test_variants_are_separate_uids_under_one_vault(tmp_path):
    # OG-MARL's {Good,Medium,Poor} layout: one .vlt dir, a sub-dir per variant.
    expert = _batch(meta={"variant": "expert", "env": "lbf_12x12"})
    worst = _batch(meta={"variant": "br_vs_worst", "env": "lbf_12x12"})
    write_vault(expert, tmp_path / "dataset.vlt")
    write_vault(worst, tmp_path / "dataset.vlt")
    assert (tmp_path / "dataset.vlt" / "expert").is_dir()
    assert (tmp_path / "dataset.vlt" / "br_vs_worst").is_dir()
    # Each reads back to its own variant, not the other.
    assert read_vault(tmp_path / "dataset.vlt", variant="expert").meta["variant"] == "expert"
    assert (
        read_vault(tmp_path / "dataset.vlt", variant="br_vs_worst").meta["variant"] == "br_vs_worst"
    )


def test_meta_matches_npz_round_trip(tmp_path):
    # The vault's meta must equal what the npz store would give back, so the two
    # backends are interchangeable.
    batch = _batch(meta={"variant": "expert", "paired_roles": False, "eligible_members": [0, 1, 2]})
    batch.save(tmp_path / "dataset.npz")
    npz_meta = EpisodeBatch.load(tmp_path / "dataset.npz").meta
    write_vault(batch, tmp_path / "dataset.vlt")
    vault_meta = read_vault(tmp_path / "dataset.vlt", variant="expert").meta
    assert vault_meta == npz_meta
