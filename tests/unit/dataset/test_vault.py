"""Flashbax Vault round-trip. Collection is ragged; the vault stores flat
transitions and :func:`read_vault` reconstructs the padded ``EpisodeBatch`` that
``make_windows`` consumes. Padding is a read-side artifact, never stored
(``docs/dataset_design.md`` §2).
"""

import numpy as np

from oaht_bench.dataset.schema import Episode, EpisodeBatch
from oaht_bench.dataset.vault import read_vault, to_flat, write_vault


def _episodes(*, lengths=(3, 5, 2), num_agents=2, obs_dim=4, num_actions=5, meta=None):
    """Ragged Episodes as collect_episode returns them: real steps only, (agent, T, …)."""
    rng = np.random.default_rng(0)
    episodes = [
        Episode(
            obs=rng.normal(size=(num_agents, n, obs_dim)).astype(np.float32),
            actions=rng.integers(0, num_actions, size=(num_agents, n)),
            rewards=rng.normal(size=(num_agents, n)).astype(np.float32),
            avail_actions=rng.integers(0, 2, size=(num_agents, n, num_actions)).astype(np.float32),
            dones=np.arange(n) == (n - 1),  # terminate on the last real step
        )
        for n in lengths
    ]
    member_ids = rng.integers(0, 5, size=(len(lengths), num_agents))
    return episodes, member_ids, meta or {"variant": "expert", "env": "lbf_12x12"}


def _assert_matches(episodes, member_ids, batch: EpisodeBatch):
    """The read-back ragged batch equals the source episodes, per episode."""
    assert batch.num_episodes == len(episodes)
    np.testing.assert_array_equal(batch.member_ids, member_ids)
    for ep, e in enumerate(episodes):
        for f in ("obs", "actions", "rewards", "avail_actions", "dones"):
            np.testing.assert_array_equal(
                np.asarray(getattr(batch.episodes[ep], f)), np.asarray(getattr(e, f)),
                err_msg=f"{f}[{ep}]",
            )


def test_to_flat_concatenates_ragged_and_marks_boundaries():
    episodes, member_ids, meta = _episodes(lengths=(3, 5, 2))
    exp, vmeta = to_flat(episodes, member_ids, ego_index=0, meta=meta)
    n = 3 + 5 + 2
    assert exp["observations"].shape == (1, n, 2, 4)
    assert exp["actions"].shape == (1, n, 2)
    # episode_id names each transition's source episode, in order -- no padding.
    assert list(exp["episode_id"][0]) == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
    assert vmeta["ego_index"] == 0
    assert "ego_response_quality" not in exp  # none in this batch's meta


def test_round_trip_reconstructs_the_ragged_batch(tmp_path):
    episodes, member_ids, meta = _episodes(lengths=(3, 5, 2))
    write_vault(episodes, member_ids, tmp_path / "dataset.vlt", ego_index=0, meta=meta)
    back = read_vault(tmp_path / "dataset.vlt", variant="expert")
    # Variable-length episodes come back with their own lengths, not padded.
    assert list(back.episode_lengths()) == [3, 5, 2]
    _assert_matches(episodes, member_ids, back)
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
    episodes, member_ids, meta = _episodes(lengths=(3, 5, 2), meta=meta)
    exp, _ = to_flat(episodes, member_ids, ego_index=0, meta=meta)
    assert exp["ego_response_quality"].shape == (1, 10)
    # broadcast: episode 1 (5 transitions) all carry 0.0
    assert list(exp["ego_response_quality"][0]) == [1, 1, 1, 0, 0, 0, 0, 0, 0.5, 0.5]

    write_vault(episodes, member_ids, tmp_path / "dataset.vlt", ego_index=0, meta=meta)
    back = read_vault(tmp_path / "dataset.vlt", variant="br_vs_worst")
    assert back.meta["ego_response_quality"] == [1.0, 0.0, 0.5]
    assert back.meta["roster"] == meta["roster"]


def test_variants_are_separate_uids_under_one_vault(tmp_path):
    # OG-MARL's {Good,Medium,Poor} layout: one .vlt dir, a sub-dir per variant.
    e1, m1, _ = _episodes(meta={"variant": "expert", "env": "lbf_12x12"})
    e2, m2, _ = _episodes(meta={"variant": "br_vs_worst", "env": "lbf_12x12"})
    write_vault(e1, m1, tmp_path / "dataset.vlt", ego_index=0, meta={"variant": "expert"})
    write_vault(e2, m2, tmp_path / "dataset.vlt", ego_index=0, meta={"variant": "br_vs_worst"})
    assert (tmp_path / "dataset.vlt" / "expert").is_dir()
    assert (tmp_path / "dataset.vlt" / "br_vs_worst").is_dir()
    assert read_vault(tmp_path / "dataset.vlt", variant="expert").meta["variant"] == "expert"
    assert read_vault(tmp_path / "dataset.vlt", variant="br_vs_worst").meta["variant"] == "br_vs_worst"


def test_read_vault_autodiscovers_a_single_variant(tmp_path):
    # A collection writes one variant; the offline reader loads it without naming it.
    episodes, member_ids, meta = _episodes(meta={"generator": "test"})  # no 'variant' -> uid 'data'
    write_vault(episodes, member_ids, tmp_path / "dataset.vlt", ego_index=0, meta=meta)
    back = read_vault(tmp_path / "dataset.vlt")  # no variant= given
    _assert_matches(episodes, member_ids, back)
