"""The batched collector must be transition-for-transition identical to the eager
per-step ``collect_episode`` -- it exists only to move that loop onto the
accelerator (``lax.scan`` over steps, ``vmap`` over episodes), not to change the
data. Bit-identity per ep-rng is what lets it replace the eager path.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

_POP = Path("populations/hanabi/comedi")
pytestmark = pytest.mark.skipif(
    not _POP.exists(), reason="released hanabi comedi population absent"
)


def _setup():
    from oaht_bench.configs.env import HanabiConfig
    from oaht_bench.envs import make_env
    from oaht_bench.envs.log_wrapper import LogWrapper
    from oaht_bench.population.pooled_crossplay import build_roster

    env_cfg = HanabiConfig(
        name="hanabi",
        hand_size=5,
        max_info_tokens=8,
        max_life_tokens=3,
        num_cards_of_rank=[3, 2, 2, 2, 1],
        num_colors=5,
        num_ranks=5,
        rollout_length=128,
    )
    base_env = make_env(env_cfg.env_name, env_cfg.env_kwargs())
    roster = build_roster([_POP], LogWrapper(base_env))
    ego = next(e for e in roster if e.role == "self" and int(e.member) == 0)
    return base_env, [(ego.params, ego.policy_cls)] * 2, env_cfg.rollout_length


@pytest.mark.parametrize("greedy", [False, True])
def test_batched_matches_eager(greedy):
    from oaht_bench.dataset.construction.collect import collect_episode, collect_episodes_batched

    base_env, seats, rollout_length = _setup()
    n, rng = 6, jax.random.PRNGKey(0)
    # batch_size < n exercises the chunking path too.
    batched = collect_episodes_batched(
        rng,
        base_env,
        seats,
        max_episode_steps=rollout_length,
        num_episodes=n,
        batch_size=4,
        greedy=greedy,
    )
    ep_rngs = jax.random.split(rng, n)
    for i in range(n):
        eager = collect_episode(
            ep_rngs[i], base_env, seats, max_episode_steps=rollout_length, greedy=greedy
        )
        b = batched[i]
        assert eager.length == b.length, i
        assert np.array_equal(np.asarray(eager.actions), np.asarray(b.actions)), i
        assert np.allclose(np.asarray(eager.obs), np.asarray(b.obs)), i
        assert np.allclose(np.asarray(eager.rewards), np.asarray(b.rewards)), i
        assert np.array_equal(np.asarray(eager.dones), np.asarray(b.dones)), i
