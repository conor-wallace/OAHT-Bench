"""Roll a trained policy into the environment against real teammates (§8).

Everything before this produces losses. A falling behaviour-cloning loss says
the policy predicts dataset actions better; it does not say the policy
coordinates, and on offline data those come apart — which is the whole reason
the benchmark exists. This is the first number worth reporting.

Each offline baseline is a
:class:`~oaht_bench.models.return_conditioned_agent.ReturnConditionedAgent`, so
:func:`evaluate_agent` drives it through the shared vmapped ``run_episodes`` loop:
the return-conditioning deployment (a target return decremented by the reward
received, over a rolling left-padded ``K``-window matching training) lives in the
agent's ``get_action``, not here.

Results are reported **per teammate** as well as averaged. An average hides the
failure mode this benchmark is about: a policy that plays well with the
teammates resembling its training data and badly with the rest scores the same
as one that is uniformly mediocre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvalScores:
    """Returns from evaluating one policy against a set of teammates."""

    #: teammate member index -> mean episode return for the ego agent
    per_teammate: dict[int, float]
    #: teammate member index -> standard error over episodes
    per_teammate_stderr: dict[int, float]
    episodes_per_teammate: int
    target_return: float

    @property
    def mean_return(self) -> float:
        """Averaged over teammates, not over episodes.

        Each teammate gets equal weight regardless of how many episodes it
        appeared in, so the number is not tilted by collection coverage.
        """
        return float(np.mean(list(self.per_teammate.values())))

    @property
    def worst_teammate_return(self) -> float:
        """The teammate the policy plays worst with.

        Ad-hoc teamwork is about the partners you did not train for, so the
        floor is at least as informative as the mean.
        """
        return float(min(self.per_teammate.values()))

    def describe(self) -> str:
        rows = "\n".join(
            f"    teammate {t:3d}   {v:7.4f} ± {self.per_teammate_stderr[t]:.4f}"
            for t, v in sorted(self.per_teammate.items())
        )
        return (
            f"target return {self.target_return:.4f}, "
            f"{self.episodes_per_teammate} episodes each\n{rows}\n"
            f"    mean {self.mean_return:.4f}   worst {self.worst_teammate_return:.4f}"
        )


def dataset_target_return(batch, *, quantile: float = 1.0) -> float:
    """Target return to condition on, taken from the data.

    The reference sets this per opponent from a config table. We have no such
    table, so it comes from the dataset the policy was trained on: the given
    quantile of per-episode ego return. Conditioning on the maximum asks the
    policy for the best behaviour the data contains, which is the usual
    Decision Transformer convention.
    """
    return float(np.quantile(batch.episode_returns()[:, batch.ego_index], quantile))


def evaluate_agent(
    agent,
    params,
    env,
    loaded,
    members,
    *,
    rng,
    target_return: float,
    max_episode_steps: int,
    num_episodes: int = 20,
    ego_index: int = 0,
) -> EvalScores:
    """Play each teammate ``num_episodes`` times and record the ego's returns.

    The ego is a :class:`~oaht_bench.models.agent_interface.AgentPolicy` (every
    offline baseline is a
    :class:`~oaht_bench.models.return_conditioned_agent.ReturnConditionedAgent`),
    driven through the shared, vmapped
    :func:`~oaht_bench.common.run_episodes.run_episodes`. The rolling window and
    return-to-go bookkeeping live in the agent, so nothing here needs the context
    length, observation dimension or normalisation -- they are baked into the
    agent. The ego takes seat ``agent_0``; the return is the ego seat of
    LogWrapper's per-agent episode return.
    """
    import jax

    from oaht_bench.common.run_episodes import run_episodes
    from oaht_bench.population.members import get_member_params

    per_teammate, stderr = {}, {}
    for m in members:
        mate_params = get_member_params(loaded.params, int(m))
        rng, ep_rng = jax.random.split(rng)
        out = run_episodes(
            ep_rng,
            env,
            agent_0_param=params,
            agent_0_policy=agent,
            agent_1_param=mate_params,
            agent_1_policy=loaded.policy_cls,
            max_episode_steps=max_episode_steps,
            num_eps=num_episodes,
        )
        returns = np.asarray(out["returned_episode_returns"])[:, ego_index]
        per_teammate[int(m)] = float(returns.mean())
        stderr[int(m)] = (
            float(returns.std(ddof=1) / np.sqrt(len(returns))) if len(returns) > 1 else 0.0
        )

    return EvalScores(
        per_teammate=per_teammate,
        per_teammate_stderr=stderr,
        episodes_per_teammate=num_episodes,
        target_return=float(target_return),
    )
