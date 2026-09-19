"""The environment interface the training code actually requires.

``BaseEnv`` cannot be used as the annotation, because training runs against a
*wrapped* environment: ``LogWrapper(make_env(...))`` inherits from JaxMARL's
``JaxMARLWrapper`` and shares no ancestor with ``BaseEnv``. Annotating either
concrete type would be wrong for the other, and a union of two unrelated classes
says nothing about what is needed.

A structural protocol says exactly what the training loop touches, and both the
raw environment and the wrapped one satisfy it without either having to know
this module exists.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import chex


@runtime_checkable
class TrainingEnv(Protocol):
    """A multi-agent environment usable by the training loops.

    Deliberately narrow: only the members the PPO loop and the generators call.
    Widening it should be a considered act, since every added member is a
    constraint on anything that wants to stand in as an environment.
    """

    #: Agent identifiers, ordered. Index 0 is conventionally the ego agent.
    agents: list[str]

    @property
    def num_agents(self) -> int: ...

    def reset(self, key: chex.PRNGKey) -> tuple[dict[str, chex.Array], Any]: ...

    def step(
        self,
        key: chex.PRNGKey,
        state: Any,
        actions: dict[str, chex.Array],
        reset_state: Any | None = None,
    ) -> tuple[dict[str, chex.Array], Any, dict[str, float], dict[str, bool], dict]: ...

    def action_space(self, agent: str) -> Any: ...

    def observation_space(self, agent: str) -> Any: ...

    def get_avail_actions(self, state: Any) -> dict[str, chex.Array]: ...
