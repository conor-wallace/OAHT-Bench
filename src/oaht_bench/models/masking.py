"""Action-availability masking, shared by every policy that samples actions.

Kept at the model layer because both the reactive actor-critics and the
return-conditioned offline agents need it at inference, and the offline losses
need it in training. It carries no dataset or training dependency, so it does not
pull the offline stack into :mod:`oaht_bench.models`.
"""

from __future__ import annotations


def mask_logits(logits, avail):
    """Suppress unavailable actions, as every absorbed policy does.

    ``logits - (1 - avail) * 1e10`` (``agents/mlp_actor_critic.py:36-37``) rather
    than ``-inf``, which would make a fully-masked row NaN after softmax.

    Collection already applies this: every seat's ``get_action`` receives
    ``avail_actions``, so a recorded action is always legal. Without it here the
    learned policy is trained and evaluated under a weaker constraint than the
    data was generated under -- on LBF that is 20.5% of (step, action) pairs, and
    action 5 is unavailable 67% of the time.
    """
    if avail is None:
        return logits
    return logits - (1.0 - avail) * 1e10
