"""Offline baselines built on one shared sequence backbone (§3.1).

The backbone is TAO's In-context Control Decoder (its Appendix F), which is the
published precedent for this design: one architecture, each baseline stating
exactly what it adds. Differences between baselines are therefore differences
the papers describe rather than ones we introduced.

Both are two-stage: stage 1 learns a teammate representation, stage 2 trains the
policy against a frozen encoder. LIAM's original single-loop training with a
``stop_gradient`` was a consequence of learning online; offline, staging removes
the need for it. So the two differ only in what the encoder reads and how its
output reaches the policy:

  LIAM  encoder over the *ego* history -> embedding concatenated to the observation
  TAO   encoder over the *teammate* stream -> embedding as cross-attention key/value

**A dataset requirement neither method can be run honestly without.** Both are
trained to predict the *ego* action, and TAO specifically targets *near-optimal*
actions ``a^{1,*}``. Our collected datasets seat two population members against
each other, so the "ego" stream is another teammate's behaviour, not a best
response to the teammate being modelled. Cloning it trains the policy toward
population-average play. BRDiv and L-BRDiv already train a best response per
confederate (``final_params_br``), so for those two the data exists; FCP and
CoMeDi have no best response and would need one trained.
"""

from oaht_bench.models.backbone import DecisionTransformer
from oaht_bench.offline.bc import BcTrainer, bc_loss
from oaht_bench.offline.liam import (
    LiamTrainer,
    liam_policy_loss,
    liam_reconstruction_loss,
)
from oaht_bench.offline.meliba import (
    MelibaTrainer,
    meliba_belief,
    meliba_policy_loss,
    meliba_reconstruction_loss,
)
from oaht_bench.offline.omis import (
    OmisTrainer,
    omis_actor_loss,
    omis_representation_loss,
    omis_search,
)
from oaht_bench.offline.registry import BaseAhtTrainer, get_trainer
from oaht_bench.offline.tao import (
    TaoTrainer,
    embedding_loss,
    supervised_contrastive,
    tao_policy_loss,
)

__all__ = [
    "BaseAhtTrainer",
    "DecisionTransformer",
    "LiamTrainer",
    "get_trainer",
    "MelibaTrainer",
    "OmisTrainer",
    "BcTrainer",
    "bc_loss",
    "TaoTrainer",
    "embedding_loss",
    "supervised_contrastive",
    "tao_policy_loss",
    "liam_policy_loss",
    "liam_reconstruction_loss",
    "meliba_belief",
    "meliba_policy_loss",
    "meliba_reconstruction_loss",
    "omis_actor_loss",
    "omis_representation_loss",
    "omis_search",
]
