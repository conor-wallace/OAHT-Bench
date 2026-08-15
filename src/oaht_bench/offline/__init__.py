"""Offline baselines built on one shared sequence backbone (§3.1).

The backbone is TAO's In-context Control Decoder (its Appendix F), which is the
published precedent for this design: one architecture, each baseline stating
exactly what it adds. Differences between baselines are therefore differences
the papers describe rather than ones we introduced.

  LIAM  = backbone - cross-attention + teammate-reconstruction head
  TAO   = backbone + cross-attention, keyed by a learned policy embedding

**A dataset requirement neither method can be run honestly without.** Both are
trained to predict the *ego* action, and TAO specifically targets *near-optimal*
actions ``a^{1,*}``. Our collected datasets seat two population members against
each other, so the "ego" stream is another teammate's behaviour, not a best
response to the teammate being modelled. Cloning it trains the policy toward
population-average play. BRDiv and L-BRDiv already train a best response per
confederate (``final_params_br``), so for those two the data exists; FCP and
CoMeDi have no best response and would need one trained.
"""

from oaht_bench.offline.backbone import ControlDecoder
from oaht_bench.offline.dataset import Windows, make_windows, return_to_go
from oaht_bench.offline.liam import LiamOffline, liam_loss
from oaht_bench.offline.tao import (
    AncillaryActionDecoder,
    OpponentPolicyEncoder,
    TaoPolicy,
    embedding_loss,
    supervised_contrastive,
)

__all__ = [
    "AncillaryActionDecoder",
    "ControlDecoder",
    "LiamOffline",
    "OpponentPolicyEncoder",
    "TaoPolicy",
    "Windows",
    "embedding_loss",
    "supervised_contrastive",
    "liam_loss",
    "make_windows",
    "return_to_go",
]
