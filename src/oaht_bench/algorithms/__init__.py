"""Baseline algorithms.

Populated in dependency order (project plan §10.6):

* learning-history family — AD, DPT, AMAGO-Offline, Hybrid-AD, absorbed from
  ICRL4AHT and made environment-generic
* trajectory-view family — LIAM and MeLIBA as modeling modules over the shared
  backbone (offline conversions specified in TAO Appendix F), plus TAO, OMIS and
  TAGET reimplemented in JAX
* floors and reference rows — random, %BC, Prompt-DT, and the FIAM-style oracle

The shared sequence-model backbone these plug into lives in
:mod:`oaht_bench.offline`, not here; this package holds what distinguishes each
method (§3.1).
"""
