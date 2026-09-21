"""Environments: LBF, Overcooked-v1 and Hanabi over Jumanji/JaxMARL.

``make_env`` and the wrappers beneath it are absorbed from jax-aht (MIT); see
``PROVENANCE.md``. :func:`make` is ours and builds environments from validated
configs.
"""

from oaht_bench.envs.factory import make
from oaht_bench.envs.make_env import make_env

__all__ = ["make", "make_env"]
