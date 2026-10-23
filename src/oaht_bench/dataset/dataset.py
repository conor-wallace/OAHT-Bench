"""The offline training :class:`Dataset`: a thin holder of the collected
episode store (:class:`~oaht_bench.dataset.vault.VaultReader`) plus the
normalisation each baseline applies when it windows that store.

Windowing itself has no class here: each baseline builds its own training
batch with :func:`~oaht_bench.dataset.windowing.window_slice_batch`/
:func:`~oaht_bench.dataset.windowing.teammate_episode_ids`
(:mod:`oaht_bench.dataset.windowing`), composed however that baseline's own
paper describes its batch -- see ``offline/bc.py``, ``offline/liam.py``,
``offline/tao.py``, ``offline/omis.py``. ``Dataset`` only resolves the seats
and the normalisation once, since every one of those methods needs both and
neither is a window-level concept.

Return-to-go is computed rather than stored, because it is a function of the
rewards already in the artifact and storing it would let the two disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oaht_bench.dataset.vault import VaultReader
from oaht_bench.dataset.windowing import resolve_seats


@dataclass(frozen=True)
class Normalization:
    """The transform applied to a dataset, so evaluation can repeat it.

    Stored rather than recomputed: a policy trained on standardised
    observations must see standardised observations at rollout, and a target
    return expressed in raw units has to be divided by the same scale it was
    trained under.
    """

    obs_mean: np.ndarray
    obs_std: np.ndarray
    rtg_scale: float

    def apply_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean) / self.obs_std

    def apply_rtg(self, rtg):
        return rtg / self.rtg_scale


class Dataset:
    """The offline training dataset: load and process only, in the spirit of
    PyTorch's ``Dataset`` (data, not sampling).

    Constructed from the path to the data, it reads the episodes off disk and
    resolves the ego/mate seats and the observation/return-to-go
    normalisation -- so a baseline runner hands over a path and gets what
    every batch method needs. Drawing batches is deliberately not its job:
    see :mod:`oaht_bench.dataset.windowing` and each baseline's own
    ``_sample_batch``.
    """

    def __init__(
        self,
        vault_dir: str | Path,
        *,
        teammate_index: int | None = None,
        normalize: bool = True,
        variant: str | None = None,
        rtg_scale: float | None = None,
    ):
        """Read the Flashbax Vault at ``vault_dir`` and resolve its transforms.

        ``variant`` selects the sub-directory; if omitted and the vault holds
        exactly one, that one is used (see :class:`~oaht_bench.dataset.vault.VaultReader`).
        The normalisation is read from the collection-time sidecar when
        present, else computed once in a bounded pass (see
        :meth:`VaultReader.norm_stats`).

        ``rtg_scale``, when given, overrides the *computed* return-to-go scale
        (the dataset-wide RTG standard deviation) with a fixed constant --
        matching references (e.g. OMIS App. H.2's ``reward_scale_dict``) that
        divide by one environment-specific number rather than a learned
        statistic. ``None`` (default) keeps today's computed scale exactly.
        Ignored when ``normalize=False`` (there is no ``Normalization`` to
        carry it).
        """
        self.batch = VaultReader(vault_dir, variant=variant)
        self.ego, self.mate = resolve_seats(self.batch, teammate_index)
        if normalize:
            obs_mean, obs_std, computed_rtg_scale = self.batch.norm_stats()
            self.norm = Normalization(
                obs_mean, obs_std, rtg_scale if rtg_scale is not None else computed_rtg_scale
            )
        else:
            self.norm = None

    @property
    def obs_dim(self) -> int:
        return int(self.batch._obs_dim)

    @property
    def action_dim(self) -> int:
        """Size of the action space, from the collected availability masks."""
        return int(self.batch.episodes[0].avail_actions.shape[-1])

    @property
    def meta(self) -> dict:
        return self.batch.meta


def return_to_go(rewards: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Reverse-cumulative reward, zeroed past the end of the episode.

    ``rewards`` and ``valid`` are ``(episode, T)``. Padding contributes nothing,
    which matters because episodes are padded to a common length and a nonzero
    tail would inflate every earlier target.
    """
    r = np.asarray(rewards) * np.asarray(valid)
    return np.flip(np.cumsum(np.flip(r, axis=-1), axis=-1), axis=-1)
