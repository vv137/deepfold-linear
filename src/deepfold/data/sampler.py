"""Cluster-weighted sampler (AF3-style, Boltz ClusterSampler).

Weight formula per chain:
    w = β / n_clust × (α_prot × n_prot + α_nucl × n_nuc + α_ligand × n_ligand)

Rarer clusters get upweighted so the model sees diverse structures.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler

from deepfold.data import const
from deepfold.data.types import ChainInfo, Record

logger = logging.getLogger(__name__)


def _chain_weight(
    chain: ChainInfo,
    cluster_sizes: dict[str, int],
    alpha_prot: float,
    alpha_nucl: float,
    alpha_ligand: float,
    beta: float,
) -> float:
    """Compute sampling weight for a single chain."""
    cid = str(chain.cluster_id)
    n_clust = cluster_sizes.get(cid, 1)
    mol = chain.mol_type
    if mol == const.MOL_PROTEIN:
        alpha = alpha_prot
    elif mol in (const.MOL_RNA, const.MOL_DNA):
        alpha = alpha_nucl
    elif mol == const.MOL_LIGAND:
        alpha = alpha_ligand
    else:
        alpha = 1.0
    return beta * alpha / n_clust


class ClusterWeightedSampler(Sampler):
    """AF3-style cluster-weighted infinite sampler.

    Pre-computes per-record sampling weights from chain cluster IDs and
    molecule types. Yields record indices infinitely.

    Parameters
    ----------
    records : list[Record]
        Dataset records with chain metadata.
    alpha_prot : float
        Weight multiplier for protein chains.
    alpha_nucl : float
        Weight multiplier for nucleic acid chains.
    alpha_ligand : float
        Weight multiplier for ligand chains.
    beta : float
        Base weight factor.
    seed : int
        Random seed.
    samples_per_epoch : int
        Number of samples per "epoch" (for __len__).
    """

    def __init__(
        self,
        records: list[Record],
        alpha_prot: float = 3.0,
        alpha_nucl: float = 3.0,
        alpha_ligand: float = 1.0,
        beta: float = 1.0,
        seed: int = 42,
        samples_per_epoch: int = 100_000,
    ):
        self.records = records
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.RandomState(seed)

        # Build cluster size counts across all chains in dataset
        cluster_counts: Counter[str] = Counter()
        for rec in records:
            for chain in rec.chains:
                cluster_counts[str(chain.cluster_id)] += 1

        # Compute per-record weight = sum of chain weights
        weights = np.zeros(len(records), dtype=np.float64)
        for i, rec in enumerate(records):
            for chain in rec.chains:
                weights[i] += _chain_weight(
                    chain, cluster_counts, alpha_prot, alpha_nucl, alpha_ligand, beta
                )

        # Handle records with no chains
        weights = np.maximum(weights, 1e-12)

        # Normalize to probability distribution
        self.probs = weights / weights.sum()

        n_clusters = len(cluster_counts)
        logger.info(
            "ClusterWeightedSampler: %d records, %d clusters, "
            "top weight=%.4f, min weight=%.6f",
            len(records), n_clusters, self.probs.max(), self.probs.min(),
        )

    def __iter__(self):
        while True:
            idx = self.rng.choice(len(self.records), p=self.probs)
            yield idx

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """No-op for compatibility with DDP epoch setting."""
        pass


def load_manifest(manifest_path: str | Path) -> list[Record]:
    """Load a list of Records from a JSON manifest file.

    The manifest is a JSON array of Record dicts, or a directory of
    per-record JSON files.
    """
    import json

    path = Path(manifest_path)

    if path.is_dir():
        records = []
        for p in sorted(path.glob("*.json")):
            records.append(Record.load(p))
        return records

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        records = []
        for entry in data:
            from deepfold.data.types import StructureInfo
            records.append(Record(
                id=entry["id"],
                structure=StructureInfo(**entry["structure"]),
                chains=[ChainInfo(**c) for c in entry["chains"]],
            ))
        return records

    raise ValueError(f"Unsupported manifest format: {path}")
