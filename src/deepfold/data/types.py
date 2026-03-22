"""Data types for DeepFold-Linear.

Clean Python dataclasses replacing Boltz-1's numpy structured arrays.
All tensor-bearing types use torch.Tensor for GPU compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor


####################################################################################################
# ATOM-LEVEL TYPES
####################################################################################################


@dataclass
class Atom:
    """Single atom record.

    Attributes
    ----------
    name : str
        Atom name (e.g. "CA", "N", "O5'").  Up to 4 characters.
    element : int
        Atomic number (1-based, 0 = unknown).  Range [0, 127].
    coords : Tensor
        (3,) float32 — x, y, z coordinates in Angstroms.
    charge : int
        Formal charge.
    is_present : bool
        Whether this atom has resolved coordinates.
    """

    name: str
    element: int
    coords: Tensor  # (3,)
    charge: int = 0
    is_present: bool = True


@dataclass
class Bond:
    """Covalent bond between two atoms.

    Attributes
    ----------
    atom_1 : int
        Global index of first atom.
    atom_2 : int
        Global index of second atom.
    type : int
        Bond type id (see const.bond_type_ids).
    """

    atom_1: int
    atom_2: int
    type: int = 0


####################################################################################################
# RESIDUE / CHAIN
####################################################################################################


@dataclass
class Residue:
    """A residue (or nucleotide, or ligand group).

    Attributes
    ----------
    name : str
        Residue name (e.g. "ALA", "DA", "ATP").
    res_type : int
        Token vocabulary index (see const.token_ids).
    res_idx : int
        Residue index within the chain (0-based).
    atom_idx : int
        Start index into the global atom list.
    atom_num : int
        Number of atoms belonging to this residue.
    atom_center : int
        Global index of the center atom (CA / C1').
    atom_disto : int
        Global index of the distogram atom (CB / base).
    is_standard : bool
        Whether this is a standard residue type.
    is_present : bool
        Whether at least one atom is resolved.
    """

    name: str
    res_type: int
    res_idx: int
    atom_idx: int
    atom_num: int
    atom_center: int
    atom_disto: int
    is_standard: bool = True
    is_present: bool = True


@dataclass
class Chain:
    """A polymer chain or ligand entity.

    Attributes
    ----------
    name : str
        Chain identifier string (e.g. "A", "B").
    mol_type : int
        Molecule type (0=protein, 1=rna, 2=dna, 3=ligand per SPEC §3.1).
    entity_id : int
        Entity id — identical sequences share the same entity_id.
    sym_id : int
        Symmetry copy index within this entity.
    asym_id : int
        Unique chain index across the structure.
    atom_idx : int
        Start index into the global atom list.
    atom_num : int
        Total number of atoms in this chain.
    res_idx : int
        Start index into the global residue list.
    res_num : int
        Number of residues in this chain.
    """

    name: str
    mol_type: int
    entity_id: int
    sym_id: int
    asym_id: int
    atom_idx: int
    atom_num: int
    res_idx: int
    res_num: int


####################################################################################################
# TOKEN (for tokenized representation fed to model)
####################################################################################################


@dataclass
class Token:
    """A single token in the tokenized representation.

    One token = one residue (protein), one nucleotide (RNA/DNA),
    or one atom/group (ligand).

    Attributes
    ----------
    token_idx : int
        Global token index.
    atom_idx : int
        Start index into the atom list.
    atom_num : int
        Number of atoms belonging to this token.
    res_idx : int
        Residue index.
    res_type : int
        Token vocabulary index.
    asym_id : int
        Chain index.
    entity_id : int
        Entity index.
    mol_type : int
        Molecule type (0=protein, 1=rna, 2=dna, 3=ligand).
    center_idx : int
        Global atom index of center atom.
    resolved_mask : bool
        Whether the token has resolved coordinates.
    """

    token_idx: int
    atom_idx: int
    atom_num: int
    res_idx: int
    res_type: int
    asym_id: int
    entity_id: int
    mol_type: int
    center_idx: int
    resolved_mask: bool = True


####################################################################################################
# MSA DATA
####################################################################################################


@dataclass
class MSAData:
    """Multiple sequence alignment data for a single chain.

    Attributes
    ----------
    sequences : Tensor
        (S, N_prot) int — residue type indices per MSA row.
    deletions : Tensor
        (S, N_prot) float — deletion counts (or normalized values).
    """

    sequences: Tensor  # (S, N_prot) int
    deletions: Tensor  # (S, N_prot) float


####################################################################################################
# STRUCTURE
####################################################################################################


@dataclass
class Structure:
    """A parsed molecular structure.

    Attributes
    ----------
    atoms : list[Atom]
        All atoms in the structure.
    residues : list[Residue]
        All residues / nucleotides / ligand groups.
    chains : list[Chain]
        All chains.
    bonds : list[Bond]
        All covalent bonds.
    """

    atoms: list[Atom] = field(default_factory=list)
    residues: list[Residue] = field(default_factory=list)
    chains: list[Chain] = field(default_factory=list)
    bonds: list[Bond] = field(default_factory=list)

    @property
    def num_atoms(self) -> int:
        return len(self.atoms)

    @property
    def num_residues(self) -> int:
        return len(self.residues)

    @property
    def num_chains(self) -> int:
        return len(self.chains)

    def get_coords(self) -> Tensor:
        """Return (N_atom, 3) coordinate tensor."""
        if len(self.atoms) == 0:
            return torch.zeros(0, 3)
        return torch.stack([a.coords for a in self.atoms], dim=0)

    def get_present_mask(self) -> Tensor:
        """Return (N_atom,) boolean mask of resolved atoms."""
        return torch.tensor([a.is_present for a in self.atoms], dtype=torch.bool)


####################################################################################################
# MODEL INPUT FEATURES (SPEC §3)
####################################################################################################


@dataclass
class ProteinFeatures:
    """All tensors needed for a single forward pass.

    Shapes use N = number of tokens, N_atom = number of atoms,
    S = MSA depth, N_prot = number of protein tokens with MSA.

    Attributes — Input Embedding (SPEC §3.1)
    -----------------------------------------
    token_type : Tensor
        (N,) int — molecule type per token (0-3).
    profile : Tensor
        (N, 32) float — MSA frequency profile, zero for non-protein.
    del_mean : Tensor
        (N, 1) float — mean deletion count, zero for non-protein.
    has_msa : Tensor
        (N, 1) float — 1.0 for protein/RNA tokens, 0.0 otherwise.

    Attributes — MSA (SPEC §3.2)
    -----------------------------
    msa_feat : Tensor
        (S, N_prot, 34) float — concatenated MSA features.

    Attributes — Atom Reference (SPEC §3.3, §3.4)
    -----------------------------------------------
    c_atom : Tensor
        (N_atom, 128) float — frozen atom single representation.
    p_lm : Tensor
        (num_local_pairs, 16) float — frozen atom pair representation.
    p_lm_idx : Tensor
        (num_local_pairs, 2) int — (atom_l, atom_m) index pairs for p_lm.
    token_idx : Tensor
        (N_atom,) int — maps each atom to its token index.

    Attributes — Position Encoding (SPEC §4)
    ------------------------------------------
    chain_id : Tensor
        (N,) int — chain index per token.
    global_idx : Tensor
        (N,) int — global residue index per token (preserved after crop).
    bond_matrix : Tensor
        (N, N) bool — covalent bond adjacency at token level.
        NOTE: dense for reference impl; will be sparse in production.
    protein_mask : Tensor
        (N,) bool — True for protein tokens (used to index MSA).

    Attributes — Training targets
    ------------------------------
    x_atom_true : Tensor or None
        (N_atom, 3) float — ground truth atom coordinates.
    x_res_true : Tensor or None
        (N, 3) float — ground truth token (center atom) coordinates.
    """

    # Input embedding
    token_type: Tensor  # (N,) int
    profile: Tensor  # (N, 32)
    del_mean: Tensor  # (N, 1)
    has_msa: Tensor  # (N, 1)

    # MSA
    msa_feat: Tensor  # (S, N_prot, 34)

    # Atom reference
    c_atom: Tensor  # (N_atom, 128)
    p_lm: Tensor  # (local_pairs, 16)
    p_lm_idx: Tensor  # (local_pairs, 2)
    token_idx: Tensor  # (N_atom,)

    # Position encoding
    chain_id: Tensor  # (N,)
    global_idx: Tensor  # (N,)
    bond_matrix: Tensor  # (N, N) bool
    protein_mask: Tensor  # (N,) bool

    # Training targets (optional)
    x_atom_true: Optional[Tensor] = None  # (N_atom, 3)
    x_res_true: Optional[Tensor] = None  # (N, 3)

    def to(self, device: torch.device) -> "ProteinFeatures":
        """Move all tensors to the given device."""
        kwargs = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor):
                kwargs[k] = v.to(device)
            else:
                kwargs[k] = v
        return ProteinFeatures(**kwargs)

    def pin_memory(self) -> "ProteinFeatures":
        """Pin all tensors for faster host→device transfer."""
        kwargs = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor):
                kwargs[k] = v.pin_memory()
            else:
                kwargs[k] = v
        return ProteinFeatures(**kwargs)


####################################################################################################
# RECORD METADATA (for dataset management)
####################################################################################################


@dataclass
class StructureInfo:
    """Metadata about a structure entry."""

    resolution: Optional[float] = None
    method: Optional[str] = None
    deposited: Optional[str] = None
    released: Optional[str] = None
    revised: Optional[str] = None
    num_chains: Optional[int] = None
    num_interfaces: Optional[int] = None


@dataclass
class ChainInfo:
    """Metadata about a chain within a structure."""

    chain_id: int
    chain_name: str
    mol_type: int
    cluster_id: int | str
    msa_id: int | str
    num_residues: int
    valid: bool = True
    entity_id: Optional[int | str] = None
    template_id: Optional[int | str] = None


@dataclass
class Record:
    """A dataset record pointing to structure + MSA files.

    Serializable to/from JSON for manifest files.
    """

    id: str
    structure: StructureInfo
    chains: list[ChainInfo]

    def dump(self, path: Path) -> None:
        """Write record to JSON file."""
        import dataclasses

        with path.open("w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Record":
        """Load record from JSON file."""
        with path.open("r") as f:
            data = json.load(f)
        return cls(
            id=data["id"],
            structure=StructureInfo(**data["structure"]),
            chains=[ChainInfo(**c) for c in data["chains"]],
        )
