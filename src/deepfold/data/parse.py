"""Structure parsing for DeepFold-Linear.

Simplified parser adapted from Boltz-1 mmCIF parsing.
Handles:
  1. mmCIF files via gemmi -> extract atoms, residues, chains, bonds
  2. Pre-processed NPZ files (Boltz format) for compatibility
  3. Basic connection (covalent bond) parsing
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gemmi
import numpy as np

# ---------------------------------------------------------------------------
# Constants (minimal subset; mirrors Boltz const.py)
# ---------------------------------------------------------------------------

CHAIN_TYPES = {"PROTEIN": 0, "DNA": 1, "RNA": 2, "NONPOLYMER": 3}

# Boltz-1 token vocabulary (pad, gap, 20 AA, 5 RNA, 5 DNA)
TOKENS = [
    "<pad>",
    "-",
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
    "A",
    "G",
    "C",
    "U",
    "N",  # RNA
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",  # DNA
]
TOKEN_IDS = {t: i for i, t in enumerate(TOKENS)}
NUM_TOKENS = len(TOKENS)
UNK_TOKEN = {"PROTEIN": "UNK", "DNA": "DN", "RNA": "N"}
UNK_TOKEN_IDS = {m: TOKEN_IDS[t] for m, t in UNK_TOKEN.items()}

# Reference atom lists per standard residue (same ordering as Boltz)
# fmt: off
REF_ATOMS: dict[str, list[str]] = {
    "PAD": [],
    "UNK": ["N", "CA", "C", "O", "CB"],
    "-": [],
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "A":  ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'","N9","C8","N7","C5","C6","N6","N1","C2","N3","C4"],
    "G":  ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'","N9","C8","N7","C5","C6","O6","N1","C2","N2","N3","C4"],
    "C":  ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'","N1","C2","O2","N3","C4","N4","C5","C6"],
    "U":  ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'","N1","C2","O2","N3","C4","O4","C5","C6"],
    "N":  ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'"],
    "DA": ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","C1'","N9","C8","N7","C5","C6","N6","N1","C2","N3","C4"],
    "DG": ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","C1'","N9","C8","N7","C5","C6","O6","N1","C2","N2","N3","C4"],
    "DC": ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","C1'","N1","C2","O2","N3","C4","N4","C5","C6"],
    "DT": ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","C1'","N1","C2","O2","N3","C4","O4","C5","C7","C6"],
    "DN": ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","C1'"],
}
# fmt: on

# Center / distogram atoms per residue type
RES_TO_CENTER_ATOM: dict[str, str] = {
    "UNK": "CA",
    "ALA": "CA",
    "ARG": "CA",
    "ASN": "CA",
    "ASP": "CA",
    "CYS": "CA",
    "GLN": "CA",
    "GLU": "CA",
    "GLY": "CA",
    "HIS": "CA",
    "ILE": "CA",
    "LEU": "CA",
    "LYS": "CA",
    "MET": "CA",
    "PHE": "CA",
    "PRO": "CA",
    "SER": "CA",
    "THR": "CA",
    "TRP": "CA",
    "TYR": "CA",
    "VAL": "CA",
    "A": "C1'",
    "G": "C1'",
    "C": "C1'",
    "U": "C1'",
    "N": "C1'",
    "DA": "C1'",
    "DG": "C1'",
    "DC": "C1'",
    "DT": "C1'",
    "DN": "C1'",
}

RES_TO_DISTO_ATOM: dict[str, str] = {
    "UNK": "CB",
    "ALA": "CB",
    "ARG": "CB",
    "ASN": "CB",
    "ASP": "CB",
    "CYS": "CB",
    "GLN": "CB",
    "GLU": "CB",
    "GLY": "CA",
    "HIS": "CB",
    "ILE": "CB",
    "LEU": "CB",
    "LYS": "CB",
    "MET": "CB",
    "PHE": "CB",
    "PRO": "CB",
    "SER": "CB",
    "THR": "CB",
    "TRP": "CB",
    "TYR": "CB",
    "VAL": "CB",
    "A": "C4",
    "G": "C4",
    "C": "C2",
    "U": "C2",
    "N": "C1'",
    "DA": "C4",
    "DG": "C4",
    "DC": "C2",
    "DT": "C2",
    "DN": "C1'",
}

RES_TO_CENTER_ATOM_ID = {
    res: REF_ATOMS[res].index(atom) for res, atom in RES_TO_CENTER_ATOM.items()
}
RES_TO_DISTO_ATOM_ID = {
    res: REF_ATOMS[res].index(atom) for res, atom in RES_TO_DISTO_ATOM.items()
}

# Letter <-> 3-letter-code mappings for polymers
PROT_LETTER_TO_TOKEN = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
    "J": "UNK",
    "B": "UNK",
    "Z": "UNK",
    "O": "UNK",
    "U": "UNK",
    "-": "-",
}

# ---------------------------------------------------------------------------
# Numpy structured dtypes (compatible with Boltz NPZ)
# ---------------------------------------------------------------------------

AtomDtype = np.dtype(
    [
        ("name", "<U4"),
        ("coords", "3f4"),
        ("is_present", "?"),
        ("bfactor", "f4"),
    ]
)

BondDtype = np.dtype(
    [
        ("chain_1", "i4"),
        ("chain_2", "i4"),
        ("res_1", "i4"),
        ("res_2", "i4"),
        ("atom_1", "i4"),
        ("atom_2", "i4"),
        ("type", "i1"),
    ]
)

ResidueDtype = np.dtype(
    [
        ("name", "<U5"),
        ("res_type", "i1"),
        ("res_idx", "i4"),
        ("atom_idx", "i4"),
        ("atom_num", "i4"),
        ("atom_center", "i4"),
        ("atom_disto", "i4"),
        ("is_standard", "?"),
        ("is_present", "?"),
    ]
)

ChainDtype = np.dtype(
    [
        ("name", "<U5"),
        ("mol_type", "i1"),
        ("entity_id", "i4"),
        ("sym_id", "i4"),
        ("asym_id", "i4"),
        ("atom_idx", "i4"),
        ("atom_num", "i4"),
        ("res_idx", "i4"),
        ("res_num", "i4"),
    ]
)

ConnectionDtype = np.dtype(
    [
        ("chain_1", "i4"),
        ("chain_2", "i4"),
        ("res_1", "i4"),
        ("res_2", "i4"),
        ("atom_1", "i4"),
        ("atom_2", "i4"),
    ]
)

InterfaceDtype = np.dtype(
    [
        ("chain_1", "i4"),
        ("chain_2", "i4"),
    ]
)


# ---------------------------------------------------------------------------
# Parsed intermediate dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedAtom:
    name: str
    coords: tuple[float, float, float]
    is_present: bool
    bfactor: float = 0.0


@dataclass(frozen=True, slots=True)
class ParsedResidue:
    name: str
    type: int
    idx: int
    atoms: list[ParsedAtom]
    atom_center: int
    atom_disto: int
    is_standard: bool
    is_present: bool
    orig_idx: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ParsedChain:
    name: str
    entity: str
    type: int
    residues: list[ParsedResidue]
    sequence: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ParsedConnection:
    chain_1: str
    chain_2: str
    residue_index_1: int
    residue_index_2: int
    atom_index_1: int
    atom_index_2: int


@dataclass(frozen=True)
class Structure:
    """Flat numpy-array based structure representation."""

    atoms: np.ndarray  # AtomDtype
    bonds: np.ndarray  # BondDtype
    residues: np.ndarray  # ResidueDtype
    chains: np.ndarray  # ChainDtype
    connections: np.ndarray  # ConnectionDtype
    interfaces: np.ndarray  # InterfaceDtype
    mask: np.ndarray  # bool per chain
    sequences: dict[str, str]  # chain_name -> 1-letter sequence

    def save(self, path: Path) -> None:
        np.savez_compressed(
            str(path),
            atoms=self.atoms,
            bonds=self.bonds,
            residues=self.residues,
            chains=self.chains,
            connections=self.connections,
            interfaces=self.interfaces,
            mask=self.mask,
        )

    @classmethod
    def load_npz(cls, path: Path) -> "Structure":
        """Load from a Boltz-compatible NPZ file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            atoms=data["atoms"],
            bonds=data["bonds"],
            residues=data["residues"],
            chains=data["chains"],
            connections=data.get("connections", np.array([], dtype=ConnectionDtype)),
            interfaces=data.get("interfaces", np.array([], dtype=InterfaceDtype)),
            mask=data.get("mask", np.ones(len(data["chains"]), dtype=bool)),
            sequences={},
        )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get_resolution(block: gemmi.cif.Block) -> float:
    for key in (
        "_refine.ls_d_res_high",
        "_em_3d_reconstruction.resolution",
        "_reflns.d_resolution_high",
    ):
        with contextlib.suppress(Exception):
            return float(block.find([key])[0].str(0))
    return 0.0


def _get_polymer_type(entity: gemmi.Entity) -> Optional[int]:
    """Map gemmi polymer type to our chain type id."""
    name = entity.polymer_type.name
    mapping = {
        "PeptideL": CHAIN_TYPES["PROTEIN"],
        "Dna": CHAIN_TYPES["DNA"],
        "Rna": CHAIN_TYPES["RNA"],
    }
    return mapping.get(name)


def _unk_token_for_polymer(polymer_type: gemmi.PolymerType) -> str:
    if polymer_type == gemmi.PolymerType.PeptideL:
        return UNK_TOKEN["PROTEIN"]
    elif polymer_type == gemmi.PolymerType.Dna:
        return UNK_TOKEN["DNA"]
    elif polymer_type == gemmi.PolymerType.Rna:
        return UNK_TOKEN["RNA"]
    raise ValueError(f"Unsupported polymer type: {polymer_type}")


# ---------------------------------------------------------------------------
# Polymer parsing
# ---------------------------------------------------------------------------


def _parse_polymer(
    polymer: gemmi.ResidueSpan,
    polymer_type: gemmi.PolymerType,
    sequence: list[str],
    chain_id: str,
    entity_name: str,
) -> Optional[ParsedChain]:
    """Parse a polymer chain from gemmi into ParsedChain.

    Aligns the full entity sequence to the observed polymer residues,
    extracting coordinates for reference atoms in canonical order.
    Non-standard / modified residues that are not in REF_ATOMS are
    mapped to UNK (protein) or the corresponding unknown nucleotide.
    """
    # Remove microheterogeneities
    sequence = [gemmi.Entity.first_mon(item) for item in sequence]

    ref_res_set = set(TOKENS)

    # Align sequence to observed polymer
    result = gemmi.align_sequence_to_polymer(
        sequence, polymer, polymer_type, gemmi.AlignmentScoring()
    )

    i = 0  # polymer residue index
    parsed: list[ParsedResidue] = []

    for j, match in enumerate(result.match_string):
        res_name = sequence[j]
        res: Optional[gemmi.Residue] = None
        name_to_atom: dict[str, gemmi.Atom] = {}

        if match == "|":
            res = polymer[i]
            name_to_atom = {a.name.upper(): a for a in res}
            if res.name != res_name:
                raise ValueError(
                    f"Alignment mismatch: expected {res_name}, got {res.name}"
                )
            i += 1

        # MSE -> MET substitution
        if res_name == "MSE":
            res_name = "MET"
            if "SE" in name_to_atom:
                name_to_atom["SD"] = name_to_atom["SE"]
        elif res_name not in ref_res_set:
            # Map unknown modified residues to UNK
            unk = _unk_token_for_polymer(polymer_type)
            res_name = unk

        # Get the canonical atom list for this residue type
        atom_names = REF_ATOMS.get(res_name, [])
        atoms: list[ParsedAtom] = []
        for atom_name in atom_names:
            if atom_name in name_to_atom:
                a = name_to_atom[atom_name]
                coords = (a.pos.x, a.pos.y, a.pos.z)
                atoms.append(ParsedAtom(atom_name, coords, True, a.b_iso))
            else:
                atoms.append(ParsedAtom(atom_name, (0.0, 0.0, 0.0), False, 0.0))

        # Original sequence index for connection lookup
        orig_idx: Optional[str] = None
        if res is not None:
            sid = res.seqid
            orig_idx = str(sid.num) + str(sid.icode).strip()

        center_id = RES_TO_CENTER_ATOM_ID.get(res_name, 0)
        disto_id = RES_TO_DISTO_ATOM_ID.get(res_name, 0)

        parsed.append(
            ParsedResidue(
                name=res_name,
                type=TOKEN_IDS.get(res_name, UNK_TOKEN_IDS.get("PROTEIN", 0)),
                idx=j,
                atoms=atoms,
                atom_center=center_id,
                atom_disto=disto_id,
                is_standard=True,
                is_present=res is not None,
                orig_idx=orig_idx,
            )
        )

    chain_type = _get_polymer_type_id(polymer_type)
    return ParsedChain(
        name=chain_id,
        entity=entity_name,
        residues=parsed,
        type=chain_type,
        sequence=gemmi.one_letter_code(sequence),
    )


def _get_polymer_type_id(polymer_type: gemmi.PolymerType) -> int:
    if polymer_type == gemmi.PolymerType.PeptideL:
        return CHAIN_TYPES["PROTEIN"]
    elif polymer_type == gemmi.PolymerType.Dna:
        return CHAIN_TYPES["DNA"]
    elif polymer_type == gemmi.PolymerType.Rna:
        return CHAIN_TYPES["RNA"]
    raise ValueError(f"Unknown polymer type: {polymer_type}")


# ---------------------------------------------------------------------------
# Non-polymer (ligand / ion) parsing
# ---------------------------------------------------------------------------


def _parse_nonpolymer_residue(
    residue: gemmi.Residue,
    res_idx: int,
) -> ParsedResidue:
    """Parse a non-polymer residue (ligand / ion / branched sugar).

    Each atom becomes its own entry; the whole CCD component is one residue.
    We do *not* depend on RDKit here -- just take coordinates from gemmi.
    """
    atoms: list[ParsedAtom] = []
    for atom in residue:
        atom: gemmi.Atom
        if atom.element.name == "H":
            continue
        coords = (atom.pos.x, atom.pos.y, atom.pos.z)
        atoms.append(ParsedAtom(atom.name, coords, True, atom.b_iso))

    if not atoms:
        # Fallback: single dummy atom
        atoms.append(ParsedAtom("X", (0.0, 0.0, 0.0), False, 0.0))

    unk_id = UNK_TOKEN_IDS["PROTEIN"]
    orig_idx = str(residue.seqid.num) + str(residue.seqid.icode).strip()

    return ParsedResidue(
        name=residue.name,
        type=unk_id,
        idx=res_idx,
        atoms=atoms,
        atom_center=0,
        atom_disto=0,
        is_standard=False,
        is_present=True,
        orig_idx=orig_idx,
    )


# ---------------------------------------------------------------------------
# Connection parsing
# ---------------------------------------------------------------------------


def _parse_connections(
    structure: gemmi.Structure,
    chains: list[ParsedChain],
    subchain_map: dict[tuple[str, str], str],
) -> list[ParsedConnection]:
    """Parse covalent connections from the gemmi structure."""
    chain_map = {c.name: c for c in chains}
    connections: list[ParsedConnection] = []

    for conn in structure.connections:
        conn: gemmi.Connection
        if conn.type.name != "Covale":
            continue
        try:
            c1_name = conn.partner1.chain_name
            c2_name = conn.partner2.chain_name
            r1_id = conn.partner1.res_id.seqid
            r1_id = str(r1_id.num) + str(r1_id.icode).strip()
            r2_id = conn.partner2.res_id.seqid
            r2_id = str(r2_id.num) + str(r2_id.icode).strip()

            sub1 = subchain_map.get((c1_name, r1_id))
            sub2 = subchain_map.get((c2_name, r2_id))
            if sub1 is None or sub2 is None:
                continue

            chain1 = chain_map.get(sub1)
            chain2 = chain_map.get(sub2)
            if chain1 is None or chain2 is None:
                continue

            # Find residue index and atom index within that residue
            r1_idx = next(
                (i for i, r in enumerate(chain1.residues) if r.orig_idx == r1_id),
                None,
            )
            r2_idx = next(
                (i for i, r in enumerate(chain2.residues) if r.orig_idx == r2_id),
                None,
            )
            if r1_idx is None or r2_idx is None:
                continue

            a1_name = conn.partner1.atom_name
            a2_name = conn.partner2.atom_name
            a1_idx = next(
                (
                    i
                    for i, a in enumerate(chain1.residues[r1_idx].atoms)
                    if a.name == a1_name
                ),
                None,
            )
            a2_idx = next(
                (
                    i
                    for i, a in enumerate(chain2.residues[r2_idx].atoms)
                    if a.name == a2_name
                ),
                None,
            )
            if a1_idx is None or a2_idx is None:
                continue

            connections.append(
                ParsedConnection(
                    chain_1=sub1,
                    chain_2=sub2,
                    residue_index_1=r1_idx,
                    residue_index_2=r2_idx,
                    atom_index_1=a1_idx,
                    atom_index_2=a2_idx,
                )
            )
        except Exception:  # noqa: BLE001
            continue

    return connections


# ---------------------------------------------------------------------------
# Interface computation
# ---------------------------------------------------------------------------


def _compute_interfaces(
    atoms: np.ndarray,
    chains: np.ndarray,
    cutoff: float = 5.0,
) -> np.ndarray:
    """Compute pairwise chain interfaces (heavy atoms within *cutoff* A)."""
    # Build per-atom chain id array
    chain_ids = np.concatenate(
        [
            np.full(int(c["atom_num"]), idx, dtype=np.int32)
            for idx, c in enumerate(chains)
        ]
    )

    coords = atoms["coords"]
    mask = atoms["is_present"]
    if mask.sum() < 2:
        return np.array([], dtype=InterfaceDtype)

    coords_m = coords[mask]
    cids_m = chain_ids[mask]

    # Simple O(N^2) distance -- fine for preprocessed structures
    from scipy.spatial import cKDTree  # local import to keep module light

    tree = cKDTree(coords_m)
    pairs = tree.query_pairs(cutoff)
    iface_set: set[tuple[int, int]] = set()
    for i, j in pairs:
        c1, c2 = int(cids_m[i]), int(cids_m[j])
        if c1 != c2:
            iface_set.add((min(c1, c2), max(c1, c2)))

    return (
        np.array(list(iface_set), dtype=InterfaceDtype)
        if iface_set
        else np.array([], dtype=InterfaceDtype)
    )


# ---------------------------------------------------------------------------
# Main mmCIF parser
# ---------------------------------------------------------------------------

BOND_TYPE_COVALENT = 5  # matches Boltz const.bond_type_ids["COVALENT"]


def parse_mmcif(
    path: str | Path,
    *,
    use_assembly: bool = True,
    compute_ifaces: bool = True,
) -> Structure:
    """Parse an mmCIF file into a :class:`Structure`.

    This is a simplified version of Boltz-1's ``parse_mmcif``.  It does *not*
    use RDKit for CCD component lookup; non-polymer residues are parsed
    directly from the gemmi model.  This keeps the dependency footprint small
    and is sufficient for v1.

    Parameters
    ----------
    path : str | Path
        Path to the mmCIF file.
    use_assembly : bool
        Whether to expand biological assembly 1.
    compute_ifaces : bool
        Whether to compute chain-chain interface pairs.

    Returns
    -------
    Structure
    """
    path = str(path)
    block = gemmi.cif.read(path)[0]
    _get_resolution(block)

    # Load gemmi structure
    struct = gemmi.make_structure_from_block(block)
    struct.merge_chain_parts()
    struct.remove_waters()
    struct.remove_hydrogens()
    struct.remove_alternative_conformations()
    struct.remove_empty_chains()

    # Optionally expand assembly
    if use_assembly and struct.assemblies:
        how = gemmi.HowToNameCopiedChain.AddNumber
        struct.transform_to_assembly(struct.assemblies[0].name, how=how)

    # Build entity maps
    entities: dict[str, gemmi.Entity] = {}
    entity_ids: dict[str, int] = {}
    for eid, entity in enumerate(struct.entities):
        if entity.entity_type.name == "Water":
            continue
        for sid in entity.subchains:
            entities[sid] = entity
            entity_ids[sid] = eid

    # Chain/residue -> subchain map (needed for connections)
    subchain_map: dict[tuple[str, str], str] = {}
    for chain in struct[0]:
        for residue in chain:
            seq_id = residue.seqid
            key = (chain.name, str(seq_id.num) + str(seq_id.icode).strip())
            subchain_map[key] = residue.subchain

    # Parse chains
    parsed_chains: list[ParsedChain] = []
    for raw_chain in struct[0].subchains():
        subchain_id = raw_chain.subchain_id()
        if subchain_id not in entities:
            continue
        entity = entities[subchain_id]
        etype = entity.entity_type.name

        if etype == "Polymer":
            chain_type_id = _get_polymer_type(entity)
            if chain_type_id is None:
                continue
            try:
                pc = _parse_polymer(
                    polymer=raw_chain,
                    polymer_type=entity.polymer_type,
                    sequence=entity.full_sequence,
                    chain_id=subchain_id,
                    entity_name=entity.name,
                )
            except Exception:  # noqa: BLE001
                continue
            if pc is not None:
                parsed_chains.append(pc)

        elif etype in {"NonPolymer", "Branched"}:
            residues = []
            for lig_idx, ligand in enumerate(raw_chain):
                if ligand.name == "UNL":
                    continue
                residues.append(_parse_nonpolymer_residue(ligand, lig_idx))
            if residues:
                parsed_chains.append(
                    ParsedChain(
                        name=subchain_id,
                        entity=entity.name,
                        residues=residues,
                        type=CHAIN_TYPES["NONPOLYMER"],
                    )
                )

    if not parsed_chains:
        raise ValueError(f"No chains parsed from {path}")

    # Parse connections
    connections = _parse_connections(struct, parsed_chains, subchain_map)

    # Convert to flat numpy arrays
    atom_data: list[tuple] = []
    bond_data: list[tuple] = []
    res_data: list[tuple] = []
    chain_data: list[tuple] = []
    chain_to_idx: dict[str, int] = {}
    res_to_idx: dict[tuple[str, int], tuple[int, int]] = {}
    chain_to_seq: dict[str, str] = {}

    atom_idx = 0
    res_idx = 0
    sym_count: dict[int, int] = {}

    for asym_id, chain in enumerate(parsed_chains):
        res_num = len(chain.residues)
        atom_num = sum(len(r.atoms) for r in chain.residues)
        eid = entity_ids.get(chain.name, asym_id)
        sym_id = sym_count.get(eid, 0)

        chain_data.append(
            (
                chain.name,
                chain.type,
                eid,
                sym_id,
                asym_id,
                atom_idx,
                atom_num,
                res_idx,
                res_num,
            )
        )
        chain_to_idx[chain.name] = asym_id
        sym_count[eid] = sym_id + 1
        if chain.sequence is not None:
            chain_to_seq[chain.name] = chain.sequence

        for i, res in enumerate(chain.residues):
            atom_center = atom_idx + res.atom_center
            atom_disto = atom_idx + res.atom_disto
            res_data.append(
                (
                    res.name,
                    res.type,
                    res.idx,
                    atom_idx,
                    len(res.atoms),
                    atom_center,
                    atom_disto,
                    res.is_standard,
                    res.is_present,
                )
            )
            res_to_idx[(chain.name, i)] = (res_idx, atom_idx)

            for atom in res.atoms:
                atom_data.append(
                    (atom.name, atom.coords, atom.is_present, atom.bfactor)
                )
                atom_idx += 1

            res_idx += 1

    # Convert connections to bond table entries
    for conn in connections:
        c1_idx = chain_to_idx.get(conn.chain_1)
        c2_idx = chain_to_idx.get(conn.chain_2)
        if c1_idx is None or c2_idx is None:
            continue
        r1_info = res_to_idx.get((conn.chain_1, conn.residue_index_1))
        r2_info = res_to_idx.get((conn.chain_2, conn.residue_index_2))
        if r1_info is None or r2_info is None:
            continue
        r1_idx, a1_off = r1_info
        r2_idx, a2_off = r2_info
        bond_data.append(
            (
                c1_idx,
                c2_idx,
                r1_idx,
                r2_idx,
                a1_off + conn.atom_index_1,
                a2_off + conn.atom_index_2,
                BOND_TYPE_COVALENT,
            )
        )

    # Build final numpy arrays
    atoms = (
        np.array(atom_data, dtype=AtomDtype)
        if atom_data
        else np.empty(0, dtype=AtomDtype)
    )
    bonds = (
        np.array(bond_data, dtype=BondDtype)
        if bond_data
        else np.empty(0, dtype=BondDtype)
    )
    residues = (
        np.array(res_data, dtype=ResidueDtype)
        if res_data
        else np.empty(0, dtype=ResidueDtype)
    )
    chains = (
        np.array(chain_data, dtype=ChainDtype)
        if chain_data
        else np.empty(0, dtype=ChainDtype)
    )
    conn_arr = (
        np.array(
            [(b[0], b[1], b[2], b[3], b[4], b[5]) for b in bond_data],
            dtype=ConnectionDtype,
        )
        if bond_data
        else np.empty(0, dtype=ConnectionDtype)
    )
    mask = np.ones(len(chain_data), dtype=bool)

    if compute_ifaces and len(atoms) > 0:
        interfaces = _compute_interfaces(atoms, chains)
    else:
        interfaces = np.array([], dtype=InterfaceDtype)

    return Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        connections=conn_arr,
        interfaces=interfaces,
        mask=mask,
        sequences=chain_to_seq,
    )
