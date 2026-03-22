"""Constants for DeepFold-Linear data pipeline.

Adapted from Boltz-1 with molecule type ordering per SPEC §3.1:
    protein=0, rna=1, dna=2, ligand=3
"""

####################################################################################################
# MOLECULE / CHAIN TYPES (SPEC §3.1)
####################################################################################################

# Canonical ordering matching Boltz-1: protein=0, dna=1, rna=2, nonpolymer=3
MOL_PROTEIN = 0
MOL_DNA = 1
MOL_RNA = 2
MOL_NONPOLYMER = 3
MOL_LIGAND = MOL_NONPOLYMER  # backward-compat alias

mol_type_names = ["protein", "dna", "rna", "nonpolymer"]
mol_type_ids = {name: i for i, name in enumerate(mol_type_names)}

# Boltz uses uppercase chain_type strings internally; map them to our ids
chain_type_to_mol_type = {
    "PROTEIN": MOL_PROTEIN,
    "DNA": MOL_DNA,
    "RNA": MOL_RNA,
    "NONPOLYMER": MOL_NONPOLYMER,
}

####################################################################################################
# RESIDUE / TOKEN TYPES
####################################################################################################

canonical_protein_tokens = [
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
    "UNK",  # unknown protein token
]

# Full token vocabulary: pad, gap, 21 protein, 5 RNA, 5 DNA
tokens = [
    "<pad>",
    "-",
    *canonical_protein_tokens,
    # RNA tokens
    "A",
    "G",
    "C",
    "U",
    "N",  # N = unknown RNA
    # DNA tokens
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",  # DN = unknown DNA
]

token_ids = {token: i for i, token in enumerate(tokens)}
num_tokens = len(tokens)

unk_token = {"PROTEIN": "UNK", "DNA": "DN", "RNA": "N"}
unk_token_ids = {m: token_ids[t] for m, t in unk_token.items()}

# Single-letter ↔ three-letter mappings
prot_letter_to_token = {
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
prot_token_to_letter = {v: k for k, v in prot_letter_to_token.items()}
prot_token_to_letter["UNK"] = "X"

rna_letter_to_token = {"A": "A", "G": "G", "C": "C", "U": "U", "N": "N"}
rna_token_to_letter = {v: k for k, v in rna_letter_to_token.items()}

dna_letter_to_token = {"A": "DA", "G": "DG", "C": "DC", "T": "DT", "N": "DN"}
dna_token_to_letter = {v: k for k, v in dna_letter_to_token.items()}

####################################################################################################
# ELEMENTS
####################################################################################################

num_elements = 128

####################################################################################################
# REFERENCE ATOMS PER RESIDUE TYPE
####################################################################################################

# fmt: off
ref_atoms = {
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
    # RNA
    "A":  ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "G":  ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "C":  ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "U":  ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
    "N":  ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"],
    # DNA
    "DA": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "DG": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "DC": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "DT": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"],
    "DN": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"],
}
# fmt: on

# Maximum atoms in any standard residue type
max_atoms_per_token = max(len(v) for v in ref_atoms.values())

protein_backbone_atom_names = ["N", "CA", "C", "O"]
nucleic_backbone_atom_names = [
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
]

protein_backbone_atom_index = {
    name: i for i, name in enumerate(protein_backbone_atom_names)
}
nucleic_backbone_atom_index = {
    name: i for i, name in enumerate(nucleic_backbone_atom_names)
}

# Center atom per residue type (used for token coordinate)
res_to_center_atom = {
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

# Distogram atom per residue type (CB for protein, base atoms for nucleic)
res_to_disto_atom = {
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

# Precompute center/disto atom indices within ref_atoms
res_to_center_atom_id = {
    res: ref_atoms[res].index(atom) for res, atom in res_to_center_atom.items()
}
res_to_disto_atom_id = {
    res: ref_atoms[res].index(atom) for res, atom in res_to_disto_atom.items()
}

# Symmetric atom permutations (for computing symmetric RMSD)
ref_symmetries = {
    "PAD": [],
    "ALA": [],
    "ARG": [],
    "ASN": [],
    "CYS": [],
    "GLN": [],
    "GLY": [],
    "HIS": [],
    "ILE": [],
    "LEU": [],
    "LYS": [],
    "MET": [],
    "PRO": [],
    "SER": [],
    "THR": [],
    "TRP": [],
    "VAL": [],
    "ASP": [[(6, 7), (7, 6)]],
    "GLU": [[(7, 8), (8, 7)]],
    "PHE": [[(6, 7), (7, 6), (8, 9), (9, 8)]],
    "TYR": [[(6, 7), (7, 6), (8, 9), (9, 8)]],
    "A": [[(1, 2), (2, 1)]],
    "G": [[(1, 2), (2, 1)]],
    "C": [[(1, 2), (2, 1)]],
    "U": [[(1, 2), (2, 1)]],
    "DA": [[(1, 2), (2, 1)]],
    "DG": [[(1, 2), (2, 1)]],
    "DC": [[(1, 2), (2, 1)]],
    "DT": [[(1, 2), (2, 1)]],
}

####################################################################################################
# BOND TYPES
####################################################################################################

bond_types = ["OTHER", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "COVALENT"]
bond_type_ids = {bond: i for i, bond in enumerate(bond_types)}
num_bond_types = len(bond_types)

####################################################################################################
# DISTANCE CUTOFFS
####################################################################################################

atom_interface_cutoff = 5.0  # Angstroms
interface_cutoff = 15.0  # Angstroms

####################################################################################################
# MSA
####################################################################################################

max_msa_seqs = 16384
max_paired_seqs = 8192

####################################################################################################
# POSITION / BOND ENCODING (SPEC §4)
####################################################################################################

# 68-bin scheme: bins 0..64 = same-chain sequence separation,
#                bin 65 = cross-chain no bond,
#                bin 66 = covalent bond same chain,
#                bin 67 = covalent bond cross chain
num_rel_pos_bins = 68
max_rel_seq_sep = 32  # clip(g_i - g_j, -32, 32)

####################################################################################################
# VDW RADII (indexed by atomic number, 0-based)
####################################################################################################

# fmt: off
vdw_radii = [
    1.2, 1.4, 2.2, 1.9, 1.8, 1.7, 1.6, 1.55, 1.5, 1.54,
    2.4, 2.2, 2.1, 2.1, 1.95, 1.8, 1.8, 1.88, 2.8, 2.4,
    2.3, 2.15, 2.05, 2.05, 2.05, 2.05, 2.0, 2.0, 2.0, 2.1,
    2.1, 2.1, 2.05, 1.9, 1.9, 2.02, 2.9, 2.55, 2.4, 2.3,
    2.15, 2.1, 2.05, 2.05, 2.0, 2.05, 2.1, 2.2, 2.2, 2.25,
    2.2, 2.1, 2.1, 2.16, 3.0, 2.7, 2.5, 2.48, 2.47, 2.45,
    2.43, 2.42, 2.4, 2.38, 2.37, 2.35, 2.33, 2.32, 2.3, 2.28,
    2.27, 2.25, 2.2, 2.1, 2.05, 2.0, 2.0, 2.05, 2.1, 2.05,
    2.2, 2.3, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.4,
    2.0, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
]
# fmt: on

####################################################################################################
# EXCLUDED LIGANDS (crystallization artifacts, buffers, etc.)
####################################################################################################

ligand_exclusion = {
    "144",
    "15P",
    "1PE",
    "2F2",
    "2JC",
    "3HR",
    "3SY",
    "7N5",
    "7PE",
    "9JE",
    "AAE",
    "ABA",
    "ACE",
    "ACN",
    "ACT",
    "ACY",
    "AZI",
    "BAM",
    "BCN",
    "BCT",
    "BDN",
    "BEN",
    "BME",
    "BO3",
    "BTB",
    "BTC",
    "BU1",
    "C8E",
    "CAD",
    "CAQ",
    "CBM",
    "CCN",
    "CIT",
    "CL",
    "CLR",
    "CM",
    "CMO",
    "CO3",
    "CPT",
    "CXS",
    "D10",
    "DEP",
    "DIO",
    "DMS",
    "DN",
    "DOD",
    "DOX",
    "EDO",
    "EEE",
    "EGL",
    "EOH",
    "EOX",
    "EPE",
    "ETF",
    "FCY",
    "FJO",
    "FLC",
    "FMT",
    "FW5",
    "GOL",
    "GSH",
    "GTT",
    "GYF",
    "HED",
    "IHP",
    "IHS",
    "IMD",
    "IOD",
    "IPA",
    "IPH",
    "LDA",
    "MB3",
    "MEG",
    "MES",
    "MLA",
    "MLI",
    "MOH",
    "MPD",
    "MRD",
    "MSE",
    "MYR",
    "N",
    "NA",
    "NH2",
    "NH4",
    "NHE",
    "NO3",
    "O4B",
    "OHE",
    "OLA",
    "OLC",
    "OMB",
    "OME",
    "OXA",
    "P6G",
    "PE3",
    "PE4",
    "PEG",
    "PEO",
    "PEP",
    "PG0",
    "PG4",
    "PGE",
    "PGR",
    "PLM",
    "PO4",
    "POL",
    "POP",
    "PVO",
    "SAR",
    "SCN",
    "SEO",
    "SEP",
    "SIN",
    "SO4",
    "SPD",
    "SPM",
    "SR",
    "STE",
    "STO",
    "STU",
    "TAR",
    "TBU",
    "TME",
    "TPO",
    "TRS",
    "UNK",
    "UNL",
    "UNX",
    "UPL",
    "URE",
}
