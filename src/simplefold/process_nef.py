"""
NEF to MELD Format Converter for SimpleFold (Fixed Heavy Atom Mapping)
"""

import os
import argparse
from typing import List, Tuple, Dict
from dataclasses import dataclass

# ============================================================================
# RESTRAINT DATA STRUCTURES
# ============================================================================

@dataclass
class DistanceRestraint:
    res_i: int
    atom_i: str
    res_j: int
    atom_j: str
    distance: float

@dataclass
class TorsionRestraint:
    res1: int
    atom1: str
    res2: int
    atom2: str
    res3: int
    atom3: str
    res4: int
    atom4: str
    phi_min: float
    phi_max: float

# ============================================================================
# HEAVY ATOM MAPPING - CORRECTED
# ============================================================================

# Map (residue_type, hydrogen_atom_name) -> (list_of_heavy_atoms, distance_correction)
map_to_heavy = {

    # Alpha hydrogens (HA) -> CA
    ('ALA', 'HA'): (['CA'], 1.0),
    ('ARG', 'HA'): (['CA'], 1.0),
    ('ASN', 'HA'): (['CA'], 1.0),
    ('ASP', 'HA'): (['CA'], 1.0),
    ('CYS', 'HA'): (['CA'], 1.0),
    ('GLN', 'HA'): (['CA'], 1.0),
    ('GLU', 'HA'): (['CA'], 1.0),
    ('HIS', 'HA'): (['CA'], 1.0),
    ('ILE', 'HA'): (['CA'], 1.0),
    ('LEU', 'HA'): (['CA'], 1.0),
    ('LYS', 'HA'): (['CA'], 1.0),
    ('MET', 'HA'): (['CA'], 1.0),
    ('PHE', 'HA'): (['CA'], 1.0),
    ('PRO', 'HA'): (['CA'], 1.0),
    ('SER', 'HA'): (['CA'], 1.0),
    ('THR', 'HA'): (['CA'], 1.0),
    ('TRP', 'HA'): (['CA'], 1.0),
    ('TYR', 'HA'): (['CA'], 1.0),
    ('VAL', 'HA'): (['CA'], 1.0),

    # Backbone amide hydrogens -> N
    ('ALA', 'H'): (['N'], 1.0), ('ARG', 'H'): (['N'], 1.0),
    ('ASN', 'H'): (['N'], 1.0), ('ASP', 'H'): (['N'], 1.0),
    ('CYS', 'H'): (['N'], 1.0), ('GLN', 'H'): (['N'], 1.0),
    ('GLU', 'H'): (['N'], 1.0), ('GLY', 'H'): (['N'], 1.0),
    ('HIS', 'H'): (['N'], 1.0), ('ILE', 'H'): (['N'], 1.0),
    ('LEU', 'H'): (['N'], 1.0), ('LYS', 'H'): (['N'], 1.0),
    ('MET', 'H'): (['N'], 1.0), ('PHE', 'H'): (['N'], 1.0),
    ('PRO', 'H'): (['N'], 1.0), ('SER', 'H'): (['N'], 1.0),
    ('THR', 'H'): (['N'], 1.0), ('TRP', 'H'): (['N'], 1.0),
    ('TYR', 'H'): (['N'], 1.0), ('VAL', 'H'): (['N'], 1.0),
    
    # Beta hydrogens -> CB (generic for all amino acids with CB)
    ('ALA', 'HB'): (['CB'], 1.0), ('ALA', 'HB%'): (['CB'], 1.0),
    ('ALA', 'HB1'): (['CB'], 1.0), ('ALA', 'HB2'): (['CB'], 1.0), ('ALA', 'HB3'): (['CB'], 1.0),
    ('ARG', 'HB'): (['CB'], 1.0), ('ARG', 'HB%'): (['CB'], 1.0), ('ARG', 'HBx'): (['CB'], 1.0), ('ARG', 'HBy'): (['CB'], 1.0),
    ('ASN', 'HB'): (['CB'], 1.0), ('ASN', 'HB%'): (['CB'], 1.0), ('ASN', 'HBx'): (['CB'], 1.0), ('ASN', 'HBy'): (['CB'], 1.0),
    ('ASP', 'HB'): (['CB'], 1.0), ('ASP', 'HB%'): (['CB'], 1.0), ('ASP', 'HBx'): (['CB'], 1.0), ('ASP', 'HBy'): (['CB'], 1.0),
    ('CYS', 'HB'): (['CB'], 1.0), ('CYS', 'HB%'): (['CB'], 1.0), ('CYS', 'HBx'): (['CB'], 1.0), ('CYS', 'HBy'): (['CB'], 1.0),
    ('GLN', 'HB'): (['CB'], 1.0), ('GLN', 'HB%'): (['CB'], 1.0), ('GLN', 'HBx'): (['CB'], 1.0), ('GLN', 'HBy'): (['CB'], 1.0),
    ('GLU', 'HB'): (['CB'], 1.0), ('GLU', 'HB%'): (['CB'], 1.0), ('GLU', 'HBx'): (['CB'], 1.0), ('GLU', 'HBy'): (['CB'], 1.0),
    ('HIS', 'HB'): (['CB'], 1.0), ('HIS', 'HB%'): (['CB'], 1.0), ('HIS', 'HBx'): (['CB'], 1.0), ('HIS', 'HBy'): (['CB'], 1.0),
    ('ILE', 'HB'): (['CB'], 1.0),
    ('LEU', 'HB'): (['CB'], 1.0), ('LEU', 'HB%'): (['CB'], 1.0), ('LEU', 'HBx'): (['CB'], 1.0), ('LEU', 'HBy'): (['CB'], 1.0),
    ('LYS', 'HB'): (['CB'], 1.0), ('LYS', 'HB%'): (['CB'], 1.0), ('LYS', 'HBx'): (['CB'], 1.0), ('LYS', 'HBy'): (['CB'], 1.0),
    ('MET', 'HB'): (['CB'], 1.0), ('MET', 'HB%'): (['CB'], 1.0), ('MET', 'HBx'): (['CB'], 1.0), ('MET', 'HBy'): (['CB'], 1.0),
    ('PHE', 'HB'): (['CB'], 1.0), ('PHE', 'HB%'): (['CB'], 1.0), ('PHE', 'HBx'): (['CB'], 1.0), ('PHE', 'HBy'): (['CB'], 1.0),
    ('PRO', 'HB'): (['CB'], 1.0), ('PRO', 'HB%'): (['CB'], 1.0), ('PRO', 'HBx'): (['CB'], 1.0), ('PRO', 'HBy'): (['CB'], 1.0),
    ('SER', 'HB'): (['CB'], 1.0), ('SER', 'HB%'): (['CB'], 1.0), ('SER', 'HBx'): (['CB'], 1.0), ('SER', 'HBy'): (['CB'], 1.0),
    ('THR', 'HB'): (['CB'], 1.0),
    ('TRP', 'HB'): (['CB'], 1.0), ('TRP', 'HB%'): (['CB'], 1.0), ('TRP', 'HBx'): (['CB'], 1.0), ('TRP', 'HBy'): (['CB'], 1.0),
    ('TYR', 'HB'): (['CB'], 1.0), ('TYR', 'HB%'): (['CB'], 1.0), ('TYR', 'HBx'): (['CB'], 1.0), ('TYR', 'HBy'): (['CB'], 1.0),
    ('VAL', 'HB'): (['CB'], 1.0),
    
    # ARG
    ('ARG', 'HG'): (['CG'], 1.0), ('ARG', 'HG%'): (['CG'], 1.0), ('ARG', 'HG*'): (['CG'], 1.0),
    ('ARG', 'HG2'): (['CG'], 1.0), ('ARG', 'HG3'): (['CG'], 1.0),
    ('ARG', 'HGx'): (['CG'], 1.0), ('ARG', 'HGy'): (['CG'], 1.0),
    ('ARG', 'HGx%'): (['CG'], 1.0), ('ARG', 'HGy%'): (['CG'], 1.0),
    ('ARG', 'HD'): (['CD'], 1.0), ('ARG', 'HD%'): (['CD'], 1.0), ('ARG', 'HD*'): (['CD'], 1.0),
    ('ARG', 'HD2'): (['CD'], 1.0), ('ARG', 'HD3'): (['CD'], 1.0),
    ('ARG', 'HDx'): (['CD'], 1.0), ('ARG', 'HDy'): (['CD'], 1.0),
    ('ARG', 'HDx%'): (['CD'], 1.0), ('ARG', 'HDy%'): (['CD'], 1.0),
    ('ARG', 'HH1%'): (['NH1'], 1.0), ('ARG', 'HH2%'): (['NH2'], 1.0),
    
    # ASN
    ('ASN', 'HD2%'): (['ND2'], 1.0), ('ASN', 'HD2*'): (['ND2'], 1.0),
    ('ASN', 'HD21'): (['ND2'], 1.0), ('ASN', 'HD22'): (['ND2'], 1.0),
    
    # ASP
    ('ASP', 'OD%'): (['OD1', 'OD2'], 0.0), ('ASP', 'OD*'): (['OD1', 'OD2'], 0.0),
    
    # GLN
    ('GLN', 'HG'): (['CG'], 1.0), ('GLN', 'HG%'): (['CG'], 1.0), ('GLN', 'HG*'): (['CG'], 1.0),
    ('GLN', 'HG2'): (['CG'], 1.0), ('GLN', 'HG3'): (['CG'], 1.0),
    ('GLN', 'HGx'): (['CG'], 1.0), ('GLN', 'HGy'): (['CG'], 1.0),
    ('GLN', 'HGx%'): (['CG'], 1.0), ('GLN', 'HGy%'): (['CG'], 1.0),
    ('GLN', 'HE2%'): (['NE2'], 1.0), ('GLN', 'HE2*'): (['NE2'], 1.0),
    ('GLN', 'HE21'): (['NE2'], 1.0), ('GLN', 'HE22'): (['NE2'], 1.0),
    
    # GLU
    ('GLU', 'HG'): (['CG'], 1.0), ('GLU', 'HG%'): (['CG'], 1.0), ('GLU', 'HG*'): (['CG'], 1.0),
    ('GLU', 'HG2'): (['CG'], 1.0), ('GLU', 'HG3'): (['CG'], 1.0),
    ('GLU', 'HGx'): (['CG'], 1.0), ('GLU', 'HGy'): (['CG'], 1.0),
    ('GLU', 'HGx%'): (['CG'], 1.0), ('GLU', 'HGy%'): (['CG'], 1.0),
    ('GLU', 'OE%'): (['OE1', 'OE2'], 0.0), ('GLU', 'OE*'): (['OE1', 'OE2'], 0.0),
    
    # GLY
    ('GLY', 'HA%'): (['CA'], 1.0), ('GLY', 'HA*'): (['CA'], 1.0),
    ('GLY', 'HA2'): (['CA'], 1.0), ('GLY', 'HA3'): (['CA'], 1.0),
    ('GLY', 'HAx'): (['CA'], 1.0), ('GLY', 'HAy'): (['CA'], 1.0),
    
    # ILE
    ('ILE', 'HG1'): (['CG1'], 1.0), ('ILE', 'HG1%'): (['CG1'], 1.0), ('ILE', 'HG1*'): (['CG1'], 1.0),
    ('ILE', 'HG12'): (['CG1'], 1.0), ('ILE', 'HG13'): (['CG1'], 1.0),
    ('ILE', 'HG2'): (['CG2'], 1.0), ('ILE', 'HG2%'): (['CG2'], 1.0), ('ILE', 'HG2*'): (['CG2'], 1.0),
    ('ILE', 'HD1'): (['CD1'], 1.0), ('ILE', 'HD1%'): (['CD1'], 1.0), ('ILE', 'HD1*'): (['CD1'], 1.0),
    ('ILE', 'HG1x'): (['CG1'], 1.0),
    ('ILE', 'HG1y'): (['CG1'], 1.0),
    ('ILE', 'HG11'): (['CG1'], 1.0),
    ('ILE', 'HG12'): (['CG1'], 1.0),
    ('ILE', 'HG13'): (['CG1'], 1.0),
    
    # LEU
    ('LEU', 'HG'): (['CG'], 1.0),
    ('LEU', 'HD1'): (['CD1'], 1.0), ('LEU', 'HD1%'): (['CD1'], 1.0), ('LEU', 'HD1*'): (['CD1'], 1.0),
    ('LEU', 'HD2'): (['CD2'], 1.0), ('LEU', 'HD2%'): (['CD2'], 1.0), ('LEU', 'HD2*'): (['CD2'], 1.0),
    ('LEU', 'HD%'): (['CD1', 'CD2'], 1.0), ('LEU', 'HD*'): (['CD1', 'CD2'], 1.0),
    ('LEU', 'HDx%'): (['CD1', 'CD2'], 1.0), ('LEU', 'HDy%'): (['CD1', 'CD2'], 1.0),
    ('LEU', 'HG1x'): (['CG'], 1.0),
    ('LEU', 'HG1y'): (['CG'], 1.0),
    
    # LYS
    ('LYS', 'HG'): (['CG'], 1.0), ('LYS', 'HG%'): (['CG'], 1.0), ('LYS', 'HG*'): (['CG'], 1.0),
    ('LYS', 'HG2'): (['CG'], 1.0), ('LYS', 'HG3'): (['CG'], 1.0),
    ('LYS', 'HGx'): (['CG'], 1.0), ('LYS', 'HGy'): (['CG'], 1.0),
    ('LYS', 'HGx%'): (['CG'], 1.0), ('LYS', 'HGy%'): (['CG'], 1.0),
    ('LYS', 'HD'): (['CD'], 1.0), ('LYS', 'HD%'): (['CD'], 1.0), ('LYS', 'HD*'): (['CD'], 1.0),
    ('LYS', 'HD2'): (['CD'], 1.0), ('LYS', 'HD3'): (['CD'], 1.0),
    ('LYS', 'HDx'): (['CD'], 1.0), ('LYS', 'HDy'): (['CD'], 1.0),
    ('LYS', 'HDx%'): (['CD'], 1.0), ('LYS', 'HDy%'): (['CD'], 1.0),
    ('LYS', 'HE'): (['CE'], 1.0), ('LYS', 'HE%'): (['CE'], 1.0), ('LYS', 'HE*'): (['CE'], 1.0),
    ('LYS', 'HE2'): (['CE'], 1.0), ('LYS', 'HE3'): (['CE'], 1.0),
    ('LYS', 'HEx'): (['CE'], 1.0), ('LYS', 'HEy'): (['CE'], 1.0),
    ('LYS', 'HEx%'): (['CE'], 1.0), ('LYS', 'HEy%'): (['CE'], 1.0),
    ('LYS', 'HZ'): (['NZ'], 1.0), ('LYS', 'HZ%'): (['NZ'], 1.0), ('LYS', 'HZ*'): (['NZ'], 1.0),
    
    # MET
    ('MET', 'HG'): (['CG'], 1.0), ('MET', 'HG%'): (['CG'], 1.0), ('MET', 'HG*'): (['CG'], 1.0),
    ('MET', 'HG2'): (['CG'], 1.0), ('MET', 'HG3'): (['CG'], 1.0),
    ('MET', 'HGx'): (['CG'], 1.0), ('MET', 'HGy'): (['CG'], 1.0),
    ('MET', 'HGx%'): (['CG'], 1.0), ('MET', 'HGy%'): (['CG'], 1.0),
    ('MET', 'HE'): (['CE'], 1.0), ('MET', 'HE%'): (['CE'], 1.0), ('MET', 'HE*'): (['CE'], 1.0),
    
    # PHE
    ('PHE', 'HD%'): (['CD1', 'CD2'], 1.0), ('PHE', 'HD*'): (['CD1', 'CD2'], 1.0),
    ('PHE', 'HD1'): (['CD1'], 1.0), ('PHE', 'HD2'): (['CD2'], 1.0),
    ('PHE', 'HDx'): (['CD1', 'CD2'], 1.0), ('PHE', 'HDy'): (['CD1', 'CD2'], 1.0),
    ('PHE', 'HE%'): (['CE1', 'CE2'], 1.0), ('PHE', 'HE*'): (['CE1', 'CE2'], 1.0),
    ('PHE', 'HE1'): (['CE1'], 1.0), ('PHE', 'HE2'): (['CE2'], 1.0),
    ('PHE', 'HEx'): (['CE1', 'CE2'], 1.0), ('PHE', 'HEy'): (['CE1', 'CE2'], 1.0),
    ('PHE', 'HZ'): (['CZ'], 1.0),
    
    # PRO
    ('PRO', 'HG'): (['CG'], 1.0), ('PRO', 'HG%'): (['CG'], 1.0), ('PRO', 'HG*'): (['CG'], 1.0),
    ('PRO', 'HG2'): (['CG'], 1.0), ('PRO', 'HG3'): (['CG'], 1.0),
    ('PRO', 'HGx'): (['CG'], 1.0), ('PRO', 'HGy'): (['CG'], 1.0),
    ('PRO', 'HD'): (['CD'], 1.0), ('PRO', 'HD%'): (['CD'], 1.0), ('PRO', 'HD*'): (['CD'], 1.0),
    ('PRO', 'HD2'): (['CD'], 1.0), ('PRO', 'HD3'): (['CD'], 1.0),
    ('PRO', 'HDx'): (['CD'], 1.0), ('PRO', 'HDy'): (['CD'], 1.0),
    ('PRO', 'HDx%'): (['CD'], 1.0), ('PRO', 'HDy%'): (['CD'], 1.0),
    
    # SER
    ('SER', 'HG'): (['OG'], 1.0),
    
    # THR
    ('THR', 'HG1'): (['OG1'], 1.0),
    ('THR', 'HG2'): (['CG2'], 1.0), ('THR', 'HG2%'): (['CG2'], 1.0), ('THR', 'HG2*'): (['CG2'], 1.0),
    ('THR', 'HG1x'): (['CG2'], 1.0),
    ('THR', 'HG1y'): (['CG2'], 1.0),
    
    # TYR - CRITICAL FIX
    ('TYR', 'HD%'): (['CD1', 'CD2'], 1.0), ('TYR', 'HD*'): (['CD1', 'CD2'], 1.0),
    ('TYR', 'HD1'): (['CD1'], 1.0), ('TYR', 'HD2'): (['CD2'], 1.0),
    ('TYR', 'HDx'): (['CD1', 'CD2'], 1.0), ('TYR', 'HDy'): (['CD1', 'CD2'], 1.0),
    ('TYR', 'HE%'): (['CE1', 'CE2'], 1.0), ('TYR', 'HE*'): (['CE1', 'CE2'], 1.0),
    ('TYR', 'HE1'): (['CE1'], 1.0), ('TYR', 'HE2'): (['CE2'], 1.0),
    ('TYR', 'HEx'): (['CE1', 'CE2'], 1.0), ('TYR', 'HEy'): (['CE1', 'CE2'], 1.0),
    ('TYR', 'HH'): (['OH'], 1.0),
    
    # VAL
    ('VAL', 'HG1'): (['CG1'], 1.0), ('VAL', 'HG1%'): (['CG1'], 1.0), ('VAL', 'HG1*'): (['CG1'], 1.0),
    ('VAL', 'HG2'): (['CG2'], 1.0), ('VAL', 'HG2%'): (['CG2'], 1.0), ('VAL', 'HG2*'): (['CG2'], 1.0),
    ('VAL', 'HG%'): (['CG1', 'CG2'], 1.0), ('VAL', 'HG*'): (['CG1', 'CG2'], 1.0),
    ('VAL', 'HGx%'): (['CG1', 'CG2'], 1.0), ('VAL', 'HGy%'): (['CG1', 'CG2'], 1.0),
    ('VAL', 'HG1x'): (['CG1'], 1.0),
    ('VAL', 'HG1y'): (['CG1'], 1.0),
    ('VAL', 'HG11'): (['CG1'], 1.0),
    ('VAL', 'HG12'): (['CG1'], 1.0),
    ('VAL', 'HG13'): (['CG1'], 1.0),
}

# ============================================================================
# AMINO ACID CODE CONVERSION
# ============================================================================

def one_to_three_letter(aa: str) -> str:
    """Convert 1-letter to 3-letter amino acid code"""
    mapping = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
    }
    return mapping.get(aa.upper(), 'UNK')

# ============================================================================
# FASTA PARSER
# ============================================================================

def parse_fasta(fasta_file: str) -> Tuple[Dict[str, str], Dict[Tuple[int, str], str]]:
    """Parse FASTA file to extract sequence"""
    sequences = {}
    residue_names = {}
    current_chain = None
    current_seq = []

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('>'):
                # Save previous sequence
                if current_chain is not None and current_seq:
                    seq_str = ''.join(current_seq)
                    sequences[current_chain] = seq_str
                    
                    # Build residue name mapping
                    for pos, aa in enumerate(seq_str):
                        residue_names[(pos + 1, current_chain)] = one_to_three_letter(aa)
                
                # Parse header
                parts = line[1:].split('|')
                if len(parts) >= 2:
                    chain_part = parts[1].strip()
                    if chain_part.startswith('Chain '):
                        current_chain = chain_part.split()[-1]
                    else:
                        current_chain = 'A'
                else:
                    current_chain = 'A'
                
                current_seq = []
                
            elif line and not line.startswith(';'):
                current_seq.append(line)
    
    # Save last sequence
    if current_chain is not None and current_seq:
        seq_str = ''.join(current_seq)
        sequences[current_chain] = seq_str
        
        for pos, aa in enumerate(seq_str):
            residue_names[(pos + 1, current_chain)] = one_to_three_letter(aa)
    
    return sequences, residue_names

def build_sequence_map_from_fasta(fasta_file: str) -> Tuple[Dict[Tuple[int, str], int], Dict[Tuple[int, str], str]]:
    """Build residue mapping from FASTA"""
    sequences, residue_names = parse_fasta(fasta_file)
    
    residue_map = {}
    continuous_idx = 0
    
    for chain, sequence in sequences.items():
        print(f"Chain {chain}: {len(sequence)} residues")
        
        for nef_pos in range(1, len(sequence) + 1):
            residue_map[(nef_pos, chain)] = continuous_idx
            continuous_idx += 1
    
    print(f"Built sequence map: {continuous_idx} total residues")
    return residue_map, residue_names

# ============================================================================
# NEF PARSER
# ============================================================================

class NEFParser:
    """Simple NEF file parser with proper heavy atom mapping"""
    
    def __init__(self, nef_filepath: str, fasta_filepath: str):
        self.nef_filepath = nef_filepath
        self.fasta_filepath = fasta_filepath
        self.sequence = {}
        self.residue_names = {}
        self.distance_restraints = []
        self.dihedral_restraints = []
    
    def parse(self):
        """Parse FASTA for sequence, NEF for restraints"""
        # Parse sequence from FASTA
        self.sequence, self.residue_names = build_sequence_map_from_fasta(self.fasta_filepath)
        
        # Load NEF file
        with open(self.nef_filepath, 'r') as f:
            nef_content = f.read()
        
        # Parse restraints
        self._parse_distance_restraints(nef_content)
        self._parse_dihedral_restraints(nef_content)
        
        return self.sequence, self.distance_restraints, self.dihedral_restraints
    
    def _parse_distance_restraints(self, content: str):
        """Parse all distance restraint blocks"""
        pos = 0
        while True:
            # Find next distance restraint saveframe
            save_pos = content.find('save_XPLOR-NIH/CNS_distance_restraints', pos)
            if save_pos == -1:
                save_pos = content.find('_nef_distance_restraint_list.sf_framecode', pos)
                if save_pos == -1:
                    break
            
            # Find the loop within this saveframe
            loop_pos = content.find('_nef_distance_restraint.index', save_pos)
            if loop_pos == -1:
                pos = save_pos + 1
                continue
            
            # Extract data
            loop_data = self._extract_loop_data_fixed(content, loop_pos)
            
            print(f"Found {len(loop_data)} distance restraint rows")
            
            # Process restraints
            restraints_by_id = {}
            
            for row in loop_data:
                if len(row) < 13:
                    continue
                
                try:
                    restraint_id = int(row[1])
                    
                    seq1 = int(row[4])
                    chain1 = row[3].strip()
                    res1_name = row[5].strip()
                    atom1 = row[6].strip()
                    
                    seq2 = int(row[8])
                    chain2 = row[7].strip()
                    res2_name = row[9].strip()
                    atom2 = row[10].strip()
                    
                    target_value = float(row[12])
                    
                    # Map to continuous indices
                    if (seq1, chain1) not in self.sequence or (seq2, chain2) not in self.sequence:
                        continue
                    
                    res_i = self.sequence[(seq1, chain1)]
                    res_j = self.sequence[(seq2, chain2)]
                    
                    # Apply heavy atom mapping - THIS IS THE KEY FIX
                    mapped_restraints = self._map_to_heavy_atoms(
                        res_i, res1_name, atom1,
                        res_j, res2_name, atom2,
                        target_value
                    )
                    
                    if restraint_id not in restraints_by_id:
                        restraints_by_id[restraint_id] = []
                    
                    restraints_by_id[restraint_id].extend(mapped_restraints)
                    
                except (ValueError, IndexError, KeyError) as e:
                    continue
            
            # Convert to groups
            for group in restraints_by_id.values():
                if group:
                    self.distance_restraints.append(group)
            
            pos = loop_pos + 1
        
        print(f"Parsed {len(self.distance_restraints)} distance restraint groups")
    
    def _parse_dihedral_restraints(self, content: str):
        """Parse all dihedral restraint blocks"""
        pos = 0
        while True:
            # Find next dihedral restraint saveframe
            save_pos = content.find('save_XPLOR-NIH/CNS_dihedral_angle_restraints', pos)
            if save_pos == -1:
                save_pos = content.find('_nef_dihedral_restraint_list.sf_framecode', pos)
                if save_pos == -1:
                    break
            
            # Find the loop
            loop_pos = content.find('_nef_dihedral_restraint.index', save_pos)
            if loop_pos == -1:
                pos = save_pos + 1
                continue
            
            # Extract data
            loop_data = self._extract_loop_data_fixed(content, loop_pos)
            
            print(f"Found {len(loop_data)} dihedral restraint rows")
            
            # Process restraints
            restraints_by_id = {}
            
            for row in loop_data:
                if len(row) < 25:
                    continue
                
                try:
                    restraint_id = int(row[1])
                    
                    seq1 = int(row[4])
                    chain1 = row[3].strip()
                    atom1 = row[6].strip()
                    
                    seq2 = int(row[8])
                    chain2 = row[7].strip()
                    atom2 = row[10].strip()
                    
                    seq3 = int(row[12])
                    chain3 = row[11].strip()
                    atom3 = row[14].strip()
                    
                    seq4 = int(row[16])
                    chain4 = row[15].strip()
                    atom4 = row[18].strip()
                    
                    lower_limit = float(row[23])
                    upper_limit = float(row[24])
                    
                    # Map to continuous indices
                    try:
                        res1 = self.sequence[(seq1, chain1)]
                        res2 = self.sequence[(seq2, chain2)]
                        res3 = self.sequence[(seq3, chain3)]
                        res4 = self.sequence[(seq4, chain4)]
                    except KeyError:
                        continue
                    
                    restraint = TorsionRestraint(
                        res1=res1, atom1=atom1,
                        res2=res2, atom2=atom2,
                        res3=res3, atom3=atom3,
                        res4=res4, atom4=atom4,
                        phi_min=lower_limit,
                        phi_max=upper_limit
                    )
                    
                    if restraint_id not in restraints_by_id:
                        restraints_by_id[restraint_id] = []
                    restraints_by_id[restraint_id].append(restraint)
                    
                except (ValueError, IndexError, KeyError) as e:
                    continue
            
            for group in restraints_by_id.values():
                if group:
                    self.dihedral_restraints.append(group)
            
            pos = loop_pos + 1
        
        print(f"Parsed {len(self.dihedral_restraints)} dihedral restraint groups")
    
    def _extract_loop_data_fixed(self, content: str, column_start_pos: int) -> List[List[str]]:
        """Extract data from NEF loop"""
        # Find loop_ directive before the column start
        loop_start = content.rfind('loop_', max(0, column_start_pos - 2000), column_start_pos)
        if loop_start == -1:
            print("WARNING: No loop_ found before columns")
            return []
        
        # Find stop_ after loop_start
        stop_pos = content.find('stop_', loop_start)
        if stop_pos == -1:
            print("WARNING: No stop_ found")
            return []
        
        # Extract the entire loop section
        loop_section = content[loop_start:stop_pos]
        lines = loop_section.split('\n')
        
        # Count column definitions (lines starting with _)
        column_count = 0
        data_start_line = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if not stripped or stripped == 'loop_':
                continue
            
            if stripped.startswith('_'):
                column_count += 1
            else:
                # First non-column line is where data starts
                data_start_line = i
                break
        
        print(f"DEBUG: Found {column_count} columns, data starts at line {data_start_line}")
        
        if column_count == 0:
            return []
        
        # Parse data rows
        data_rows = []
        current_row = []
        
        for line in lines[data_start_line:]:
            stripped = line.strip()
            
            # Skip empty lines, comments, and stop
            if not stripped or stripped.startswith('#') or stripped.startswith('stop_'):
                continue
            
            # Split and accumulate
            parts = stripped.split()
            current_row.extend(parts)
            
            # When we have a complete row
            while len(current_row) >= column_count:
                data_rows.append(current_row[:column_count])
                current_row = current_row[column_count:]
        
        # Handle any remaining partial row
        if len(current_row) == column_count:
            data_rows.append(current_row)
        
        return data_rows
    
    def _map_to_heavy_atoms(
        self,
        res_i: int, res_i_name: str, atom_i: str,
        res_j: int, res_j_name: str, atom_j: str,
        distance: float
    ) -> List[DistanceRestraint]:
        """Map ambiguous hydrogens to heavy atoms - CRITICAL FIX"""
        
        # Map atom 1
        key1 = (res_i_name, atom_i)
        if key1 in map_to_heavy:
            heavy_atoms_1, correction_1 = map_to_heavy[key1]
        else:
            # If not in mapping, assume it's already a heavy atom
            heavy_atoms_1, correction_1 = [atom_i], 0.0
        
        # Map atom 2
        key2 = (res_j_name, atom_j)
        if key2 in map_to_heavy:
            heavy_atoms_2, correction_2 = map_to_heavy[key2]
        else:
            # If not in mapping, assume it's already a heavy atom
            heavy_atoms_2, correction_2 = [atom_j], 0.0
        
        # Create all combinations
        restraints = []
        for heavy1 in heavy_atoms_1:
            for heavy2 in heavy_atoms_2:
                corrected_dist = distance + correction_1 + correction_2
                
                restraint = DistanceRestraint(
                    res_i=res_i,
                    atom_i=heavy1,
                    res_j=res_j,
                    atom_j=heavy2,
                    distance=corrected_dist
                )
                restraints.append(restraint)
        
        return restraints

# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def write_distance_restraints_split(
    groups: List[List[DistanceRestraint]],
    local_cutoff: int = 8
) -> Tuple[str, str]:
    """Split distance restraints into local and global"""
    global_str = ""
    local_str = ""
    
    for group in groups:
        group_lines = []
        is_local = False
        
        for r in group:
            # Output in 1-based indexing for MELD
            line = f"{r.res_i + 1} {r.atom_i} {r.res_j + 1} {r.atom_j} {r.distance:.3f}"
            group_lines.append(line)
            
            if abs(r.res_i - r.res_j) < local_cutoff:
                is_local = True
        
        group_text = "\n".join(group_lines) + "\n\n"
        
        if is_local:
            local_str += group_text
        else:
            global_str += group_text
    
    return global_str, local_str

def write_torsion_restraints(groups: List[List[TorsionRestraint]]) -> str:
    """Write torsion restraints in MELD format"""
    output = ""
    
    for group in groups:
        for r in group:
            line = (f"{r.res1 + 1} {r.atom1} {r.res2 + 1} {r.atom2} "
                    f"{r.res3 + 1} {r.atom3} {r.res4 + 1} {r.atom4} "
                    f"{r.phi_min:.2f} {r.phi_max:.2f}")
            output += line + "\n"
        output += "\n"
    
    return output

# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert NEF to MELD format for SimpleFold"
    )
    parser.add_argument('--nef', type=str, required=True,
                        help='Input NEF file')
    parser.add_argument('--fasta', type=str, required=True,
                        help='Input FASTA file')
    parser.add_argument('-d', '--directory', type=str, default='.',
                        help='Output directory')
    parser.add_argument('--local_cutoff', type=int, default=8,
                        help='Sequence separation cutoff')
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.directory, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"NEF to MELD Converter (FASTA Sequence Input)")
    print(f"{'='*60}\n")
    
    if not os.path.exists(args.fasta):
        print(f"ERROR: FASTA file not found: {args.fasta}")
        return
    
    if not os.path.exists(args.nef):
        print(f"ERROR: NEF file not found: {args.nef}")
        return
    
    print(f"Loading sequence from: {args.fasta}")
    print(f"Loading restraints from: {args.nef}")
    
    parser = NEFParser(args.nef, args.fasta)
    sequence, distance_groups, torsion_groups = parser.parse()
    
    if not sequence:
        print("ERROR: No sequence information")
        return
    
    # Split restraints
    print(f"\nSplitting restraints (cutoff={args.local_cutoff})...")
    global_noe, local_noe = write_distance_restraints_split(distance_groups, args.local_cutoff)
    rotamers_str = write_torsion_restraints(torsion_groups)
    
    # Write files
    print("\n" + "="*60)
    print("Writing MELD Files")
    print("="*60)
    
    files_written = {
        "NOE_global.dat": global_noe,
        "NOE_local.dat": local_noe,
        "NOE_local_and_global.dat": global_noe + local_noe,
        "rotamers.dat": rotamers_str
    }
    
    for filename, content in files_written.items():
        filepath = os.path.join(args.directory, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ {filepath} ({len(content)} bytes)")
    
    # Summary
    print("\n" + "="*60)
    print("Conversion Complete")
    print("="*60)
    print(f"Total residues: {len(sequence)}")
    print(f"Distance restraint groups: {len(distance_groups)}")
    
    n_global = global_noe.count('\n\n')
    n_local = local_noe.count('\n\n')
    print(f"  - Global: {n_global}")
    print(f"  - Local: {n_local}")
    
    print(f"Torsion restraint groups: {len(torsion_groups)}")
    print("\n" + "="*60)
    print("Usage with SimpleFold")
    print("="*60)
    print(f"simplefold --fasta protein.fasta \\")
    print(f"           --noe_file {os.path.join(args.directory, 'NOE_global.dat')} \\")
    print(f"           --talos_file {os.path.join(args.directory, 'rotamers.dat')} \\")
    print(f"           --nmr_guidance_scale 1.5")
    print()

if __name__ == '__main__':
    main()