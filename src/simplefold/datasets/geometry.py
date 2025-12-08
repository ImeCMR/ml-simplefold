import numpy as np

def get_covalent_bonds(tokens, atom_to_idx):
    """
    Generates a list of covalent bonds for a given sequence.
    """
    bonds = []
    for res_idx in sorted(list(set(token[0] for token in tokens))):
        for atom1_name, atom2_name in [('N', 'CA'), ('CA', 'C'), ('C', 'O')]:
            atom1 = (res_idx, atom1_name)
            atom2 = (res_idx, atom2_name)
            if atom1 in atom_to_idx and atom2 in atom_to_idx:
                bonds.append((atom_to_idx[atom1], atom_to_idx[atom2]))

    for i in range(len(tokens) - 1):
        res1_idx, _, _, _, _, _, _, _ = tokens[i]
        res2_idx, _, _, _, _, _, _, _ = tokens[i+1]

        if res1_idx != res2_idx:
            # Peptide bond
            atom1 = (res1_idx, 'C')
            atom2 = (res2_idx, 'N')
            if atom1 in atom_to_idx and atom2 in atom_to_idx:
                bonds.append((atom_to_idx[atom1], atom_to_idx[atom2]))
    return bonds

def get_bond_angles(tokens, atom_to_idx):
    """
    Generates a list of bond angles for a given sequence.
    """
    angles = []
    for i in range(len(tokens)):
        res_idx, _, _, atom_name, _, _, _, _ = tokens[i]

        # Backbone angles
        atom1 = (res_idx, 'N')
        atom2 = (res_idx, 'CA')
        atom3 = (res_idx, 'C')
        if atom1 in atom_to_idx and atom2 in atom_to_idx and atom3 in atom_to_idx:
            angles.append((atom_to_idx[atom1], atom_to_idx[atom2], atom_to_idx[atom3]))

        if i < len(tokens) - 1:
            res2_idx, _, _, _, _, _, _, _ = tokens[i+1]
            if res_idx != res2_idx:
                atom1 = (res_idx, 'CA')
                atom2 = (res_idx, 'C')
                atom3 = (res2_idx, 'N')
                if atom1 in atom_to_idx and atom2 in atom_to_idx and atom3 in atom_to_idx:
                    angles.append((atom_to_idx[atom1], atom_to_idx[atom2], atom_to_idx[atom3]))
    return angles
