import starfile
import pandas as pd
import numpy as np

def _parse_atom_selection(selection_str):
    """
    Parses an atom selection string into a list of atom identifiers.
    """
    # This is a simplified parser. A more robust implementation would handle a wider range of selection syntaxes.
    parts = selection_str.split()
    residue_number = int(parts[0])
    atom_name = parts[2]
    return (residue_number, atom_name)

def parse_nef(file_path, atom_to_idx):
    """
    Parses a NEF file to extract distance restraints.
    """
    nef_data = starfile.read(file_path)
    restraints = []
    for _, row in nef_data['distance_restraints'].iterrows():
        try:
            selections1 = [_parse_atom_selection(sel) for sel in row['Atom_selection_1'].split('OR')]
            selections2 = [_parse_atom_selection(sel) for sel in row['Atom_selection_2'].split('OR')]

            indices1 = [atom_to_idx[sel] for sel in selections1 if sel in atom_to_idx]
            indices2 = [atom_to_idx[sel] for sel in selections2 if sel in atom_to_idx]

            if indices1 and indices2:
                restraint = {
                    'id': row['ID'],
                    'indices1': indices1,
                    'indices2': indices2,
                    'upper_bound': row['Upper_distance_threshold']
                }
                restraints.append(restraint)
        except Exception as e:
            print(f"Skipping restraint due to parsing error: {e}")
            continue
    return restraints
