import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Optional, Tuple

def parse_nef_restraints(nef_path: str) -> pd.DataFrame:
    """
    NEF parser to extract distance restraints, supporting ambiguous assignments.
    """
    with open(nef_path, 'r') as f:
        lines = f.readlines()

    restraints = []
    in_loop = False
    headers = []

    current_restraint_id = 0

    for line in lines:
        line = line.strip()
        if line.startswith('_distance_restraint.'):
            headers.append(line.split('.')[-1])
            in_loop = True
            continue

        if in_loop:
            if line.startswith('loop_') or line.startswith('stop_') or line.startswith('save_') or not line:
                if line.startswith('loop_'): continue
                if not line: continue
                if not (line[0].isdigit() or line[0] in "'.\""):
                    in_loop = False
                    continue

            parts = line.split()
            if len(parts) == len(headers):
                # We'll use a simple way to group ambiguous restraints if the NEF doesn't provide a restraint_id
                # In many NEF files, multiple lines with same residue/atom indices but different distances
                # or explicit restraint IDs are used.
                restraints.append(parts)

    df = pd.DataFrame(restraints, columns=headers)

    # Filter out invalid entries
    for col in ['residue_index_1', 'residue_index_2', 'distance_upper_bound']:
        if col in df.columns:
            df = df[df[col] != '.']

    return df

def process_nef_restraints(
    record_id: str,
    data_dir: str,
) -> Optional[Dict]:
    """
    Processes NEF restraints for a given record, handling ambiguity.
    """
    nef_search_paths = [
        os.path.join(data_dir, "nef", f"{record_id}.nef"),
        os.path.join(data_dir, "structures", f"{record_id}.nef"),
        os.path.join(data_dir, f"{record_id}.nef"),
    ]

    nef_path = None
    for path in nef_search_paths:
        if os.path.exists(path):
            nef_path = path
            break

    if nef_path is None: return None

    df = parse_nef_restraints(nef_path)
    if df.empty: return None

    try:
        # We need to group by restraint if ambiguous
        # If 'restraint_id' is present, use it. Otherwise, assume each line is a separate restraint
        # unless they share indices and are intended to be ambiguous.
        # For simplicity, if 'restraint_id' is missing, we treat each line as a restraint.

        if 'id' in df.columns:
            id_col = 'id'
        elif 'restraint_id' in df.columns:
            id_col = 'restraint_id'
        else:
            df['temp_id'] = range(len(df))
            id_col = 'temp_id'

        grouped = df.groupby(id_col)

        res1_list = []
        res2_list = []
        atom1_list = []
        atom2_list = []
        upper_bounds = []

        for name, group in grouped:
            res1_list.append((group['residue_index_1'].astype(int).values - 1).tolist())
            res2_list.append((group['residue_index_2'].astype(int).values - 1).tolist())
            atom1_list.append(group['atom_name_1'].values.tolist() if 'atom_name_1' in group.columns else ['CA'] * len(group))
            atom2_list.append(group['atom_name_2'].values.tolist() if 'atom_name_2' in group.columns else ['CA'] * len(group))
            upper_bounds.append(group['distance_upper_bound'].astype(float).mean()) # usually upper bound is same for all members

        return {
            'res1': res1_list, # List of lists (ambiguous)
            'res2': res2_list,
            'atom1': atom1_list,
            'atom2': atom2_list,
            'upper_bounds': np.array(upper_bounds),
        }
    except Exception as e:
        print(f"Error processing NEF {nef_path}: {e}")
        return None
