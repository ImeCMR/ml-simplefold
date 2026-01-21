#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
import numpy as np
from simplefold.utils.residue_constants import atom_order

def process_nef_restraints(restraints, sequence, atom_names):
    """
    Maps NEF restraints to atom indices.

    Args:
        restraints: List of dictionaries from NEFParser
        sequence: The protein sequence (to verify residues)
        atom_names: List of atom names for each residue in the model's format [N, 37]

    Returns:
        Processed restraints ready for collation:
        - at1_idx: [M, K]
        - at2_idx: [M, K]
        - mask: [M, K]
        - upper_bounds: [M]
    """
    # Group by restraint ID if present, or treat each row as a separate restraint
    # In NEF, restraints with the same 'restraint_id' or consecutive rows with same 'id'
    # might represent ambiguity.

    grouped = {}
    for r in restraints:
        rid = r.get('restraint_id', r.get('index', len(grouped)))
        if rid not in grouped:
            grouped[rid] = []
        grouped[rid].append(r)

    processed = []
    for rid, group in grouped.items():
        pairs = []
        ub = None
        for r in group:
            try:
                # NEF uses 1-based indexing for residues
                res1_idx = int(r['resid_1']) - 1
                res2_idx = int(r['resid_2']) - 1
                atom1_name = r['atom_id_1']
                atom2_name = r['atom_id_2']
                ub = float(r['distance_upper_bound'])

                # Map atom name to model index [0..36]
                if atom1_name not in atom_order or atom2_name not in atom_order:
                    continue

                a1_type_idx = atom_order[atom1_name]
                a2_type_idx = atom_order[atom2_name]

                # Global index in [N_res * 37]
                idx1 = res1_idx * 37 + a1_type_idx
                idx2 = res2_idx * 37 + a2_type_idx

                pairs.append((idx1, idx2))
            except (KeyError, ValueError, TypeError):
                continue

        if pairs and ub is not None:
            processed.append({
                'pairs': pairs,
                'upper_bound': ub
            })

    if not processed:
        return None

    # Find max ambiguity (K)
    max_k = max(len(p['pairs']) for p in processed)
    M = len(processed)

    at1_idx = np.zeros((M, max_k), dtype=np.int64)
    at2_idx = np.zeros((M, max_k), dtype=np.int64)
    mask = np.zeros((M, max_k), dtype=np.float32)
    upper_bounds = np.zeros(M, dtype=np.float32)

    for i, p in enumerate(processed):
        upper_bounds[i] = p['upper_bound']
        for k, (idx1, idx2) in enumerate(p['pairs']):
            at1_idx[i, k] = idx1
            at2_idx[i, k] = idx2
            mask[i, k] = 1.0

    return {
        'noe_at1_idx': at1_idx,
        'noe_at2_idx': at2_idx,
        'noe_mask': mask,
        'noe_upper_bounds': upper_bounds
    }
