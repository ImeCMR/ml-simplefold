#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
from pathlib import Path
from typing import Dict, Any, Optional
from nef.parser import NEFParser

def process_nef_restraints(record_id: str, dataset_target_dir: Path, tokenized: Any, features: Dict[str, Any]) -> Dict[str, Any]:
    """Load and process NEF restraints for the given record."""
    # Try different possible locations for the NEF file
    nef_locations = [
        Path(dataset_target_dir) / "nef" / f"{record_id}.nef",
        Path(dataset_target_dir) / "structures" / f"{record_id}.nef",
        Path(dataset_target_dir) / f"{record_id}.nef",
    ]

    nef_path = None
    for loc in nef_locations:
        if loc.exists():
            nef_path = loc
            break

    if nef_path is None:
        return features

    try:
        parser = NEFParser(nef_path)
        restraints = parser.get_restraints()
        if not restraints:
            return features

        ambiguous_groups = parser.get_ambiguous_restraints()

        # Build atom map from tokenized structure: (chain, res, name) -> idx
        atom_map = {}
        curr_atom_idx = 0
        for token in tokenized.tokens:
            chain_idx = token["asym_id"]
            res_id = token["res_idx"]
            # Get chain name, handle potential bytes or different structure
            chain_obj = tokenized.structure.chains[chain_idx]
            chain_code = chain_obj.get("name", "A")
            if isinstance(chain_code, bytes):
                chain_code = chain_code.decode()

            num_atoms = token["atom_num"]
            start = token["atom_idx"]
            token_atoms = tokenized.structure.atoms[start:start+num_atoms]

            for i in range(num_atoms):
                name_bytes = token_atoms[i]["name"]
                name = "".join([chr(c + 32) for c in name_bytes if c != 0])
                atom_map[(str(chain_code), int(res_id), name)] = curr_atom_idx + i
            curr_atom_idx += num_atoms

        # Group restraints (handling ambiguity)
        groups = []
        used_ambiguous_ids = set()

        for r in restraints:
            if r.restraint_id is not None:
                if r.restraint_id in used_ambiguous_ids:
                    continue
                groups.append(ambiguous_groups.get(r.restraint_id, [r]))
                used_ambiguous_ids.add(r.restraint_id)
            else:
                groups.append([r])

        if not groups:
            return features

        # Convert to tensors for batching
        max_assignments = max(len(g) for g in groups)
        num_restraints = len(groups)

        at1_idx = torch.zeros((num_restraints, max_assignments), dtype=torch.long)
        at2_idx = torch.zeros((num_restraints, max_assignments), dtype=torch.long)
        noe_mask = torch.zeros((num_restraints, max_assignments), dtype=torch.float)
        upper_bounds = torch.zeros(num_restraints, dtype=torch.float)
        weights = torch.zeros(num_restraints, dtype=torch.float)

        valid_restraints_count = 0
        for i, group in enumerate(groups):
            upper_bounds[valid_restraints_count] = group[0].upper_bound
            weights[valid_restraints_count] = group[0].weight

            any_valid_assignment = False
            for j, r in enumerate(group):
                # Try to find both atoms in our map
                idx1 = atom_map.get((str(r.atom1.chain_code), int(r.atom1.sequence_code), r.atom1.atom_name))
                idx2 = atom_map.get((str(r.atom2.chain_code), int(r.atom2.sequence_code), r.atom2.atom_name))

                if idx1 is not None and idx2 is not None:
                    at1_idx[valid_restraints_count, j] = idx1
                    at2_idx[valid_restraints_count, j] = idx2
                    noe_mask[valid_restraints_count, j] = 1.0
                    any_valid_assignment = True

            if any_valid_assignment:
                valid_restraints_count += 1

        if valid_restraints_count == 0:
            return features

        # Trim tensors to valid restraints
        features["noe_at1_idx"] = at1_idx[:valid_restraints_count]
        features["noe_at2_idx"] = at2_idx[:valid_restraints_count]
        features["noe_mask"] = noe_mask[:valid_restraints_count]
        features["noe_upper_bounds"] = upper_bounds[:valid_restraints_count]
        features["noe_weights"] = weights[:valid_restraints_count]
        features["noe_restraints_indices"] = torch.tensor(True)

    except Exception as e:
        print(f"Warning: Failed to process NEF for {record_id}: {e}")

    return features
