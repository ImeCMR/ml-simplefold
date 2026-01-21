#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from simplefold.utils.residue_constants import van_der_waals_radius, atom_types, load_stereo_chemical_props, restype_1to3, atom_order

def generate_covalent_blueprint(sequence):
    """
    Generates bond indices and ideal lengths for a given sequence.
    """
    residue_bonds, _, _ = load_stereo_chemical_props()

    at1_idx = []
    at2_idx = []
    lengths = []

    # Split sequence by chain (assuming ':' as separator)
    chains = sequence.split(':')
    offset = 0
    for chain in chains:
        for i, res_letter in enumerate(chain):
            res_name = restype_1to3.get(res_letter, "UNK")
            bonds = residue_bonds.get(res_name, [])

            # Internal bonds
            for b in bonds:
                if b.atom1_name in atom_order and b.atom2_name in atom_order:
                    at1_idx.append(offset * 37 + atom_order[b.atom1_name])
                    at2_idx.append(offset * 37 + atom_order[b.atom2_name])
                    lengths.append(b.length)

            # Peptide bond to next residue
            if i < len(chain) - 1:
                # C from current to N of next
                at1_idx.append(offset * 37 + atom_order["C"])
                at2_idx.append((offset + 1) * 37 + atom_order["N"])
                lengths.append(1.329) # Avg peptide bond

            offset += 1

    return {
        "bond_at1_idx": torch.tensor(at1_idx),
        "bond_at2_idx": torch.tensor(at2_idx),
        "bond_lengths": torch.tensor(lengths),
        "bond_mask": torch.ones(len(lengths))
    }

class GeomEnergy(nn.Module):
    """
    Geometric energy term to ensure structural plausibility.
    Includes Clash (excluded volume) and Covalent (bond lengths) penalties.
    """
    def __init__(self, clash_weight=1.0, covalent_weight=1.0, overlap_tolerance=0.8):
        super().__init__()
        self.clash_weight = clash_weight
        self.covalent_weight = covalent_weight
        self.overlap_tolerance = overlap_tolerance

        # Initialize VdW radii tensor
        self.vdw_radii = torch.ones(len(atom_types)) * 1.5 # Default 1.5A
        for i, atom_name in enumerate(atom_types):
            element = atom_name[0]
            if element in van_der_waals_radius:
                self.vdw_radii[i] = van_der_waals_radius[element]

    def compute_clash_energy(self, coords, batch, cutoff=10.0):
        """
        Penalty for overlapping non-bonded atoms.
        O(N) scalability using a distance cutoff.
        """
        B, N, _ = coords.shape
        atom_mask = batch["atom_pad_mask"] # [B, N]
        atom_types_idx = batch["atom_types"] # [B, N]

        # Get VdW radii for all atoms in batch
        radii = self.vdw_radii.to(coords.device)[atom_types_idx] # [B, N]

        # pairwise sum of radii
        sum_radii = radii.unsqueeze(1) + radii.unsqueeze(2) # [B, N, N]
        threshold = sum_radii * self.overlap_tolerance

        # Pairwise distances with cutoff for efficiency
        # We only compute energy for pairs within the cutoff
        # Note: torch.cdist is still O(N^2) memory but we can mask it.
        # For true O(N) we would need neighbor lists.
        dist_mat = torch.cdist(coords, coords) # [B, N, N]

        # Mask: padding, self, and cutoff
        mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        eye = torch.eye(N, device=coords.device).unsqueeze(0)
        mask = mask * (1.0 - eye)
        mask = mask * (dist_mat < cutoff).float()

        clash_violation = F.relu(threshold - dist_mat)
        clash_energy = torch.sum(torch.pow(clash_violation, 2) * mask, dim=(1, 2)) / 2.0

        return clash_energy

    def compute_covalent_energy(self, coords, batch):
        """
        Penalty for deviations in bond lengths.
        Requires connectivity information in batch.
        """
        if "bond_at1_idx" not in batch:
            return torch.zeros(coords.shape[0], device=coords.device)

        B, N, _ = coords.shape
        at1_idx = batch["bond_at1_idx"] # [B, L]
        at2_idx = batch["bond_at2_idx"] # [B, L]
        ideal_dists = batch["bond_lengths"] # [B, L]
        bond_mask = batch["bond_mask"] # [B, L]

        batch_offset = torch.arange(B, device=coords.device).view(B, 1) * N
        at1_idx_offset = (at1_idx + batch_offset).view(-1)
        at2_idx_offset = (at2_idx + batch_offset).view(-1)

        pos1 = coords.view(-1, 3)[at1_idx_offset].view(B, -1, 3)
        pos2 = coords.view(-1, 3)[at2_idx_offset].view(B, -1, 3)

        dists = torch.norm(pos1 - pos2, dim=-1)
        bond_penalty = torch.pow(dists - ideal_dists, 2) * bond_mask

        return torch.sum(bond_penalty, dim=-1)

    def forward(self, coords, batch):
        clash_e = self.compute_clash_energy(coords, batch)
        covalent_e = self.compute_covalent_energy(coords, batch)

        return self.clash_weight * clash_e + self.covalent_weight * covalent_e
