#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nef_utils import parse_restraint_assignments

class BayesianSteering(nn.Module):
    def __init__(self, enoe_weight=1.0, egeom_weight=1.0, time_dependent_variance=True,
                 ideal_bond_lengths=None, ideal_bond_angles=None, vdw_radii=None):
        super().__init__()
        self.enoe_weight = enoe_weight
        self.egeom_weight = egeom_weight
        self.time_dependent_variance = time_dependent_variance
        self.ideal_bond_lengths = ideal_bond_lengths
        self.ideal_bond_angles = ideal_bond_angles
        self.vdw_radii = vdw_radii

    def forward(self, xt, restraints, t, bonds, atom_to_idx, atom_names):
        enoe = self.calculate_enoe(xt, restraints, t, atom_to_idx)
        egeom = self.calculate_egeom(xt, bonds, atom_names)
        e_bayesian = self.enoe_weight * enoe + self.egeom_weight * egeom

        if e_bayesian.requires_grad:
            f_steering = -torch.autograd.grad(e_bayesian.sum(), xt, retain_graph=True)[0]
        else:
            f_steering = torch.zeros_like(xt)

        return f_steering, e_bayesian

    def calculate_enoe(self, xt, restraints, t, atom_to_idx):
        if not restraints or not any(restraints):
            return torch.tensor(0.0, device=xt.device, requires_grad=True)

        batch_size = xt.shape[0]
        device = xt.device

        # Unpack restraints into tensors for vectorized processing
        atom1_indices = []
        atom2_indices = []
        upper_bounds = []
        batch_indices = []
        restraint_ids = []

        for batch_idx in range(batch_size):
            if not restraints[batch_idx]: # Propper check for None/Empty
                continue

            atom_to_idx_batch = atom_to_idx[batch_idx]
            for restraint in restraints[batch_idx]:
                # Handle both single and multi-assignment restraints
                assignments = parse_restraint_assignments(restraint)

                for assignment in assignments:
                    res1 = assignment['atom1_residue_number']
                    res2 = assignment['atom2_residue_number']
                    atom1 = assignment['atom1_atom_name']
                    atom2 = assignment['atom2_atom_name']

                    # Validate indices exist
                    if atom1 not in atom_to_idx_batch or atom2 not in atom_to_idx_batch:
                        continue

                    atom1_indices.append(atom_to_idx_batch[atom1])
                    atom2_indices.append(atom_to_idx_batch[atom2])
                    upper_bounds.append(assignment['upper_bound'])
                    batch_indices.append(batch_idx)
                    # Use unique ID combining batch and restraint ID
                    restraint_ids.append(f"batch_{batch_idx}_id_{restraint['id']}")

        if not atom1_indices:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        atom1_indices = torch.tensor(atom1_indices, device=device, dtype=torch.long)
        atom2_indices = torch.tensor(atom2_indices, device=device, dtype=torch.long)
        upper_bounds = torch.tensor(upper_bounds, device=device, dtype=torch.float3)
        batch_indices = torch.tensor(batch_indices, device=device, dtype=torch.long)

        atom1_coords = xt[batch_indices, atom1_indices, :]
        atom2_coords = xt[batch_indices, atom2_indices, :]

        dist_sq = ((atom1_coords - atom2_coords) ** 2).sum(dim=-1)
        dist_minus_6 = dist_sq.pow(-3)

        # Create proper ID mapping
        unique_str_ids = sorted(list(set(restraint_ids)))
        id_map = {str_id: idx for idx, str_id in enumerate(unique_str_ids)}
        int_restraint_ids = torch.tensor(
            [id_map[rid] for rid in restraint_ids],
            device=device,
            dtype=torch.long
        )

        # Handle case where no unique restraints 
        num_unique_restraints = len(unique_str_ids)
        if num_unique_restraints == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Sum r^-6 distances for each restraint
        summed_dist_minus_6 = torch.zeros(num_unique_restraints, device=device)
        summed_dist_minus_6.scatter_add_(0, int_restraint_ids, dist_minus_6)

        # Compute effective distance
        effective_dist = summed_dist_minus_6.pow(-1.0/6.0)

        # Create propper upper bound mapping
        id_to_upper_bound = {}
        restraint_idx = 0
        for batch_idx in range(batch_size):
            if not restraints[batch_idx]:
                continue
            for restraint in restraints[batch_idx]:
                key = f"batch_{batch_idx}_id_{restraint['id']}"
                # Use mean of all assignments' upper bounds
                assignments = parse_restraint_assignments(restraint)
                if assignments:
                    ub = sum(a['upper_bound'] for a in assignments) / len(assignments)
                    id_to_upper_bound[key] = ub

        unique_upper_bounds = torch.tensor(
            [id_to_upper_bound[str_id] for str_id in unique_str_ids],
            device=device,
            dtype=torch.float32
        )

        # Compute violations with time-dependent variance
        violations = F.relu(effective_dist - unique_upper_bounds)
        variance = self.get_variance(t)
        enoe = 0.5 * (violations ** 2) / variance

        return enoe.sum()

    def get_variance(self, t):
        if self.time_dependent_variance:
            # Loosely enforce constraints early in the process, and rigorously as t -> 1
            return 1.0 - t + 1e-6
        else:
            return 1.0

    def calculate_egeom(self, xt, bonds, atom_names):
        batch_size = xt.shape[0]
        total_e_geom = 0.0

        for i in range(batch_size):
            total_e_geom += self.calculate_clash_penalty(xt[i], atom_names[i])
            total_e_geom += self.calculate_covalent_penalty(xt[i], bonds[i], atom_names[i])
            total_e_geom += self.calculate_bond_angle_penalty(xt[i], bonds[i], atom_names[i])

        return total_e_geom / batch_size

    def calculate_bond_angle_penalty(self, x, bonds, atom_names):
        if self.ideal_bond_angles is None or not bonds:
            return 0.0

        bond_angle_penalty = 0.0

        adj = {i: [] for i in range(len(atom_names))}
        for i, j in bonds:
            adj[i].append(j)
            adj[j].append(i)

        for i in range(len(atom_names)):
            for j in adj[i]:
                for k in adj[j]:
                    if i == k:
                        continue

                    atom_i_name = atom_names[i].split('_')[0]
                    atom_j_name = atom_names[j].split('_')[0]
                    atom_k_name = atom_names[k].split('_')[0]

                    outer_atoms = sorted([atom_i_name, atom_k_name])
                    angle_type = f"{outer_atoms[0]}-{atom_j_name}-{outer_atoms[1]}"
                    ideal_angle = self.ideal_bond_angles.get(angle_type)

                    if ideal_angle:
                        v1 = x[i, :] - x[j, :]
                        v2 = x[k, :] - x[j, :]

                        cos_angle = (v1 * v2).sum(dim=-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1))
                        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

                        bond_angle_penalty += (angle - torch.deg2rad(torch.tensor(ideal_angle, device=x.device))).pow(2)

        return bond_angle_penalty

    def calculate_clash_penalty(self, x, atom_names):
        if self.vdw_radii is None:
            return 0.0

        dist_matrix = torch.cdist(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
        vdw_radii = torch.tensor([self.vdw_radii.get(name.split('_')[0], 1.7) for name in atom_names], device=x.device)
        vdw_sum = vdw_radii.unsqueeze(0) + vdw_radii.unsqueeze(1)

        sequence_dist = torch.abs(torch.arange(x.shape[0]).unsqueeze(0) - torch.arange(x.shape[0]).unsqueeze(1))
        non_bonded_mask = (sequence_dist > 2).float().to(x.device)

        clash_violations = F.relu(vdw_sum - dist_matrix) * non_bonded_mask
        return clash_violations.pow(2).sum()

    def calculate_covalent_penalty(self, x, bonds, atom_names):
        if self.ideal_bond_lengths is None or not bonds:
            return 0.0

        covalent_penalty = 0.0
        for bond in bonds:
            atom1_idx, atom2_idx = bond
            atom1_name = atom_names[atom1_idx]
            atom2_name = atom_names[atom2_idx]

            bond_type = "-".join(sorted([atom1_name.split('_')[0], atom2_name.split('_')[0]]))
            ideal_length = self.ideal_bond_lengths.get(bond_type, 1.5)

            atom1_coords = x[atom1_idx, :]
            atom2_coords = x[atom2_idx, :]

            bond_length = torch.norm(atom1_coords - atom2_coords, dim=-1)
            covalent_penalty += (bond_length - ideal_length).pow(2)

        return covalent_penalty
