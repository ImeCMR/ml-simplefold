
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ENOE(nn.Module):
    def __init__(self, sigma_t_schedule):
        super(ENOE, self).__init__()
        self.sigma_t_schedule = sigma_t_schedule

    def forward(self, pred_coords, nef_data, t):
        total_noe_loss = 0.0
        for restraint in nef_data:
            d_ub = restraint['d_ub']
            assignments = restraint['assignments']

            # Calculate r-6 summed effective distance
            d_eff_inv_6_sum = 0.0
            for assign in assignments:
                idx1, idx2 = assign['idx1'], assign['idx2']
                dist_sq = torch.sum((pred_coords[:, idx1, :] - pred_coords[:, idx2, :]) ** 2, dim=-1)
                dist = torch.sqrt(dist_sq)
                d_eff_inv_6_sum += dist.pow(-6)

            d_eff = d_eff_inv_6_sum.pow(-1/6)

            # Calculate violation (only for distances greater than the upper bound)
            violation = F.relu(d_eff - d_ub)

            # Time-dependent variance
            sigma_t = self.get_sigma_t(t)

            # Quadratic potential
            noe_loss = (violation**2) / (2 * sigma_t**2)
            total_noe_loss += noe_loss.mean()

        return total_noe_loss

    def get_sigma_t(self, t):
        start, end = self.sigma_t_schedule
        return start + (end - start) * (1 - t)

class EGeom(nn.Module):
    def __init__(self, ideal_bond_lengths, ideal_bond_angles, vdw_radii):
        super(EGeom, self).__init__()
        self.ideal_bond_lengths = ideal_bond_lengths
        self.ideal_bond_angles = ideal_bond_angles
        self.vdw_radii = vdw_radii

    def forward(self, pred_coords, batch):
        clash_loss = self.calculate_clash_loss(pred_coords, batch)
        covalent_loss = self.calculate_covalent_loss(pred_coords, batch)
        return clash_loss + covalent_loss

    def calculate_clash_loss(self, pred_coords, batch):
        # A simple clash loss based on VDW radii
        atom_types = batch.get('atom_types', [])
        if not atom_types:
            return 0.0

        dist_matrix = torch.cdist(pred_coords, pred_coords)
        vdw_radii_tensor = torch.tensor([self.vdw_radii.get(at, 1.5) for at in atom_types], device=pred_coords.device)
        vdw_sum_matrix = vdw_radii_tensor.unsqueeze(1) + vdw_radii_tensor.unsqueeze(0)

        clash_loss = F.relu(vdw_sum_matrix - dist_matrix).pow(2).sum()
        return clash_loss

    def calculate_covalent_loss(self, pred_coords, batch):
        # Covalent loss based on ideal bond lengths and angles
        bond_indices = torch.tensor(batch.get('bond_indices', []), device=pred_coords.device, dtype=torch.long)
        angle_indices = torch.tensor(batch.get('angle_indices', []), device=pred_coords.device, dtype=torch.long)
        bond_types = batch.get('bond_types', [])
        angle_types = batch.get('angle_types', [])

        covalent_loss = 0.0

        if bond_indices.numel() > 0:
            # Bond length loss
            ideal_lengths = torch.tensor([self.ideal_bond_lengths.get(bt, 1.5) for bt in bond_types], device=pred_coords.device)
            p1 = torch.index_select(pred_coords, 1, bond_indices[:, 0])
            p2 = torch.index_select(pred_coords, 1, bond_indices[:, 1])
            dists = torch.norm(p1 - p2, dim=-1)
            covalent_loss += F.mse_loss(dists, ideal_lengths.unsqueeze(0).expand_as(dists))

        if angle_indices.numel() > 0:
            # Bond angle loss
            ideal_angles = torch.tensor([np.deg2rad(self.ideal_bond_angles.get(at, 120.0)) for at in angle_types], device=pred_coords.device)
            p1 = torch.index_select(pred_coords, 1, angle_indices[:, 0])
            p2 = torch.index_select(pred_coords, 1, angle_indices[:, 1])
            p3 = torch.index_select(pred_coords, 1, angle_indices[:, 2])
            v1 = p1 - p2
            v2 = p3 - p2
            angles = torch.acos(torch.sum(v1 * v2, dim=-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1)))
            covalent_loss += F.mse_loss(angles, ideal_angles.unsqueeze(0).expand_as(angles))

        return covalent_loss

class BayesianSteering(nn.Module):
    def __init__(self, enoe_params, egeom_params):
        super(BayesianSteering, self).__init__()
        self.enoe = ENOE(**enoe_params)
        self.egeom = EGeom(**egeom_params)

    def forward(self, pred_coords, batch, t):
        nef_data = batch.get('nef_data', [])
        enoe_loss = self.enoe(pred_coords, nef_data, t) if nef_data else 0.0
        egeom_loss = self.egeom(pred_coords, batch)
        return enoe_loss + egeom_loss
