import torch
from torch import nn

class BayesianSteering(nn.Module):
    def __init__(self, noe_weight=1.0, geom_weight=1.0, time_dependent_variance=True, ideal_lengths=None, ideal_angles=None, vdw_radii=None):
        super().__init__()
        self.noe_weight = noe_weight
        self.geom_weight = geom_weight
        self.time_dependent_variance = time_dependent_variance
        self.ideal_lengths = ideal_lengths
        self.ideal_angles = ideal_angles
        self.vdw_radii = vdw_radii

    def compute_energy(self, x_t, restraints, covalent_info, t):
        batch_size = x_t.shape[0]
        e_noe_batch = []
        e_geom_batch = []

        for i in range(batch_size):
            e_noe_batch.append(self.compute_e_noe(x_t[i], restraints[i], t[i]))
            e_geom_batch.append(self.compute_e_geom(x_t[i], covalent_info[i]))

        e_noe = torch.stack(e_noe_batch)
        e_geom = torch.stack(e_geom_batch)
        e_bayesian = self.noe_weight * e_noe + self.geom_weight * e_geom
        return e_bayesian

    def forward(self, x_t, restraints, covalent_info, t):
        x_t.requires_grad_(True)
        e_bayesian = self.compute_energy(x_t, restraints, covalent_info, t)

        f_steering = -torch.autograd.grad(e_bayesian.sum(), x_t, retain_graph=True)[0]
        x_t.requires_grad_(False)
        return f_steering

    def compute_e_noe(self, x_t, restraints, t):
        total_e_noe = 0.0
        if not restraints:
            return torch.tensor(0.0, device=x_t.device)

        for restraint in restraints:
            indices1 = restraint['indices1']
            indices2 = restraint['indices2']
            d_ub = restraint['upper_bound']

            sum_r_neg_6 = 0.0
            for i1 in indices1:
                for i2 in indices2:
                    dist_sq = torch.sum((x_t[i1] - x_t[i2])**2)
                    sum_r_neg_6 += (dist_sq + 1e-8)**-3

            if sum_r_neg_6 < 1e-8:
                continue

            d_eff = sum_r_neg_6**(-1/6)
            delta_d = d_eff - d_ub

            variance = 1.0
            if self.time_dependent_variance:
                variance = variance * (1.0 - t) + 0.1 * t

            total_e_noe += torch.clamp(delta_d, min=0)**2 / variance

        return total_e_noe

    def compute_e_geom(self, x_t, covalent_info):
        total_e_geom = 0.0
        atom_types = covalent_info['atom_types']

        # E_Clash
        if self.vdw_radii is not None:
            for i in range(x_t.shape[0]):
                for j in range(i + 1, x_t.shape[0]):
                    is_bonded = any(((i, j) in covalent_info.get('bonds', []) or (j, i) in covalent_info.get('bonds', [])))
                    if not is_bonded:
                        dist = torch.norm(x_t[i] - x_t[j])
                        vdw_sum = self.vdw_radii.get(atom_types[i], 1.5) + self.vdw_radii.get(atom_types[j], 1.5)
                        if dist < vdw_sum:
                            total_e_geom += (dist - vdw_sum)**2

        # E_Covalent
        bonds = covalent_info.get('bonds', [])
        for idx1, idx2 in bonds:
            bond_type = (atom_types[idx1], atom_types[idx2])
            ideal_length = self.ideal_lengths.get(bond_type)
            if ideal_length is not None:
                dist = torch.norm(x_t[idx1] - x_t[idx2])
                total_e_geom += (dist - ideal_length)**2

        angles = covalent_info.get('angles', [])
        for idx1, idx2, idx3 in angles:
            angle_type = (atom_types[idx1], atom_types[idx2], atom_types[idx3])
            ideal_angle = self.ideal_angles.get(angle_type)
            if ideal_angle is not None:
                v1 = x_t[idx1] - x_t[idx2]
                v2 = x_t[idx3] - x_t[idx2]
                cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
                angle_val = torch.acos(cos_angle)
                total_e_geom += (angle_val - ideal_angle)**2

        return total_e_geom
