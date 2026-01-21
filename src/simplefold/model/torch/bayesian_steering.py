#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
import torch.nn as nn
from simplefold.energy.noe_energy import NOEEnergy
from simplefold.energy.geom_energy import GeomEnergy

class BayesianSteering(nn.Module):
    """
    Unified module for calculating Bayesian steering forces.
    F_steering = -grad(E_Bayesian)
    """
    def __init__(self, noe_weight=1.0, clash_weight=1.0, covalent_weight=1.0, base_sigma=0.5):
        super().__init__()
        self.noe_energy = NOEEnergy(base_sigma=base_sigma)
        self.geom_energy = GeomEnergy(clash_weight=clash_weight, covalent_weight=covalent_weight)
        self.noe_weight = noe_weight

    def forward(self, coords, batch, t):
        """
        Calculates the total energy.
        coords: [B, N, 3]
        """
        # Ensure coords requires grad for force calculation
        # But we might call this inside a larger grad context

        # Scale coordinates if necessary (model might use scaled units)
        # Assuming coords are in Angstroms for energy calculation

        e_noe = self.noe_energy(coords, batch, t) if "noe_at1_idx" in batch else torch.zeros(coords.shape[0], device=coords.device)
        e_geom = self.geom_energy(coords, batch)

        total_energy = self.noe_weight * e_noe + e_geom
        return total_energy

    def get_steering_force(self, coords, batch, t):
        """
        Calculates the negative gradient of the energy w.r.t. coordinates.
        """
        with torch.enable_grad():
            coords_grad = coords.detach().clone().requires_grad_(True)
            energy = self.forward(coords_grad, batch, t)
            total_energy = energy.sum()
            total_energy.backward()
            force = -coords_grad.grad

        return force
