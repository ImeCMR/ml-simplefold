import torch
import torch.nn as nn
from typing import Dict, Optional
from energy.noe_energy import calculate_noe_energy, get_sigma_t
from energy.geom_energy import calculate_clash_energy, calculate_covalent_energy

class BayesianSteering(nn.Module):
    def __init__(
        self,
        base_sigma: float = 0.1,
        overlap_threshold: float = 0.8,
        clash_weight: float = 1.0,
        covalent_weight: float = 1.0,
        noe_weight: float = 1.0,
    ):
        super().__init__()
        self.base_sigma = base_sigma
        self.overlap_threshold = overlap_threshold
        self.clash_weight = clash_weight
        self.covalent_weight = covalent_weight
        self.noe_weight = noe_weight

    def forward(self, coords: torch.Tensor, t: float, batch: Dict) -> torch.Tensor:
        """
        Calculates total Bayesian energy potential.
        coords: (B, N, 3) in Angstroms
        """
        total_energy = torch.zeros(1, device=coords.device, requires_grad=True)

        # NOE Energy
        if "noe_at1_idx" in batch and batch["noe_at1_idx"] is not None:
            sigma_t = get_sigma_t(t, self.base_sigma)
            noe_energy = calculate_noe_energy(
                coords,
                batch["noe_at1_idx"],
                batch["noe_at2_idx"],
                batch["noe_upper_bounds"],
                batch["noe_mask"],
                batch.get("noe_ambiguous_mask", torch.ones_like(batch["noe_at1_idx"])),
                sigma_t=sigma_t,
            )
            total_energy = total_energy + self.noe_weight * noe_energy

        # Clash Energy
        if "vdw_radii" in batch and batch["vdw_radii"] is not None:
            clash_energy = calculate_clash_energy(
                coords,
                batch["atom_pad_mask"],
                batch["vdw_radii"],
                overlap_threshold=self.overlap_threshold,
            )
            total_energy = total_energy + self.clash_weight * clash_energy

        # Covalent Energy
        if "bond_connectivity" in batch and batch["bond_connectivity"] is not None:
            covalent_energy = calculate_covalent_energy(
                coords,
                batch["bond_connectivity"],
                batch["bond_ideal_lengths"],
                batch.get("bond_mask", torch.ones_like(batch["bond_ideal_lengths"])),
            )
            total_energy = total_energy + self.covalent_weight * covalent_energy

        return total_energy

    def get_steering_force(self, coords: torch.Tensor, t: float, batch: Dict) -> torch.Tensor:
        coords = coords.detach().requires_grad_(True)
        scale = batch.get("scale", 16.0)
        coords_angstrom = coords * scale

        energy = self.forward(coords_angstrom, t, batch)

        if energy.requires_grad:
            grads = torch.autograd.grad(energy, coords, create_graph=False)[0]
            return -grads
        else:
            return torch.zeros_like(coords)
