#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""
Combined Bayesian Steering Module

This module unifies NOE energy and Geometric Prior into a single interface
for both training loss and inference-time steering.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from einops import repeat

from energy.noe_energy import NOEEnergy, NOEDistance
from energy.geometric_prior import GeometricPrior


class BayesianSteering(nn.Module):
    """
    Unified Bayesian Steering module.

    Combines E_NOE and E_Geom to provide:
    1. Training loss for fine-tuning
    2. Steering forces for inference
    """

    def __init__(
        self,
        use_noe: bool = True,
        use_geom: bool = True,
        noe_potential_type: str = "gaussian",
        noe_base_sigma: float = 0.5,
        noe_time_dependent: bool = True,
        geometric_clash_cutoff: float = 2.5,
        geometric_clash_penalty: float = 10.0,
        geometric_covalent_penalty: float = 5.0,
    ):
        super().__init__()
        self.use_noe = use_noe
        self.use_geom = use_geom

        if use_geom:
            self.geometric_prior = GeometricPrior(
                clash_cutoff=geometric_clash_cutoff,
                clash_penalty_scale=geometric_clash_penalty,
                covalent_penalty_scale=geometric_covalent_penalty
            )

        self.noe_config = {
            "potential_type": noe_potential_type,
            "base_sigma": noe_base_sigma,
            "time_dependent": noe_time_dependent
        }

    def _prepare_noe_energy(self, batch: Dict[str, Any]) -> Optional[NOEEnergy]:
        """Create a NOEEnergy module for the current batch's restraints."""
        if not self.use_noe or "noe_restraints" not in batch:
            return None

        # Batch-specific restraints should be provided as a list of lists of NOEDistance objects
        # or similar structure that we can use to initialize NOEEnergy.
        # Since NOEEnergy currently expects a flat list and handles one structure,
        # for batched training we might need to handle this carefully.

        # If we have restraints for each item in the batch:
        restraints = batch["noe_restraints"]
        if not restraints:
            return None

        return NOEEnergy(
            restraints=restraints,
            **self.noe_config
        )

    def forward(
        self,
        coords: torch.Tensor,
        batch: Dict[str, Any],
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total Bayesian energy.

        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates [B, N_atoms, 3]
        batch : Dict
            Batch features, including masks and NOE restraints
        t : torch.Tensor
            Time step [B]

        Returns
        -------
        Tuple[torch.Tensor, Dict]
            - Total energy [B]
            - Stats dictionary
        """
        batch_size = coords.shape[0]
        device = coords.device
        total_energy = torch.zeros(batch_size, device=device)
        stats = {}

        # 1. Geometric Prior
        if self.use_geom:
            atom_mask = batch["atom_resolved_mask"]
            geom_energy, geom_stats = self.geometric_prior(coords, atom_mask)
            total_energy = total_energy + geom_energy
            stats.update({f"geom/{k}": v for k, v in geom_stats.items()})
            stats["energy/geom"] = geom_energy.mean()

        # 2. NOE Energy
        if self.use_noe and "noe_restraints_indices" in batch:
            # We assume batch["noe_restraints_indices"] is a tensor of shape [B, K, 2, assignment_K]
            # or similar that allows vectorized computation.
            # For simplicity, let's assume we use a specialized vectorized NOE energy.
            noe_energy, noe_stats = self.compute_vectorized_noe_energy(coords, batch, t)
            total_energy = total_energy + noe_energy
            stats.update({f"noe/{k}": v for k, v in noe_stats.items()})
            stats["energy/noe"] = noe_energy.mean()

        stats["energy/total_bayesian"] = total_energy.mean()
        return total_energy, stats

    def compute_vectorized_noe_energy(
        self,
        coords: torch.Tensor,
        batch: Dict[str, Any],
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Vectorized computation of NOE energy for a batch.

        Requires:
        - batch["noe_at1_idx"]: [B, K, M] - atom 1 indices for K restraints, up to M assignments
        - batch["noe_at2_idx"]: [B, K, M] - atom 2 indices
        - batch["noe_mask"]: [B, K, M] - mask for valid assignments
        - batch["noe_upper_bounds"]: [B, K] - upper bounds
        - batch["noe_weights"]: [B, K] - weights
        """
        at1_idx = batch["noe_at1_idx"]  # [B, K, M]
        at2_idx = batch["noe_at2_idx"]  # [B, K, M]
        noe_mask = batch["noe_mask"]    # [B, K, M]
        upper_bounds = batch["noe_upper_bounds"] # [B, K]
        weights = batch["noe_weights"] # [B, K]

        B, K, M = at1_idx.shape

        # Native vectorized indexing
        # coords is [B, N, 3]
        # We want [B, K, M, 3]

        batch_indices = torch.arange(B, device=coords.device)[:, None, None]
        pos1 = coords[batch_indices, at1_idx] # [B, K, M, 3]
        pos2 = coords[batch_indices, at2_idx] # [B, K, M, 3]

        # Compute distances
        dists = torch.linalg.norm(pos1 - pos2, dim=-1) # [B, K, M]

        # r^-6 effective distance
        # d_eff = (sum_m (d_m^-6))^-1/6
        eps = 1e-4 # Increased epsilon for stability
        inv_d6 = torch.pow(dists + eps, -6.0) * noe_mask
        sum_inv_d6 = torch.sum(inv_d6, dim=-1) # [B, K]
        d_eff = torch.pow(sum_inv_d6 + eps, -1.0 / 6.0) # [B, K]

        # Violation
        violation = torch.relu(d_eff - upper_bounds) # [B, K]

        # Potential
        if t is None:
            t = torch.ones(B, device=coords.device)

        sigma = self.noe_config["base_sigma"]
        if self.noe_config["time_dependent"]:
            sigma = sigma * (1.0 + 5.0 * t) # [B]

        sigma = sigma.view(B, 1)

        if self.noe_config["potential_type"] == "gaussian":
            energy = (violation ** 2) / (2.0 * (sigma + eps) ** 2)
        else: # lognormal
            energy = 0.5 * (torch.log((violation + eps) / (sigma + eps)) / (sigma + eps)) ** 2

        energy = energy * weights # [B, K]

        total_noe_energy = torch.sum(energy, dim=-1) # [B]

        stats = {
            "mean_violation": violation[violation > 0].mean() if (violation > 0).any() else torch.tensor(0.0, device=coords.device),
            "max_violation": violation.max(),
        }

        return total_noe_energy, stats

    def compute_steering_force(
        self,
        coords: torch.Tensor,
        batch: Dict[str, Any],
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute F_steering = -∇_x E_Bayesian.

        Parameters
        ----------
        coords : torch.Tensor
            [B, N, 3]
        batch : Dict
        t : torch.Tensor [B]

        Returns
        -------
        torch.Tensor
            Steering force [B, N, 3]
        """
        with torch.enable_grad():
            x = coords.detach().requires_grad_(True)
            energy, _ = self.forward(x, batch, t)
            grad = torch.autograd.grad(energy.sum(), x)[0]

        return -grad
