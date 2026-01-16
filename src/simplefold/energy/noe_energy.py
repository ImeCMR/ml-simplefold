#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""
NOE (Nuclear Overhauser Effect) Energy Module

This module implements the E_NOE term for computing the agreement between
predicted structures and NOESY distance restraints using r^-6 summed effective
distance and a differentiable probabilistic potential.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class NOEDistance:
    """Represents a NOE distance restraint with possible ambiguities."""
    
    def __init__(
        self,
        atom_indices: List[Tuple[int, int]],  # List of (atom_i_idx, atom_j_idx) pairs
        upper_bound: float,  # upper distance bound in Angstroms
        lower_bound: Optional[float] = None,
        weight: float = 1.0,
    ):
        """
        Parameters
        ----------
        atom_indices : List[Tuple[int, int]]
            List of possible atom pair assignments for ambiguous NOE
        upper_bound : float
            Upper distance bound in Angstroms
        lower_bound : Optional[float]
            Lower distance bound (if None, only upper bound is used)
        weight : float
            Weight for this restraint in the loss
        """
        self.atom_indices = atom_indices
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound if lower_bound is not None else 0.0
        self.weight = weight
    
    def compute_effective_distance(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute r^-6 summed effective distance.
        
        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates [B, N_atoms, 3]
        
        Returns
        -------
        torch.Tensor
            Effective distance for this restraint
        """
        # Compute all pairwise distances for each assignment
        distances = []
        for atom_i, atom_j in self.atom_indices:
            # coords shape: [B, N_atoms, 3]
            vec = coords[:, atom_i] - coords[:, atom_j]  # [B, 3]
            dist = torch.linalg.norm(vec, dim=-1)  # [B]
            distances.append(dist)
        
        # Stack distances: [K, B] where K is number of assignments
        distances = torch.stack(distances, dim=0)  # [K, B]
        
        # Compute r^-6 average: (sum(d_k^-6))^(-1/6)
        # Add small epsilon to avoid division by zero
        eps = 1e-6
        inv_dist_6 = torch.pow(distances + eps, -6.0)  # [K, B]
        sum_inv_dist_6 = torch.sum(inv_dist_6, dim=0)  # [B]
        d_eff = torch.pow(sum_inv_dist_6, -1.0 / 6.0)  # [B]
        
        return d_eff


class NOEEnergy(nn.Module):
    """
    Computes E_NOE: the negative log-likelihood of NOESY distance restraints.
    
    This implements a fully differentiable energy term that:
    1. Computes r^-6 summed effective distances for each restraint
    2. Calculates violations relative to upper bounds
    3. Applies a time-dependent Gaussian or LogNormal potential
    """
    
    def __init__(
        self,
        restraints: List[NOEDistance],
        potential_type: str = "gaussian",
        base_sigma: float = 0.5,
        time_dependent: bool = True,
    ):
        """
        Parameters
        ----------
        restraints : List[NOEDistance]
            List of NOE restraints
        potential_type : str
            Type of potential: "gaussian" or "lognormal"
        base_sigma : float
            Base standard deviation for the potential (in Angstroms)
        time_dependent : bool
            Whether to scale sigma by time (tighter at t→1)
        """
        super().__init__()
        self.restraints = restraints
        self.potential_type = potential_type
        self.base_sigma = base_sigma
        self.time_dependent = time_dependent
        self.num_restraints = len(restraints)
    
    def compute_time_dependent_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time-dependent variance schedule.
        
        At t=0 (noisy), sigma is large (loose constraints).
        At t=1 (clean), sigma is small (tight constraints).
        
        Parameters
        ----------
        t : torch.Tensor
            Time step [B]
        
        Returns
        -------
        torch.Tensor
            Time-dependent sigma [B]
        """
        if not self.time_dependent:
            return torch.full_like(t, self.base_sigma)
        
        # Linear schedule: sigma(t) = base_sigma * (1.0 + 5.0 * (1.0 - t))
        # At t=1 (clean): sigma = base_sigma
        # At t=0 (noisy): sigma = 6 * base_sigma
        sigma = self.base_sigma * (1.0 + 5.0 * (1.0 - t))
        return sigma
    
    def gaussian_potential(
        self,
        violations: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gaussian potential: E = (Δd)^2 / (2σ^2)
        
        Parameters
        ----------
        violations : torch.Tensor
            Distance violations [B, K]
        sigma : torch.Tensor
            Standard deviation [B, 1] or scalar
        
        Returns
        -------
        torch.Tensor
            Energy values [B, K]
        """
        energy = (violations ** 2) / (2.0 * (sigma + 1e-6) ** 2)
        return energy
    
    def lognormal_potential(
        self,
        violations: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        LogNormal potential for better handling of violations > 0.
        
        Parameters
        ----------
        violations : torch.Tensor
            Distance violations [B, K]
        sigma : torch.Tensor
            Standard deviation parameter [B, 1] or scalar
        
        Returns
        -------
        torch.Tensor
            Energy values [B, K]
        """
        # Only penalize positive violations (Δd > 0)
        violations = torch.clamp(violations, min=0.0)
        
        # LogNormal: E = 0.5 * (ln(Δd / σ) / σ)^2 + ln(σ)
        eps = 1e-6
        energy = 0.5 * (torch.log((violations + eps) / (sigma + eps)) / (sigma + eps)) ** 2
        return energy
    
    def forward(
        self,
        coords: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute E_NOE energy for given coordinates.
        
        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates [B, N_atoms, 3]
        t : Optional[torch.Tensor]
            Time step [B] for time-dependent scaling
        
        Returns
        -------
        Tuple[torch.Tensor, Dict]
            - Energy [B], aggregated across restraints
            - Dictionary with detailed energy breakdown
        """
        batch_size = coords.shape[0]
        device = coords.device
        
        # Get time-dependent sigma
        if t is None:
            t = torch.ones(batch_size, device=device)
        sigma = self.compute_time_dependent_sigma(t)  # [B]
        sigma = sigma.unsqueeze(-1)  # [B, 1]
        
        # Compute violations for all restraints
        all_violations = []
        all_energies = []
        
        for restraint in self.restraints:
            d_eff = restraint.compute_effective_distance(coords)  # [B]
            
            # Compute violation relative to upper bound
            violation = d_eff - restraint.upper_bound  # [B]
            all_violations.append(violation)
            
            # Compute potential
            if self.potential_type == "gaussian":
                energy = self.gaussian_potential(violation.unsqueeze(-1), sigma)  # [B, 1]
            elif self.potential_type == "lognormal":
                energy = self.lognormal_potential(violation.unsqueeze(-1), sigma)  # [B, 1]
            else:
                raise ValueError(f"Unknown potential type: {self.potential_type}")
            
            # Apply restraint weight
            energy = energy * restraint.weight
            all_energies.append(energy.squeeze(-1))  # [B]
        
        # Stack and aggregate
        all_violations = torch.stack(all_violations, dim=-1)  # [B, K]
        all_energies = torch.stack(all_energies, dim=-1)  # [B, K]
        
        # Sum across restraints
        total_energy = torch.sum(all_energies, dim=-1)  # [B]
        
        # Compute statistics for logging
        mean_violation = torch.mean(all_violations)
        max_violation = torch.max(all_violations)
        num_violations = torch.sum(all_violations > 0.0).float() / (batch_size * self.num_restraints)
        
        return total_energy, {
            "mean_violation": mean_violation,
            "max_violation": max_violation,
            "fraction_violated": num_violations,
            "violations": all_violations.detach(),
            "energies": all_energies.detach(),
        }
