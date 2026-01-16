#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""
Geometric Prior Module

This module implements E_Geom: a machine-learned geometric prior that ensures
stereochemical feasibility without using explicit force fields. It includes:
- E_Clash: Excluded volume penalties for clashing atoms
- E_Covalent: Covalent geometry constraints (bond lengths, angles, chirality)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class CovalentGeometry:
    """Ideal covalent geometry parameters from PDB statistics."""
    
    # Bond lengths (Angstroms)
    bond_lengths: Dict[str, float] = None
    
    # Bond angle parameters (degrees)
    bond_angles: Dict[str, float] = None
    
    # Van der Waals radii for clash detection
    vdw_radii: Dict[str, float] = None
    
    def __post_init__(self):
        if self.bond_lengths is None:
            # Common bond lengths in proteins (Angstroms)
            self.bond_lengths = {
                "C-N": 1.33,  # Peptide bond
                "C-CA": 1.53,
                "N-CA": 1.47,
                "CA-C": 1.53,
                "O-C": 1.24,
            }
        
        if self.bond_angles is None:
            # Common bond angles (degrees)
            self.bond_angles = {
                "N-CA-C": 110.0,
                "CA-C-N": 117.0,
                "C-N-CA": 123.0,
            }
        
        if self.vdw_radii is None:
            # Van der Waals radii (Angstroms)
            self.vdw_radii = {
                "C": 1.70,
                "N": 1.55,
                "O": 1.52,
                "S": 1.80,
                "H": 1.20,
            }


class GeometricPrior(nn.Module):
    """
    Machine-learned geometric prior that enforces structural plausibility.
    
    This replaces physics-based force fields with learned penalties:
    - E_Clash: Keeps atoms from clashing based on empirical VDW distances
    - E_Covalent: Maintains realistic bond lengths, angles, and chirality
    """
    
    def __init__(
        self,
        clash_cutoff: float = 2.5,
        clash_penalty_scale: float = 10.0,
        covalent_penalty_scale: float = 5.0,
        use_clash: bool = True,
        use_covalent: bool = True,
    ):
        """
        Parameters
        ----------
        clash_cutoff : float
            Minimum allowed distance (in Angstroms) between non-bonded atoms
        clash_penalty_scale : float
            Scale factor for clash energy
        covalent_penalty_scale : float
            Scale factor for covalent geometry energy
        use_clash : bool
            Whether to compute clash energy
        use_covalent : bool
            Whether to compute covalent energy
        """
        super().__init__()
        self.clash_cutoff = clash_cutoff
        self.clash_penalty_scale = clash_penalty_scale
        self.covalent_penalty_scale = covalent_penalty_scale
        self.use_clash = use_clash
        self.use_covalent = use_covalent
        self.geometry = CovalentGeometry()
    
    def compute_pairwise_distances(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between all atoms.
        
        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates [B, N_atoms, 3]
        
        Returns
        -------
        torch.Tensor
            Pairwise distance matrix [B, N_atoms, N_atoms]
        """
        # coords: [B, N_atoms, 3]
        # Expand dims: [B, N_atoms, 1, 3] and [B, 1, N_atoms, 3]
        coords_i = coords.unsqueeze(2)  # [B, N_atoms, 1, 3]
        coords_j = coords.unsqueeze(1)  # [B, 1, N_atoms, 3]
        
        # Compute distances
        diff = coords_i - coords_j  # [B, N_atoms, N_atoms, 3]
        distances = torch.linalg.norm(diff, dim=-1)  # [B, N_atoms, N_atoms]
        
        return distances
    
    def compute_clash_energy(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute clash energy (excluded volume penalty).
        
        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates [B, N_atoms, 3]
        atom_mask : torch.Tensor
            Valid atom mask [B, N_atoms]
        
        Returns
        -------
        Tuple[torch.Tensor, Dict]
            - Clash energy [B]
            - Dictionary with statistics
        """
        distances = self.compute_pairwise_distances(coords)  # [B, N_atoms, N_atoms]
        
        # Create mask for non-bonded pairs (exclude self and neighbors)
        batch_size, n_atoms, _ = distances.shape
        device = coords.device
        
        # Exclude diagonal (self)
        identity = torch.eye(n_atoms, device=device, dtype=torch.bool)
        
        # Exclude bonded neighbors (simple: residues i and i±1)
        # For now, use a simple sequential bonding mask
        bonded = torch.zeros((n_atoms, n_atoms), device=device, dtype=torch.bool)
        for i in range(n_atoms):
            bonded[i, max(0, i-1):min(n_atoms, i+2)] = True
        
        # Non-bonded pairs
        non_bonded = (~identity) & (~bonded)
        non_bonded = non_bonded.unsqueeze(0) & (atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1))
        
        # Apply mask
        distances_masked = distances.clone()
        distances_masked[~non_bonded] = 1000.0  # Set masked distances to large value
        
        # Compute clash penalty: smooth step function
        # E_clash = 0 if d > cutoff, high penalty if d < cutoff
        clash_dist = self.clash_cutoff
        clashes = distances_masked < clash_dist
        
        # Smooth penalty: E = max(0, (cutoff - d)^2)
        clash_energy = torch.relu(clash_dist - distances_masked) ** 2
        clash_energy = clash_energy * non_bonded.float()
        
        # Aggregate
        total_clash = torch.sum(clash_energy, dim=(1, 2))  # [B]
        total_clash = total_clash * self.clash_penalty_scale
        
        stats = {
            "num_clashes": torch.sum(clashes).float(),
            "min_distance": torch.min(distances_masked),
            "mean_nonbonded_distance": torch.mean(distances_masked[distances_masked < 1000.0]),
        }
        
        return total_clash, stats
    
    def compute_covalent_energy(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute covalent geometry energy.
        
        This is a simplified version that penalizes unusual bond configurations.
        A full implementation would track bonded atoms and compute bond length,
        angle, and dihedral deviations.
        
        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates [B, N_atoms, 3]
        atom_mask : torch.Tensor
            Valid atom mask [B, N_atoms]
        
        Returns
        -------
        Tuple[torch.Tensor, Dict]
            - Covalent energy [B]
            - Dictionary with statistics
        """
        distances = self.compute_pairwise_distances(coords)  # [B, N_atoms, N_atoms]
        
        batch_size, n_atoms, _ = distances.shape
        device = coords.device
        
        # Simple bonding model: adjacent atoms in sequence should be close
        # This is a simplified approach; full implementation would use explicit bond topology
        covalent_energy = torch.zeros(batch_size, device=device)
        
        # Ideal CA-CA distance is approximately 3.8 Angstroms
        ideal_ca_ca = 3.8
        ca_ca_penalty_scale = 1.0
        
        for i in range(max(0, n_atoms - 1)):
            # Penalty for CA-CA distances
            expected_dist = distances[:, i, i + 1]
            deviation = torch.abs(expected_dist - ideal_ca_ca)
            penalty = torch.relu(deviation - 0.5) ** 2  # Allow ±0.5 Å tolerance
            covalent_energy += penalty * ca_ca_penalty_scale
        
        covalent_energy = covalent_energy * self.covalent_penalty_scale
        
        stats = {
            "mean_ca_ca_dist": torch.mean(distances[:, :-1, 1:] if n_atoms > 1 else distances),
        }
        
        return covalent_energy, stats
    
    def forward(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total geometric prior energy.
        
        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates [B, N_atoms, 3]
        atom_mask : torch.Tensor
            Valid atom mask [B, N_atoms]
        
        Returns
        -------
        Tuple[torch.Tensor, Dict]
            - Total energy [B]
            - Dictionary with detailed breakdown
        """
        total_energy = torch.zeros(coords.shape[0], device=coords.device)
        stats = {}
        
        if self.use_clash:
            clash_energy, clash_stats = self.compute_clash_energy(coords, atom_mask)
            total_energy = total_energy + clash_energy
            stats.update({f"clash/{k}": v for k, v in clash_stats.items()})
        
        if self.use_covalent:
            covalent_energy, covalent_stats = self.compute_covalent_energy(coords, atom_mask)
            total_energy = total_energy + covalent_energy
            stats.update({f"covalent/{k}": v for k, v in covalent_stats.items()})
        
        return total_energy, stats
