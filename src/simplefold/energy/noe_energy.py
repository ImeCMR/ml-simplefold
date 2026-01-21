#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

class NOEEnergy(nn.Module):
    """
    NOE energy term based on distance restraints.
    Calculates E_NOE = sum_i (max(0, d_eff,i - d_UB,i)^2) / (2 * sigma(t)^2)
    where d_eff,i is the r^-6 summed effective distance for ambiguous restraints.
    """
    def __init__(self, base_sigma=0.5):
        super().__init__()
        self.base_sigma = base_sigma

    def sigma_t(self, t):
        """
        Time-dependent variance schedule.
        sigma(t) = base_sigma * (1.0 + 5.0 * (1.0 - t))
        This ensures higher variance (looser constraints) at high noise (t approx 0)
        and lower variance (tighter constraints) near the clean data manifold (t -> 1).
        """
        return self.base_sigma * (1.0 + 5.0 * (1.0 - t))

    def forward(self, coords, batch, t):
        """
        Args:
            coords: [B, N, 3] atom coordinates (in Angstroms)
            batch: dictionary containing:
                - noe_at1_idx: [B, M, K] indices of first atoms in restraint pairs
                - noe_at2_idx: [B, M, K] indices of second atoms in restraint pairs
                - noe_mask: [B, M, K] mask for valid pairs in ambiguous restraints
                - noe_upper_bounds: [B, M] upper distance bounds (d_UB)
                - noe_weights: [B, M] optional weights for each restraint
            t: [B] or float, current time step
        Returns:
            energy: [B] total NOE energy per batch element
        """
        B, N, _ = coords.shape
        at1_idx = batch["noe_at1_idx"] # [B, M, K]
        at2_idx = batch["noe_at2_idx"] # [B, M, K]
        noe_mask = batch["noe_mask"]   # [B, M, K]
        upper_bounds = batch["noe_upper_bounds"] # [B, M]
        weights = batch.get("noe_weights", torch.ones_like(upper_bounds)) # [B, M]

        # Flatten for indexing
        # coords_flat: [B * N, 3]
        # at1_idx_flat: [B * M * K]
        # We need to account for batch offset in indices
        batch_offset = torch.arange(B, device=coords.device).view(B, 1, 1) * N
        at1_idx_offset = (at1_idx + batch_offset).view(-1)
        at2_idx_offset = (at2_idx + batch_offset).view(-1)

        # Extract atom positions
        pos1 = coords.view(-1, 3)[at1_idx_offset].view(B, -1, at1_idx.shape[-1], 3) # [B, M, K, 3]
        pos2 = coords.view(-1, 3)[at2_idx_offset].view(B, -1, at2_idx.shape[-1], 3) # [B, M, K, 3]

        # Distances
        dists = torch.norm(pos1 - pos2, dim=-1) # [B, M, K]

        # Avoid division by zero in r^-6
        eps = 1e-6
        dists_inv6 = torch.pow(dists + eps, -6.0)

        # Effective distance d_eff = (sum_k d_k^-6)^-1/6
        # Apply mask for ambiguous restraints
        masked_dists_inv6 = dists_inv6 * noe_mask
        sum_inv6 = torch.sum(masked_dists_inv6, dim=-1) # [B, M]

        # Handle cases where sum_inv6 is 0 (should not happen for valid restraints)
        d_eff = torch.pow(sum_inv6 + eps, -1.0/6.0) # [B, M]

        # Penalty: max(0, d_eff - d_UB)^2
        violation = F.relu(d_eff - upper_bounds)
        penalty = torch.pow(violation, 2) # [B, M]

        # Scale by weights and sigma(t)
        sigma = self.sigma_t(t)
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.view(B, 1)

        energy = torch.sum(weights * penalty, dim=-1) / (2.0 * sigma**2 + eps)

        return energy
