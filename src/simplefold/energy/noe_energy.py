import torch

def calculate_noe_energy(
    coords: torch.Tensor,
    at1_idx: torch.Tensor,
    at2_idx: torch.Tensor,
    upper_bounds: torch.Tensor,
    mask: torch.Tensor,
    ambiguous_mask: torch.Tensor, # (B, R, K)
    sigma_t: float = 1.0,
) -> torch.Tensor:
    """
    Calculates NOE energy using r^-6 summed effective distance for ambiguous restraints.

    Parameters:
    - coords: (B, N, 3) tensor of atom coordinates (Angstroms)
    - at1_idx: (B, R, K) indices of the first atom in each pair for each restraint
    - at2_idx: (B, R, K) indices of the second atom in each pair
    - upper_bounds: (B, R) upper distance bounds
    - mask: (B, R) mask for valid restraints
    - ambiguous_mask: (B, R, K) mask for valid pairs in each ambiguous restraint
    - sigma_t: Time-dependent variance

    Returns:
    - energy: scalar tensor
    """
    B, R, K = at1_idx.shape

    batch_indices = torch.arange(B, device=coords.device).view(B, 1, 1).expand(B, R, K)
    pos1 = coords[batch_indices, at1_idx] # (B, R, K, 3)
    pos2 = coords[batch_indices, at2_idx] # (B, R, K, 3)

    # Calculate Euclidean distances
    dists = torch.norm(pos1 - pos2, dim=-1) + 1e-6 # (B, R, K)

    # Effective distance: d_eff = (sum d^-6)^-1/6
    d_minus_6 = (dists ** -6) * ambiguous_mask
    sum_d_minus_6 = torch.sum(d_minus_6, dim=-1) + 1e-12 # (B, R)
    d_eff = sum_d_minus_6 ** (-1.0/6.0) # (B, R)

    # Violation
    violation = torch.clamp(d_eff - upper_bounds, min=0.0)

    # Quadratic penalty
    energy = (violation ** 2) / (2 * (sigma_t ** 2))

    # Apply mask and sum
    masked_energy = energy * mask
    return masked_energy.sum()

def get_sigma_t(t: float, base_sigma: float = 0.1) -> float:
    return base_sigma * (1.0 + 5.0 * (1.0 - t))
