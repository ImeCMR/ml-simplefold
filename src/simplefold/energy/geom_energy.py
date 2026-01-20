import torch

def calculate_clash_energy(
    coords: torch.Tensor,
    mask: torch.Tensor,
    vdw_radii: torch.Tensor,
    overlap_threshold: float = 0.8,
    cutoff: float = 10.0,
) -> torch.Tensor:
    """
    Calculates clash energy with a distance cutoff for efficiency.
    """
    B, N, _ = coords.shape

    # We use cdist, but we can mask out distances above cutoff to reduce gradient overhead
    dists = torch.cdist(coords, coords) # (B, N, N)

    r_sum = vdw_radii.unsqueeze(-1) + vdw_radii.unsqueeze(-2) # (B, N, N)
    threshold = r_sum * overlap_threshold

    # Violation only if distance < threshold AND within overall cutoff
    violation = torch.clamp(threshold - dists, min=0.0)

    # Exclude self-interaction
    eye = torch.eye(N, device=coords.device).unsqueeze(0).expand(B, -1, -1)
    violation = violation * (1.0 - eye)

    # Efficiency: ignore pairs far apart
    violation = violation * (dists < cutoff).float()

    # Apply mask
    atom_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    violation = violation * atom_mask

    return (violation ** 2).sum() / 2.0

def calculate_covalent_energy(
    coords: torch.Tensor,
    connectivity: torch.Tensor, # (B, C, 2)
    ideal_lengths: torch.Tensor, # (B, C)
    mask: torch.Tensor, # (B, C)
) -> torch.Tensor:
    B, C, _ = connectivity.shape
    if C == 0: return torch.tensor(0.0, device=coords.device, requires_grad=True)

    batch_indices = torch.arange(B, device=coords.device).view(B, 1).expand(B, C)
    pos1 = coords[batch_indices, connectivity[:, :, 0]]
    pos2 = coords[batch_indices, connectivity[:, :, 1]]

    dists = torch.norm(pos1 - pos2, dim=-1)
    violation = dists - ideal_lengths
    energy = (violation ** 2) * mask
    return energy.sum() / 2.0
