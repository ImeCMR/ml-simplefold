#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""
Verification script for Bayesian Steering loss.
"""

import torch
import torch.nn.functional as F
from model.torch.bayesian_steering import BayesianSteering

def test_bayesian_steering():
    # Setup parameters
    B, N, K, M = 2, 50, 10, 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy data
    coords = torch.randn(B, N, 3, device=device, requires_grad=True)
    atom_resolved_mask = torch.ones(B, N, device=device)

    # Dummy NEF restraints
    noe_at1_idx = torch.randint(0, N, (B, K, M), device=device)
    noe_at2_idx = torch.randint(0, N, (B, K, M), device=device)
    noe_mask = torch.ones(B, K, M, device=device)
    noe_upper_bounds = torch.ones(B, K, device=device) * 5.0
    noe_weights = torch.ones(B, K, device=device)

    batch = {
        "atom_resolved_mask": atom_resolved_mask,
        "noe_at1_idx": noe_at1_idx,
        "noe_at2_idx": noe_at2_idx,
        "noe_mask": noe_mask,
        "noe_upper_bounds": noe_upper_bounds,
        "noe_weights": noe_weights,
        "noe_restraints_indices": True # Flag to enable vectorized NOE
    }

    t = torch.tensor([0.5, 0.8], device=device)

    # Initialize BayesianSteering
    steering = BayesianSteering(
        use_noe=True,
        use_geom=True,
        geometric_clash_cutoff=2.5,
        geometric_clash_penalty=10.0,
        geometric_covalent_penalty=5.0
    ).to(device)

    # 1. Test Energy Calculation
    energy, stats = steering(coords, batch, t)
    print("Energy shape:", energy.shape)
    print("Stats:", stats)

    assert energy.shape == (B,)
    assert "energy/total_bayesian" in stats

    # 2. Test Differentiability
    loss = energy.sum()
    loss.backward()

    assert coords.grad is not None
    print("Gradient shape:", coords.grad.shape)
    print("Mean gradient magnitude:", coords.grad.abs().mean().item())

    # 3. Test Steering Force
    force = steering.compute_steering_force(coords.detach(), batch, t)
    print("Force shape:", force.shape)
    assert force.shape == (B, N, 3)

    print("Verification successful!")

if __name__ == "__main__":
    test_bayesian_steering()
