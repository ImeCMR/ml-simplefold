import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src/simplefold"))
sys.path.append(os.path.abspath("src"))

from energy.noe_energy import calculate_noe_energy, get_sigma_t
from model.torch.bayesian_steering import BayesianSteering

def test_noe_energy_ambiguous():
    print("Testing ambiguous NOE energy...")
    # One restraint with two pairs
    coords = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], requires_grad=True)
    at1_idx = torch.tensor([[[0, 2]]]) # (B, R, K)
    at2_idx = torch.tensor([[[1, 3]]])
    upper_bounds = torch.tensor([[5.0]])
    mask = torch.tensor([[1.0]])
    ambiguous_mask = torch.tensor([[[1.0, 1.0]]])

    energy = calculate_noe_energy(coords, at1_idx, at2_idx, upper_bounds, mask, ambiguous_mask, sigma_t=1.0)
    print(f"Energy: {energy.item()}")

    # d1 = 10, d2 = 1. d_eff = (10^-6 + 1^-6)^-1/6 = (1.000001)^-1/6 approx 1.0
    # violation = max(0, 1.0 - 5.0) = 0.
    # Energy should be 0.
    assert torch.allclose(energy, torch.tensor(0.0))

    # Test violation
    upper_bounds_small = torch.tensor([[0.5]])
    energy_v = calculate_noe_energy(coords, at1_idx, at2_idx, upper_bounds_small, mask, ambiguous_mask, sigma_t=1.0)
    print(f"Violation Energy: {energy_v.item()}")
    assert energy_v > 0

    energy_v.backward()
    assert coords.grad is not None
    print("Ambiguous NOE energy test passed!")

def test_steering_force():
    print("Testing steering force...")
    steering = BayesianSteering(noe_weight=1.0, clash_weight=0.0, covalent_weight=0.0)
    coords = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    batch = {
        "noe_at1_idx": torch.tensor([[[0]]]),
        "noe_at2_idx": torch.tensor([[[1]]]),
        "noe_upper_bounds": torch.tensor([[5.0]]),
        "noe_mask": torch.tensor([[1.0]]),
        "noe_ambiguous_mask": torch.tensor([[[1.0]]]),
        "scale": 1.0
    }

    force = steering.get_steering_force(coords, t=1.0, batch=batch)
    print(f"Force: {force}")
    # sigma(1.0) = 0.1
    # d_eff = 10.
    # E = (10-5)^2 / (2 * 0.01) = 25 / 0.02 = 1250.
    # grad is same as before.
    assert torch.allclose(force[0, 0, 0], torch.tensor(500.0))
    print("Steering force test passed!")

if __name__ == "__main__":
    test_noe_energy_ambiguous()
    test_steering_force()
