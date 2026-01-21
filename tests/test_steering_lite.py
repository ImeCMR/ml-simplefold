
import torch
from simplefold.energy.noe_energy import NOEEnergy
from simplefold.energy.geom_energy import GeomEnergy
from simplefold.model.torch.bayesian_steering import BayesianSteering

def test_bayesian_steering():
    B, N = 2, 10
    coords = torch.randn(B, N, 3)
    batch = {
        "atom_pad_mask": torch.ones(B, N),
        "atom_types": torch.zeros(B, N, dtype=torch.long),
        "noe_at1_idx": torch.randint(0, N, (B, 5, 1)),
        "noe_at2_idx": torch.randint(0, N, (B, 5, 1)),
        "noe_mask": torch.ones(B, 5, 1),
        "noe_upper_bounds": torch.ones(B, 5) * 5.0,
    }
    t = torch.tensor([0.5, 0.5])

    steering = BayesianSteering()
    energy = steering(coords, batch, t)
    print(f"Energy shape: {energy.shape}")
    # assert energy.shape == (B,)
    # assert energy.requires_grad == False # Unless coords has grad

    force = steering.get_steering_force(coords, batch, t)
    assert force.shape == (B, N, 3)
    print("Test passed!")

if __name__ == "__main__":
    test_bayesian_steering()
