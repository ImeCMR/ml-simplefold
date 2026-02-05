#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""
Unit tests for the BayesianSteering module.
"""

import unittest
import torch
from simplefold.model.torch.bayesian_steering import BayesianSteering


class TestBayesianSteering(unittest.TestCase):
    """Test BayesianSteering unified module."""

    def setUp(self):
        self.B, self.N, self.K, self.M = 2, 20, 5, 2
        self.device = torch.device("cpu")

        self.coords = torch.randn(self.B, self.N, 3, device=self.device, requires_grad=True)
        self.atom_resolved_mask = torch.ones(self.B, self.N, device=self.device)

        # Dummy NEF restraints
        self.noe_at1_idx = torch.randint(0, self.N, (self.B, self.K, self.M), device=self.device)
        self.noe_at2_idx = torch.randint(0, self.N, (self.B, self.K, self.M), device=self.device)
        self.noe_mask = torch.ones(self.B, self.K, self.M, device=self.device)
        self.noe_upper_bounds = torch.ones(self.B, self.K, device=self.device) * 5.0
        self.noe_weights = torch.ones(self.B, self.K, device=self.device)

        self.batch = {
            "atom_resolved_mask": self.atom_resolved_mask,
            "noe_at1_idx": self.noe_at1_idx,
            "noe_at2_idx": self.noe_at2_idx,
            "noe_mask": self.noe_mask,
            "noe_upper_bounds": self.noe_upper_bounds,
            "noe_weights": self.noe_weights,
            "noe_restraints_indices": torch.tensor(True)
        }

        self.t = torch.tensor([0.5, 0.5], device=self.device)

    def test_initialization(self):
        """Test module initialization with various configs."""
        steering = BayesianSteering(use_noe=True, use_geom=True)
        self.assertTrue(steering.use_noe)
        self.assertTrue(steering.use_geom)
        self.assertIsNotNone(steering.geometric_prior)

        steering_no_noe = BayesianSteering(use_noe=False)
        self.assertFalse(steering_no_noe.use_noe)

    def test_forward_energy(self):
        """Test forward energy calculation and shapes."""
        steering = BayesianSteering(use_noe=True, use_geom=True)
        energy, stats = steering(self.coords, self.batch, self.t)

        self.assertEqual(energy.shape, (self.B,))
        self.assertIn("energy/total_bayesian", stats)
        self.assertIn("geom/clash/num_clashes", stats)
        self.assertIn("noe/max_violation", stats)

    def test_differentiability(self):
        """Test that we can compute gradients through the energy."""
        steering = BayesianSteering(use_noe=True, use_geom=True)
        energy, _ = steering(self.coords, self.batch, self.t)
        loss = energy.sum()
        loss.backward()

        self.assertIsNotNone(self.coords.grad)
        self.assertEqual(self.coords.grad.shape, self.coords.shape)
        self.assertFalse(torch.isnan(self.coords.grad).any())

    def test_steering_force(self):
        """Test steering force computation."""
        steering = BayesianSteering(use_noe=True, use_geom=True)
        force = steering.compute_steering_force(self.coords, self.batch, self.t)

        self.assertEqual(force.shape, self.coords.shape)
        # Force should be -grad
        # We check this by manually computing grad
        self.coords.grad = None
        energy, _ = steering(self.coords, self.batch, self.t)
        loss = energy.sum()
        loss.backward()

        torch.testing.assert_close(force, -self.coords.grad)

    def test_time_dependence(self):
        """Test that energy changes with time (sigma schedule)."""
        steering = BayesianSteering(use_noe=True, use_geom=False, noe_time_dependent=True)

        # Coordinates with a violation
        # Atom 0 at (0,0,0), Atom 1 at (10,0,0) -> Distance 10.0
        # Upper bound 5.0 -> Violation 5.0
        coords = torch.zeros((self.B, self.N, 3))
        coords[:, 0] = torch.tensor([0.0, 0.0, 0.0])
        coords[:, 1] = torch.tensor([10.0, 0.0, 0.0])

        batch = self.batch.copy()
        batch = self.batch.copy()
        batch["noe_at1_idx"] = torch.zeros((self.B, 1, 1), dtype=torch.long)
        batch["noe_at2_idx"] = torch.ones((self.B, 1, 1), dtype=torch.long)
        batch["noe_mask"] = torch.ones((self.B, 1, 1))
        batch["noe_upper_bounds"] = torch.ones((self.B, 1)) * 2.0 # Upper bound 2.0 -> violation = 8.0
        batch["noe_weights"] = torch.ones((self.B, 1))

        # t=1 (clean) -> sigma = base_sigma
        # t=0 (noisy) -> sigma = 6 * base_sigma
        t0 = torch.zeros(self.B)
        t1 = torch.ones(self.B)

        energy0, _ = steering(coords, batch, t0) # sigma = 6 * base_sigma (larger)
        energy1, _ = steering(coords, batch, t1) # sigma = base_sigma (smaller)

        print(f"Energy at t=0 (sigma={6*0.5}): {energy0.tolist()}")
        print(f"Energy at t=1 (sigma={0.5}): {energy1.tolist()}")

        # Higher sigma means lower energy for same violation
        self.assertTrue((energy0 < energy1).all())


if __name__ == '__main__':
    unittest.main()
