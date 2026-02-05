#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""
Unit tests for energy potentials (NOEEnergy and GeometricPrior).
"""

import unittest
import torch
from simplefold.energy.noe_energy import NOEEnergy, NOEDistance
from simplefold.energy.geometric_prior import GeometricPrior


class TestNOEEnergy(unittest.TestCase):
    """Test NOEEnergy module."""

    def test_r6_effective_distance(self):
        """Test r^-6 effective distance calculation."""
        # Simple case: 2 assignments for one restraint
        # Assignments: (0, 1) and (0, 2)
        # Atoms at: atom 0 (0,0,0), atom 1 (3,0,0), atom 2 (4,0,0)
        # Distances: d1 = 3.0, d2 = 4.0
        # d_eff = (3^-6 + 4^-6)^(-1/6)

        coords = torch.tensor([[[0.0, 0.0, 0.0],
                                [3.0, 0.0, 0.0],
                                [4.0, 0.0, 0.0]]])

        restraint = NOEDistance(
            atom_indices=[(0, 1), (0, 2)],
            upper_bound=5.0
        )

        d_eff = restraint.compute_effective_distance(coords)

        expected = (3.0**-6 + 4.0**-6)**(-1/6)
        self.assertAlmostEqual(d_eff.item(), expected, places=5)

    def test_noe_potential_gaussian(self):
        """Test Gaussian potential calculation."""
        # violation = 1.0, sigma = 0.5
        # energy = (1.0)^2 / (2 * 0.5^2) = 1.0 / 0.5 = 2.0

        restraints = [NOEDistance([(0, 1)], upper_bound=2.0)]
        energy_mod = NOEEnergy(
            restraints,
            potential_type="gaussian",
            base_sigma=0.5,
            time_dependent=False
        )

        # distance = 3.0, violation = 1.0
        coords = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])

        energy, stats = energy_mod(coords)
        self.assertAlmostEqual(energy.item(), 2.0, places=5)
        self.assertAlmostEqual(stats["mean_violation"].item(), 1.0, places=5)

    def test_noe_potential_lognormal(self):
        """Test LogNormal potential calculation."""
        restraints = [NOEDistance([(0, 1)], upper_bound=2.0)]
        energy_mod = NOEEnergy(
            restraints,
            potential_type="lognormal",
            base_sigma=0.5,
            time_dependent=False
        )

        coords = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
        energy, stats = energy_mod(coords)

        # E = 0.5 * (ln(Δd / σ) / σ)^2
        # Δd = 1.0, σ = 0.5
        # E = 0.5 * (ln(1.0 / 0.5) / 0.5)^2 = 0.5 * (ln(2) * 2)^2 = 2 * (ln 2)^2
        import math
        expected = 2 * (math.log(2)**2)
        self.assertAlmostEqual(energy.item(), expected, places=5)


class TestGeometricPrior(unittest.TestCase):
    """Test GeometricPrior module."""

    def test_clash_energy(self):
        """Test clash penalty calculation."""
        # Atoms at (0,0,0) and (1,0,0) -> distance 1.0
        # Clash cutoff 2.5
        # penalty = (2.5 - 1.0)^2 = 1.5^2 = 2.25
        # scale = 10.0 -> energy = 22.5

        coords = torch.tensor([[[0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [10.0, 10.0, 10.0]]]) # Far away atom
        mask = torch.ones(1, 3)

        # We need to be careful with bonding exclusion
        # In current implementation, i and i+1 are excluded.
        # Let's put atoms at 0 and 2.
        coords = torch.tensor([[[0.0, 0.0, 0.0],
                                [10.0, 10.0, 10.0],
                                [1.0, 0.0, 0.0]]])
        mask = torch.ones(1, 3)

        prior = GeometricPrior(
            clash_cutoff=2.5,
            clash_penalty_scale=10.0,
            use_covalent=False
        )

        energy, stats = prior(coords, mask)
        # Note: current implementation sums over both (i,j) and (j,i), so we expect 2 * 22.5 = 45.0
        self.assertAlmostEqual(energy.item(), 45.0, places=5)
        self.assertEqual(stats["clash/num_clashes"].item(), 2.0)

    def test_covalent_energy(self):
        """Test simplified covalent energy (CA-CA distance)."""
        # Atoms at (0,0,0) and (5.0,0,0) -> distance 5.0
        # Ideal distance 3.8, tolerance 0.5
        # deviation = 5.0 - 3.8 = 1.2
        # penalty = (1.2 - 0.5)^2 = 0.7^2 = 0.49
        # scale = 5.0 -> energy = 2.45

        coords = torch.tensor([[[0.0, 0.0, 0.0],
                                [5.0, 0.0, 0.0]]])
        mask = torch.ones(1, 2)

        prior = GeometricPrior(
            covalent_penalty_scale=5.0,
            use_clash=False
        )

        energy, stats = prior(coords, mask)
        self.assertAlmostEqual(energy.item(), 2.45, places=5)


if __name__ == '__main__':
    unittest.main()
