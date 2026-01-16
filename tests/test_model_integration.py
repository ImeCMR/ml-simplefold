#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""
Integration tests for SimpleFold model with Bayesian loss.
"""

import unittest
import torch
from simplefold.model.simplefold import SimpleFold
from processor.protein_processor import ProteinDataProcessor


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.ones(1))
    def forward(self, noised_pos, t, feats):
        # Return predict_velocity same as noised_pos for simplicity
        return {"predict_velocity": torch.zeros_like(noised_pos), "latent": torch.ones(noised_pos.shape[0], 128)}

class MockPath:
    def interpolant(self, t, noise, coords):
        return None, coords, torch.zeros_like(coords)

class TestSimpleFoldIntegration(unittest.TestCase):
    """Test SimpleFold model with Bayesian Steering integration."""

    def setUp(self):
        self.processor = ProteinDataProcessor(device="cpu")
        # Disable ESM to avoid heavy downloads/memory usage in test
        self.model = SimpleFold(
            architecture=MockModel(),
            processor=lambda **kwargs: self.processor,
            loss=None,
            path=MockPath(),
            sampler=None,
            esm_model=None, # Disable ESM
            use_bayesian_loss=True,
            bayesian_loss_type="noe_geom",
            bayesian_beta_start=0.1,
            bayesian_beta_end=0.5
        )

    def test_bayesian_loss_init(self):
        """Test that Bayesian steering is initialized."""
        self.assertIsNotNone(self.model.bayesian_steering)
        self.assertTrue(self.model.bayesian_steering.use_noe)
        self.assertTrue(self.model.bayesian_steering.use_geom)

    def test_training_step_with_bayesian(self):
        """Test that training step runs and computes Bayesian loss."""
        # Minimal batch
        B, N = 2, 10
        batch = {
            "coords": torch.randn(B, N, 3),
            "atom_resolved_mask": torch.ones(B, N),
            "atom_pad_mask": torch.ones(B, N),
            "noe_at1_idx": torch.zeros((B, 1, 1), dtype=torch.long),
            "noe_at2_idx": torch.ones((B, 1, 1), dtype=torch.long),
            "noe_mask": torch.ones((B, 1, 1)),
            "noe_upper_bounds": torch.ones((B, 1)) * 2.0,
            "noe_weights": torch.ones((B, 1)),
            "noe_restraints_indices": torch.tensor(True)
        }

        # We need to mock processor.preprocess_training
        self.model.processor = MockProcessor()

        # Mocking self.log to avoid Lightning errors
        self.model.log = lambda *args, **kwargs: None

        # Mock trainer
        class MockTrainer:
            def __init__(self):
                self.global_step = 0
                self.current_epoch = 0
                self.world_size = 1
        self.model.trainer = MockTrainer()

        loss = self.model.flow_matching_train_step(batch, 0)

class MockProcessor:
    def __init__(self):
        self.scale = 1.0
    def preprocess_training(self, batch, **kwargs):
        return batch
    def __call__(self, device):
        return self


if __name__ == '__main__':
    unittest.main()
