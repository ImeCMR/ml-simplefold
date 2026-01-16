#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""
Unit tests for data pipeline modifications.
"""

import unittest
import torch
import json
import pickle
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from simplefold.datasets.train_datamodule import SimpleFoldTrainingDataset
from simplefold.utils.datamodule_utils import Dataset, DatasetConfig, collate
from boltz_data_pipeline.types import Tokenized, Record


class TestDataPipeline(unittest.TestCase):
    """Test NEF processing and collation in data pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.target_dir = Path(self.temp_dir) / "target"
        self.tokenized_dir = Path(self.temp_dir) / "tokenized"

        os.makedirs(self.target_dir / "structures")
        os.makedirs(self.target_dir / "nef")
        os.makedirs(self.tokenized_dir / "records")
        os.makedirs(self.tokenized_dir / "tokens")

        # Create a mock record
        self.record_id = "test_1"
        record_data = {"id": self.record_id, "chain_id": "A"}
        with open(self.tokenized_dir / "records" / f"{self.record_id}.json", "w") as f:
            json.dump(record_data, f)

        # Create a mock NEF file
        nef_content = """
        data_test
        save_nef_distance_restraint_list_1
           _nef_distance_restraint_list.sf_category  nef_distance_restraint_list
           loop_
              _nef_distance_restraint.index
              _nef_distance_restraint.chain_code_1
              _nef_distance_restraint.sequence_code_1
              _nef_distance_restraint.atom_name_1
              _nef_distance_restraint.chain_code_2
              _nef_distance_restraint.sequence_code_2
              _nef_distance_restraint.atom_name_2
              _nef_distance_restraint.upper_limit
              1  A  1  CA  A  2  CA  5.0
           stop_
        save_
        """
        with open(self.target_dir / "nef" / f"{self.record_id}.nef", "w") as f:
            f.write(nef_content)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_process_nef(self):
        """Test that process_nef correctly extracts restraints and maps atoms."""
        # Mock Tokenized object with nested structure mock
        tokenized = MagicMock()

        # Mock structure atoms and tokens
        # We need to match the indices in the NEF: A 1 CA and A 2 CA
        token1 = {
            "asym_id": 0, "res_idx": 1, "atom_num": 1, "atom_idx": 0
        }
        token2 = {
            "asym_id": 0, "res_idx": 2, "atom_num": 1, "atom_idx": 1
        }
        tokenized.tokens = [token1, token2]

        # atom name CA
        atom1 = {"name": [35, 33] + [0]*30}
        atom2 = {"name": [35, 33] + [0]*30}
        tokenized.structure.atoms = [atom1, atom2]
        tokenized.structure.chains = [{"name": "A"}]

        dataset_obj = MagicMock(spec=Dataset)
        # We need an actual instance to call process_nef
        # SimpleFoldTrainingDataset.__init__ is heavy, so we mock it or use it carefully
        ds = SimpleFoldTrainingDataset.__new__(SimpleFoldTrainingDataset)

        features = {}
        features = ds.process_nef(self.record_id, self.target_dir, tokenized, features)

        self.assertIn("noe_at1_idx", features)
        self.assertEqual(features["noe_at1_idx"].shape, (1, 1))
        self.assertEqual(features["noe_at1_idx"][0, 0].item(), 0) # Atom 0
        self.assertEqual(features["noe_at2_idx"][0, 0].item(), 1) # Atom 1
        self.assertEqual(features["noe_upper_bounds"][0].item(), 5.0)

    def test_collate_noe_padding(self):
        """Test that collate correctly pads NOE tensors."""
        # Batch of 2 with different number of restraints
        data = [
            {
                "noe_at1_idx": torch.tensor([[0]]), # K=1, M=1
                "noe_at2_idx": torch.tensor([[1]]),
                "noe_mask": torch.tensor([[1.0]]),
                "noe_upper_bounds": torch.tensor([5.0]),
                "noe_weights": torch.tensor([1.0]),
                "other": torch.tensor([1])
            },
            {
                "noe_at1_idx": torch.tensor([[0, 2], [1, 3]]), # K=2, M=2
                "noe_at2_idx": torch.tensor([[1, 3], [0, 2]]),
                "noe_mask": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
                "noe_upper_bounds": torch.tensor([4.0, 6.0]),
                "noe_weights": torch.tensor([1.0, 1.0]),
                "other": torch.tensor([2])
            }
        ]

        collated = collate(data)

        # max_K = 2, max_M = 2
        self.assertEqual(collated["noe_at1_idx"].shape, (2, 2, 2))
        self.assertEqual(collated["noe_upper_bounds"].shape, (2, 2))

        # Check padding
        self.assertEqual(collated["noe_at1_idx"][0, 1, 0].item(), 0) # Padded K
        self.assertEqual(collated["noe_mask"][0, 1, 0].item(), 0.0) # Padded mask should be 0


if __name__ == '__main__':
    unittest.main()
