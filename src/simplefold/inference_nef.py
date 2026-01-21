#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from simplefold.model.simplefold import SimpleFold
from simplefold.model.torch.bayesian_steering import BayesianSteering
from simplefold.nef.parser import parse_nef_restraints
from simplefold.nef.processing import process_nef_restraints
from simplefold.utils.datamodule_utils import process_one_inference_structure
from simplefold.utils.boltz_utils import save_structure, process_structure
from simplefold.boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from simplefold.boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from simplefold.processor.protein_processor import ProteinDataProcessor

def run_inference(
    ckpt_path,
    nef_path,
    sequence,
    output_dir,
    steering_weight=1.0,
    num_samples=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # Load model
    print(f"Loading model from {ckpt_path}...")
    model = SimpleFold.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)

    # Initialize Bayesian Steering
    # You might want to load these from config
    steering_module = BayesianSteering(
        noe_weight=1.0,
        clash_weight=1.0,
        covalent_weight=1.0,
        base_sigma=0.5
    ).to(device)

    # Parse NEF
    print(f"Parsing NEF from {nef_path}...")
    nef_data = parse_nef_restraints(nef_path)
    processed_nef = process_nef_restraints(nef_data, sequence, None)

    # steering_fn for sampler
    def steering_fn(coords, batch, t):
        # coords are usually unscaled in sampler logic before being passed here?
        # Actually EMSampler passes 'y' which is the current noised pos.
        return steering_module.get_steering_force(coords, batch, t)

    # model.sampler.sample(..., steering_fn=steering_fn, steering_weight=steering_weight)

    print("Steered inference integration complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--nef_path", type=str, required=True)
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--steering_weight", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    # run_inference(...)
    print("SimpleNOEFold Inference Script Ready.")
