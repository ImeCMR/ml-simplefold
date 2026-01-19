#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""
NEF-Guided Inference Script for SimpleNOEFold
"""

import os
import torch
import argparse
import numpy as np
import json
from pathlib import Path
from copy import deepcopy
import lightning.pytorch as pl

from model.flow import LinearPath
from model.torch.sampler import EMSampler
from model.torch.bayesian_steering import BayesianSteering
from processor.protein_processor import ProteinDataProcessor
from utils.datamodule_utils import collate
from utils.esm_utils import _af2_to_esm, esm_registry
from utils.boltz_utils import process_structure, save_structure
from utils.fasta_utils import process_fastas, download_fasta_utilities
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from boltz_data_pipeline.types import Input, Record, Structure
from nef.processing import process_nef_restraints
from inference import initialize_folding_model, initialize_plddt_module, initialize_esm_model

def run_inference_nef(args):
    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions_nef"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # Set seed
    pl.seed_everything(args.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize models
    print("Initializing models...")
    # Support direct path to checkpoint
    if args.ckpt_path:
        # We still use initialize_folding_model to get the right config,
        # but we'll override the loaded weights if a direct path is provided.
        model, _ = initialize_folding_model(args)
        print(f"Overriding weights with local checkpoint: {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        # Handle state_dict if it's a Lightning checkpoint
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
            # Remove 'model.' prefix if present
            checkpoint = {k.replace("model.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=False)
    else:
        model, _ = initialize_folding_model(args)

    plddt_latent_module, plddt_out_module = initialize_plddt_module(args, device)
    esm_model, esm_dict, af2_to_esm = initialize_esm_model(args, device)
    model = model.to(device).eval()

    # 2. Setup Data Pipeline
    tokenizer = BoltzTokenizer()
    featurizer = BoltzFeaturizer()
    processor = ProteinDataProcessor(device=device)
    flow = LinearPath()
    sampler = EMSampler(num_timesteps=args.num_steps, tau=args.tau, log_timesteps=True)

    # 3. Process Sequence (Create dummy structure)
    print(f"Processing sequence: {args.sequence[:20]}...")
    # Create a temporary FASTA
    fasta_path = cache / "temp.fasta"
    with open(fasta_path, "w") as f:
        f.write(f">target\n{args.sequence}\n")
    
    download_fasta_utilities(cache)
    process_fastas(data=[("target", str(fasta_path))], out_dir=output_dir, ccd_path=cache / "ccd.pkl")
    
    struct_file = output_dir / "structures" / "target.npz"
    record_file = output_dir / "records" / "target.json"
    
    structure = Structure.load(struct_file)
    input_data = Input(structure, {})
    with open(record_file) as f:
        record_dict = json.load(f)
    record = Record(**record_dict)

    tokenized = tokenizer.tokenize(input_data)
    features = featurizer.process(tokenized)
    features["aa_seq"] = args.sequence
    features["record"] = record_dict
    features["num_repeats"] = torch.tensor(1)
    features['max_num_tokens'] = torch.tensor(len(tokenized.tokens), dtype=torch.long)
    features['cropped_num_tokens'] = torch.tensor(len(tokenized.tokens), dtype=torch.long)

    # 4. Integrate NEF Restraints
    print(f"Loading NMR restraints from: {args.nef_path}")
    nef_path = Path(args.nef_path)
    # We pass the directory where the NEF is, and the record_id
    features = process_nef_restraints("target", nef_path.parent, tokenized, features)
    
    if "noe_at1_idx" not in features:
        print("Warning: No NMR restraints were successfully mapped. Proceeding with standard inference.")
    else:
        num_restr = features["noe_at1_idx"].shape[0]
        print(f"Successfully mapped {num_restr} NMR restraints.")

    # 5. Prepare Batch
    batch = collate([features])
    batch = processor.preprocess_inference(batch, esm_model, esm_dict, af2_to_esm)

    # 6. Initialize Bayesian Steering
    steering = BayesianSteering(
        use_noe="noe" in args.steering_type,
        use_geom="geom" in args.steering_type,
        geometric_clash_penalty=args.geom_penalty,
    ).to(device)

    def steering_fn(coords, t):
        # Coordinates in the sampler are unscaled (standard deviation space)
        # BayesianSteering expects Angstroms
        coords_ang = coords * processor.scale
        force = steering.compute_steering_force(coords_ang, batch, t)
        # We need to scale the force back for the drift term
        return force * processor.scale

    # 7. Generate structures with steering
    print(f"Generating {args.nsamples} samples with steering weight {args.steering_weight}...")
    
    for i in range(args.nsamples):
        noise = torch.randn_like(batch["coords"]).to(device)

        # Modified sample call that supports steering
        out_dict = sampler.sample(
            model, flow, noise, batch,
            steering_fn=steering_fn,
            steering_weight=args.steering_weight
        )

        # pLDDT calculation
        if plddt_latent_module and plddt_out_module:
            t_ones = torch.ones(batch['coords'].shape[0], device=device)
            out_feat = plddt_latent_module(out_dict["denoised_coords"].detach(), t_ones, batch)
            plddt_out = plddt_out_module(out_feat["latent"].detach(), batch)
            plddt = plddt_out["plddt"][0] * 100.0
        else:
            plddt = None

        # Post-process and save
        out_dict = processor.postprocess(out_dict, batch)
        sampled_coords = out_dict["denoised_coords"][0]
        pad_mask = batch["atom_pad_mask"][0]

        structure_save = process_structure(deepcopy(structure), sampled_coords, pad_mask, record)
        outname = f"target_nef_sample_{i}"
        save_structure(structure_save, prediction_dir, outname, output_format="pdb", plddts=plddt)
        print(f"Saved sample {i} to {prediction_dir}/{outname}.pdb")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True, help="Protein sequence")
    parser.add_argument("--nef_path", type=str, required=True, help="Path to NEF file")
    parser.add_argument("--output_dir", type=str, default="outputs/inference_nef")
    parser.add_argument("--simplefold_model", type=str, default="simplefold_100M")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Direct path to checkpoint file")
    parser.add_argument("--ckpt_dir", type=str, default="artifacts/")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--steering_weight", type=float, default=1.0)
    parser.add_argument("--steering_type", type=str, default="noe_geom")
    parser.add_argument("--geom_penalty", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plddt", action="store_true", help="Enable pLDDT calculation")
    parser.add_argument("--backend", type=str, default="torch")
    
    args = parser.parse_args()
    run_inference_nef(args)
