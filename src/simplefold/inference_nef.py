#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import torch
import argparse
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import hydra

from nef.processing import process_nef_restraints
from model.torch.bayesian_steering import BayesianSteering
from utils.boltz_utils import process_structure, save_structure
from boltz_data_pipeline.types import Record, Structure
from boltz_data_pipeline.const import vdw_radii
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from model.simplefold import SimpleFold

def get_covalent_blueprint(structure, tokenizer):
    """
    Generates bond connectivity and ideal lengths from structure and tokenizer.
    """
    atoms = structure.atoms
    # Standard bonds from tokenizer
    standard_bonds = tokenizer.standard_bonds # Dict of residue_name -> list of (atom1, atom2)

    connectivity = []
    ideal_lengths = []

    # We'll use a simple approach: for each residue, add standard bonds
    residues = structure.residues
    for i, res in enumerate(residues):
        res_name = res['name']
        if res_name in standard_bonds:
            for a1_name, a2_name in standard_bonds[res_name]:
                # Find atom indices in structure for this residue
                idx1 = np.where((atoms['res_idx'] == i) & (atoms['name'] == a1_name))[0]
                idx2 = np.where((atoms['res_idx'] == i) & (atoms['name'] == a2_name))[0]
                if len(idx1) > 0 and len(idx2) > 0:
                    connectivity.append([idx1[0], idx2[0]])
                    # For simplicity, use 1.5A as default ideal bond length if not specified
                    ideal_lengths.append(1.5)

        # Add peptide bond to next residue
        if i < len(residues) - 1:
            # C of residue i to N of residue i+1
            idx_c = np.where((atoms['res_idx'] == i) & (atoms['name'] == 'C'))[0]
            idx_n = np.where((atoms['res_idx'] == i+1) & (atoms['name'] == 'N'))[0]
            if len(idx_c) > 0 and len(idx_n) > 0:
                connectivity.append([idx_c[0], idx_n[0]])
                ideal_lengths.append(1.33)

    return np.array(connectivity), np.array(ideal_lengths)

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Bayesian-Steered SimpleNOEFold Inference")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to SimpleFold checkpoint")
    parser.add_argument("--config_path", type=str, default="configs/inference.yaml", help="Path to config file")
    parser.add_argument("--sequence", type=str, help="Amino acid sequence")
    parser.add_argument("--nef_path", type=str, help="Path to NEF restraints file")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Output directory")
    parser.add_argument("--steering_weight", type=float, default=5.0, help="Gamma_0 for Bayesian steering")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Model
    print(f"Loading model from {args.ckpt_path}...")
    model = SimpleFold.load_from_checkpoint(args.ckpt_path, map_location=args.device, strict=False)
    model.to(args.device)
    model.eval()

    # 2. Prepare Data
    print("Preparing input data...")
    record_id = Path(args.nef_path).stem if args.nef_path else "query"
    processor = model.processor

    path = Path(args.data_dir) / "structures" / f"{record_id}.npz"
    if not path.exists():
        print(f"Error: Structure data not found at {path}.")
        return

    structure: Structure = Structure.load(path)
    record_path = Path(args.data_dir) / "records" / f"{record_id}.json"
    if record_path.exists():
        import json
        with open(record_path, 'r') as f:
            record_dict = json.load(f)
        record = Record(**record_dict)
    else:
        record = Record(id=record_id, chains=[])

    # Create batch
    batch = {
        'coords': torch.from_numpy(structure.atoms['coords']).unsqueeze(0),
        'atom_pad_mask': torch.from_numpy(structure.mask).unsqueeze(0),
        'atom_resolved_mask': torch.from_numpy(structure.mask).unsqueeze(0),
        'ref_pos': torch.from_numpy(structure.atoms['coords']).unsqueeze(0),
        'ref_element': torch.zeros(1, len(structure.atoms)),
        'mol_type': torch.zeros(1, len(structure.residues)),
        'atom_to_token': torch.eye(len(structure.atoms)).unsqueeze(0),
        'record': [record.__dict__],
        'aa_seq': [args.sequence] if args.sequence else None,
        'res_type': torch.zeros(1, len(structure.residues)),
        'cropped_num_tokens': [len(structure.residues)],
    }

    # Covalent Blueprint & VDW Radii
    tokenizer = BoltzTokenizer()
    connectivity, lengths = get_covalent_blueprint(structure, tokenizer)
    batch["bond_connectivity"] = torch.from_numpy(connectivity).unsqueeze(0).to(args.device)
    batch["bond_ideal_lengths"] = torch.from_numpy(lengths).float().unsqueeze(0).to(args.device)
    batch["bond_mask"] = torch.ones_like(batch["bond_ideal_lengths"])

    # VDW Radii from const.vdw_radii
    radii = []
    for elem in structure.atoms['element']:
        radii.append(vdw_radii.get(elem, 1.5))
    batch["vdw_radii"] = torch.tensor(radii).float().unsqueeze(0).to(args.device)

    batch = processor.preprocess_inference(batch, esm_model=model.esm_model, esm_dict=model.esm_dict, af2_to_esm=model.af2_to_esm)

    # 3. Add NEF Restraints to batch
    if args.nef_path:
        print(f"Processing NEF restraints from {args.nef_path}...")
        restraints = process_nef_restraints(record_id, args.data_dir)
        if restraints:
            # Handle ambiguous restraints (List of lists)
            max_k = max(len(r) for r in restraints['res1'])
            num_r = len(restraints['res1'])

            noe_at1_idx = torch.zeros((1, num_r, max_k), dtype=torch.long)
            noe_at2_idx = torch.zeros((1, num_r, max_k), dtype=torch.long)
            noe_ambiguous_mask = torch.zeros((1, num_r, max_k))

            atoms = structure.atoms
            valid_r_count = 0

            for i in range(num_r):
                valid_k_count = 0
                for k in range(len(restraints['res1'][i])):
                    r1 = restraints['res1'][i][k]
                    r2 = restraints['res2'][i][k]
                    a1_name = restraints['atom1'][i][k]
                    a2_name = restraints['atom2'][i][k]

                    idx1 = np.where((atoms['res_idx'] == r1) & (atoms['name'] == a1_name))[0]
                    idx2 = np.where((atoms['res_idx'] == r2) & (atoms['name'] == a2_name))[0]

                    if len(idx1) > 0 and len(idx2) > 0:
                        noe_at1_idx[0, i, valid_k_count] = idx1[0]
                        noe_at2_idx[0, i, valid_k_count] = idx2[0]
                        noe_ambiguous_mask[0, i, valid_k_count] = 1.0
                        valid_k_count += 1

                if valid_k_count > 0:
                    valid_r_count += 1

            if valid_r_count > 0:
                batch["noe_at1_idx"] = noe_at1_idx.to(args.device)
                batch["noe_at2_idx"] = noe_at2_idx.to(args.device)
                batch["noe_upper_bounds"] = torch.tensor(restraints['upper_bounds'], device=args.device, dtype=torch.float32).unsqueeze(0)
                batch["noe_mask"] = (torch.sum(noe_ambiguous_mask, dim=-1) > 0).float().to(args.device)
                batch["noe_ambiguous_mask"] = noe_ambiguous_mask.to(args.device)
                print(f"Loaded {valid_r_count} restraints.")

    # 4. Setup Bayesian Steering
    steering = BayesianSteering(noe_weight=1.0, clash_weight=0.1, covalent_weight=0.1)
    steering.to(args.device)

    # 5. Steered Inference
    print(f"Starting steered sampling with weight {args.steering_weight}...")
    for s in range(args.num_samples):
        noise = torch.randn_like(batch['coords']).to(args.device)
        out_dict = model.sampler.sample(
            model.model_ema.module.forward, model.path, noise, batch,
            steering_fn=steering.get_steering_force,
            steering_weight=args.steering_weight
        )
        out_dict = processor.postprocess(out_dict, batch)
        sampled_coord = out_dict['denoised_coords'][0]
        pad_mask = batch['atom_pad_mask'][0]
        sampled_structure = process_structure(structure, sampled_coord, pad_mask, record)
        outname = f"{record_id}_steered_sample_{s}"
        save_structure(sampled_structure, Path(args.output_dir), outname, output_format="pdb")
        print(f"Saved sample {s} to {args.output_dir}/{outname}.pdb")

if __name__ == "__main__":
    main()
