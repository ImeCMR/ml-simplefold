#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import torch
import hydra
import omegaconf
import argparse
import numpy as np
from copy import deepcopy
from pathlib import Path
from itertools import starmap
import lightning.pytorch as pl

from model.flow import LinearPath
from model.torch.sampler import NMRGuidedSampler

from processor.protein_processor import ProteinDataProcessor
from utils.datamodule_utils import process_one_inference_structure
from utils.esm_utils import _af2_to_esm, esm_registry
from utils.boltz_utils import process_structure, save_structure
from utils.fasta_utils import process_fastas, download_fasta_utilities, check_fasta_inputs
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer

from nmr.nmr_restraints import NMRGuidanceEnergy, NMRRestraintParser

try: 
    import mlx.core as mx
    from mlx.utils import tree_unflatten, tree_flatten
    from model.mlx.sampler import EMSampler as EMSamplerMLX
    from model.mlx.esm_network import ESM2 as ESM2MLX
    from utils.mlx_utils import map_torch_to_mlx, map_plddt_torch_to_mlx
    MLX_AVAILABLE = True
except:
    MLX_AVAILABLE = False
    print("MLX not installed, skip importing MLX related packages.")


ckpt_url_dict = {
    "simplefold_100M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_100M.ckpt",
    "simplefold_360M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_360M.ckpt",
    "simplefold_700M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_700M.ckpt",
    "simplefold_1.1B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_1.1B.ckpt",
    "simplefold_1.6B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_1.6B.ckpt",
    "simplefold_3B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_3B.ckpt",
}

plddt_ckpt_url = "https://ml-site.cdn-apple.com/models/simplefold/plddt_module_1.6B.ckpt"


def initialize_folding_model(args):
    # define folding model
    simplefold_model = args.simplefold_model

    # create checkpoint directory
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, f"{simplefold_model}.ckpt")

    # create folding model
    ckpt_path = os.path.join(ckpt_dir, f"{simplefold_model}.ckpt")
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_dir, exist_ok=True)
        os.system(f"curl -L {ckpt_url_dict[simplefold_model]} -o {ckpt_path}")
    cfg_path = os.path.join("configs/model/architecture", f"foldingdit_{simplefold_model[11:]}.yaml")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # load model checkpoint
    if args.backend == 'torch':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config = omegaconf.OmegaConf.load(cfg_path)
        model = hydra.utils.instantiate(model_config)
        model.load_state_dict(checkpoint, strict=True)
        model = model.to(device)
    elif args.backend == 'mlx':
        device = "cpu"
        # replace torch implementations with mlx
        with open(cfg_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace('torch', 'mlx')

        model_config = omegaconf.OmegaConf.create(yaml_str)
        model = hydra.utils.instantiate(model_config)
        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, checkpoint.items()) if k is not None}
        model.update(tree_unflatten(list(mlx_state_dict.items())))
    print(f"Folding model {simplefold_model} loaded.")

    model.eval()
    return model, device


def initialize_plddt_module(args, device):
    if not args.plddt:
        return None, None

    # load pLDDT module if specified
    plddt_ckpt_path = os.path.join(args.ckpt_dir, "plddt.ckpt")
    if not os.path.exists(plddt_ckpt_path):
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.system(f"curl -L {plddt_ckpt_url} -o {plddt_ckpt_path}")

    plddt_module_path = "configs/model/architecture/plddt_module.yaml"
    plddt_checkpoint = torch.load(plddt_ckpt_path, map_location="cpu", weights_only=False)

    if args.backend == "torch":
        plddt_config = omegaconf.OmegaConf.load(plddt_module_path)
        plddt_out_module = hydra.utils.instantiate(plddt_config)
        plddt_out_module.load_state_dict(plddt_checkpoint, strict=True)
        plddt_out_module = plddt_out_module.to(device)
    elif args.backend == "mlx":
        # replace torch implementations with mlx
        with open(plddt_module_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace('torch', 'mlx')

        plddt_config = omegaconf.OmegaConf.create(yaml_str)
        plddt_out_module = hydra.utils.instantiate(plddt_config)

        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_plddt_torch_to_mlx, plddt_checkpoint.items()) if k is not None}
        plddt_out_module.update(tree_unflatten(list(mlx_state_dict.items())))

    plddt_out_module.eval()
    print(f"pLDDT output module loaded with {args.backend} backend.")

    plddt_latent_ckpt_path = os.path.join(args.ckpt_dir, "simplefold_1.6B.ckpt")
    if not os.path.exists(plddt_latent_ckpt_path):
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.system(f"curl -L {ckpt_url_dict['simplefold_1.6B']} -o {plddt_latent_ckpt_path}")

    plddt_latent_config_path = "configs/model/architecture/foldingdit_1.6B.yaml"
    plddt_latent_checkpoint = torch.load(plddt_latent_ckpt_path, map_location="cpu", weights_only=False)

    if args.backend == "torch":
        plddt_latent_config = omegaconf.OmegaConf.load(plddt_latent_config_path)
        plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)
        plddt_latent_module.load_state_dict(plddt_latent_checkpoint, strict=True)
        plddt_latent_module = plddt_latent_module.to(device)
    elif args.backend == "mlx":
        # replace torch implementations with mlx
        with open(plddt_latent_config_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace('torch', 'mlx')

        plddt_latent_config = omegaconf.OmegaConf.create(yaml_str)
        plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)
        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, plddt_latent_checkpoint.items()) if k is not None}
        plddt_latent_module.update(tree_unflatten(list(mlx_state_dict.items())))

    plddt_latent_module.eval()
    print(f"pLDDT latent module loaded with {args.backend} backend.")

    return plddt_latent_module, plddt_out_module


def initialize_esm_model(args, device):
    # load ESM2 model
    esm_model, esm_dict = esm_registry["esm2_3B"]()
    af2_to_esm = _af2_to_esm(esm_dict)

    if args.backend == 'torch':
        esm_model = esm_model.to(device)
        af2_to_esm = af2_to_esm.to(device)
    elif args.backend == 'mlx':
        esm_model_mlx = ESM2MLX(num_layers=36, embed_dim=2560, attention_heads=40)
        esm_state_dict_torch = esm_model.cpu().state_dict()

        esm_state_dict_torch = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, esm_state_dict_torch.items()) if k is not None}
        esm_model_mlx.update(tree_unflatten(list(esm_state_dict_torch.items())))
        esm_model = esm_model_mlx
    print(f"pLM ESM-3B loaded with {args.backend} backend.")

    esm_model.eval()
    return esm_model, esm_dict, af2_to_esm

#def parse_nmr_data(args):
#    """Load NMR restraints if provided"""
#    if not hasattr(args, 'noe_file') or args.noe_file is None:
#        return None, None
#    
#    
#    distance_groups = []
#    torsion_groups = []
#    
#    if args.noe_file and os.path.exists(args.noe_file):
#        distance_groups = NMRRestraintParser.parse_noe_file(args.noe_file)
#    
#    
#    if hasattr(args, 'talos_file') and args.talos_file and os.path.exists(args.talos_file):
#        torsion_groups = NMRRestraintParser.parse_talos_file(args.talos_file)
#    
#    if not distance_groups and not torsion_groups:
#        return None, None
#    
#    return distance_groups, torsion_groups


def parse_nmr_data(args):
    """Load NMR restraints if provided"""
    if not hasattr(args, 'noe_file') or args.noe_file is None:
        return None, None
    
    distance_groups = []
    torsion_groups = []
    
    # === NOE LOADING ===
    if args.noe_file and os.path.exists(args.noe_file):
        print(f"Loading NOE file: {args.noe_file}")
        distance_groups = NMRRestraintParser.parse_noe_file(args.noe_file)
    else:
        print(f"⚠️  NOE file not found or not provided")
    
    # === TORSION LOADING (ADD DIAGNOSTICS) ===
    print(f"\n=== TORSION DEBUG ===")
    print(f"hasattr(args, 'talos_file'): {hasattr(args, 'talos_file')}")
    
    if hasattr(args, 'talos_file'):
        print(f"args.talos_file: {args.talos_file}")
        print(f"File exists: {os.path.exists(args.talos_file) if args.talos_file else 'N/A'}")
        
        if args.talos_file and os.path.exists(args.talos_file):
            print(f"✓ Loading TALOS file: {args.talos_file}")
            torsion_groups = NMRRestraintParser.parse_talos_file(args.talos_file)
        else:
            print(f"✗ TALOS file check failed")
            if args.talos_file:
                print(f"  File does not exist at: {args.talos_file}")
            else:
                print(f"  args.talos_file is None or empty")
    else:
        print(f"✗ args.talos_file attribute not found")
    print(f"===================\n")
    
    if not distance_groups and not torsion_groups:
        return None, None
    
    return distance_groups, torsion_groups


def initialize_nmr_guided_sampler(args, device, batch):
    """Initialize sampler with or without NMR guidance"""
    
    # Parse NMR data
    distance_groups, torsion_groups = parse_nmr_data(args)

    #==================================== Debug prints ==============================
    print(f"\n{'='*60}")
    print("NMR Initialization")
    print(f"{'='*60}")
    print(f"Distance groups loaded: {len(distance_groups) if distance_groups else 0}")
    print(f"Torsion groups loaded: {len(torsion_groups) if torsion_groups else 0}")
    #================================================================================
    
    # Create NMR energy computer if restraints provided
    nmr_energy = None
    if distance_groups or torsion_groups:
        n_dist_active = int(len(distance_groups) * args.nmr_activation_fraction) if distance_groups else 0
        n_tors_active = int(len(torsion_groups) * args.nmr_activation_fraction) if torsion_groups else 0
        
        #==================================== Debug prints ==============================
        print(f"Active distance groups: {n_dist_active}/{len(distance_groups)}")
        print(f"Active torsion groups: {n_tors_active}/{len(torsion_groups)}")
        #================================================================================

        nmr_energy = NMRGuidanceEnergy(
            distance_groups=distance_groups or [],
            torsion_groups=torsion_groups or [],
            batch=batch,
            n_distance_active=n_dist_active,
            n_torsion_active=n_tors_active,
            device=device
        )

        #======================================= Debug prints ==============================
        print(f"Atom index map size: {len(nmr_energy.atom_index_map)}")
        print(f"Sample mappings:")
        for i, (key, val) in enumerate(list(nmr_energy.atom_index_map.items())[:5]):
            print(f"  ({key[0]}, '{key[1]}') -> atom {val}")
        #===================================================================================


        print(f"NMR guidance enabled: {len(distance_groups)} distance groups, {len(torsion_groups)} torsion groups")
    
    # Create sampler with NMR guidance
    sampler = NMRGuidedSampler(
        num_timesteps=args.num_steps,
        t_start=1e-4,
        tau=args.tau,
        log_timesteps=True,
        w_cutoff=0.99,
        nmr_energy=nmr_energy,
        guidance_scale=args.nmr_guidance_scale,
        guidance_start_t=args.nmr_start_t,
        guidance_end_t=args.nmr_end_t,
        guidance_schedule=args.nmr_schedule
    )
    
    #=================================== Debug prints ==============================
    print(f"\nSampler Configuration:")
    print(f"  Guidance scale (λ): {args.nmr_guidance_scale}")
    print(f"  Time window: [{args.nmr_start_t}, {args.nmr_end_t}]")
    print(f"  Schedule: {args.nmr_schedule}")
    print(f"  Tau (stochasticity): {args.tau}")
    print(f"{'='*60}\n")
    #===============================================================================
    return sampler

def initialize_others(args, device):
    """Modified to use NMR-guided sampler"""
    # prepare data tokenizer, featurizer, and processor
    tokenizer = BoltzTokenizer()
    featurizer = BoltzFeaturizer()
    processor = ProteinDataProcessor(
        device=device,
        scale=16.0,
        ref_scale=5.0,
        multiplicity=1,
        backend=args.backend,
    )
    
    # define flow process
    flow = LinearPath()
    
    # Note: sampler will be initialized per-structure in generate_structure()
    # because we need the batch to build atom index maps
    return tokenizer, featurizer, processor, flow, None

#def save_trajectory(trajectory, output_path, processor, batch, args):
#    """
#    Save flow matching trajectory.
#    
#    Args:
#        trajectory: List of coordinate tensors [timesteps, batch, n_atoms, 3]
#        output_path: Base path for saving
#        processor: Protein processor
#        batch: Batch dict
#        args: Command line arguments
#    """
#    #import numpy as np
#    
#    # Stack trajectory into single tensor
#    trajectory_tensor = torch.stack(trajectory)  # [T, batch, N_atoms, 3]
#    
#    # Post-process coordinates (scale back to Angstroms)
#    processed_trajectory = []
#    for t_idx in range(len(trajectory)):
#        coords = trajectory[t_idx].to(args.backend if args.backend == "torch" else "cpu")
#        # Apply same post-processing as final structure
#        coords_dict = {"denoised_coords": coords}
#        coords_processed = processor.postprocess(coords_dict, batch)
#        processed_trajectory.append(coords_processed["denoised_coords"].cpu().numpy())
#    
#    # Save as NPZ (compact format)
#    np.savez_compressed(
#        f"{output_path}_trajectory.npz",
#        coords=np.array(processed_trajectory),  # [T, N_atoms, 3]
#        timesteps=np.arange(len(trajectory)),
#        num_steps=args.num_steps,
#        stride=args.trajectory_stride if hasattr(args, 'trajectory_stride') else 1
#    )
#    
#    print(f"Saved trajectory with {len(trajectory)} frames to {output_path}_trajectory.npz")

def save_trajectory(trajectory, output_path, processor, batch, args, device):
    """
    Save flow matching trajectory.
    
    Args:
        trajectory: List of coordinate tensors [timesteps, batch, n_atoms, 3] on CPU
        output_path: Base path for saving
        processor: Protein processor
        batch: Batch dict (may be on CUDA)
        args: Command line arguments
        device: Device where batch originally lives
    """
    import numpy as np
    
    print(f"Processing trajectory with {len(trajectory)} frames...")
    
    # === FIX: Move batch to CPU for trajectory processing ===
    batch_cpu = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_cpu[key] = value.cpu()
        else:
            batch_cpu[key] = value

    # Post-process coordinates (scale back to Angstroms)
    processed_trajectory = []
    
    for t_idx in range(len(trajectory)):
        # === FIX: trajectory is already on CPU from sample_with_trajectory() ===
        # Just ensure it's on CPU (no need to check backend)
        coords = trajectory[t_idx]
        if isinstance(coords, torch.Tensor) and coords.device.type != 'cpu':
            coords = coords.cpu()
        
        # Apply same post-processing as final structure
        coords_dict = {"denoised_coords": coords}
        coords_processed = processor.postprocess(coords_dict, batch_cpu)
        processed_trajectory.append(coords_processed["denoised_coords"].cpu().numpy())
    
    # Save as NPZ (compact format)
    output_file = f"{output_path}_trajectory.npz"
    np.savez_compressed(
        output_file,
        coords=np.array(processed_trajectory),  # [T, N_atoms, 3]
        timesteps=np.arange(len(trajectory)),
        num_steps=args.num_steps,
        stride=getattr(args, 'trajectory_stride', 1)
    )
    
    # Print summary
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ Saved trajectory: {output_file}")
    print(f"  - Frames: {len(trajectory)}")
    print(f"  - Shape: {np.array(processed_trajectory).shape}")
    print(f"  - Size: {file_size_mb:.2f} MB")

def generate_structure(
    args, batch, sampler_template, flow, processor,
    model, plddt_latent_module, plddt_out_module, device
):
    """Modified to initialize NMR-guided sampler per structure and optionally save trajectory"""
    
    # Initialize sampler with NMR guidance for this specific structure
    sampler = initialize_nmr_guided_sampler(args, device, batch)
    
    # run inference for target protein
    coord_samples = []
    trajectory_samples = [] if args.save_trajectory else None
    pad_mask = batch["atom_pad_mask"]
    if args.plddt and plddt_latent_module is not None and plddt_out_module is not None:
        compute_plddt = True
        plddt_samples = []
    else:
        compute_plddt = False
        plddt_samples = None
    plddts = None
    
    #for _ in range(args.nsample_per_protein):
    for sample_idx in range(args.nsample_per_protein):
        if args.backend == "torch":
            noise = torch.randn_like(batch["coords"]).to(device)
        elif args.backend == "mlx":
            noise = mx.random.normal(batch["coords"].shape)
        
        # Choose sampling method based on trajectory flag
        if args.save_trajectory:
            stride = args.trajectory_stride if hasattr(args, 'trajectory_stride') else 1
            out_dict, trajectory = sampler.sample_with_trajectory(
                model, flow, noise, batch, stride=stride
            )
            trajectory_samples.append(trajectory)
        else:
            out_dict = sampler.sample(model, flow, noise, batch)
        
        if compute_plddt:
            if args.backend == "torch":
                t = torch.ones(batch['coords'].shape[0], device=device)
                out_feat = plddt_latent_module(
                    out_dict["denoised_coords"].detach(), t, batch
                )
                plddt_out_dict = plddt_out_module(
                    out_feat["latent"].detach(),
                    batch,
                )
            elif args.backend == "mlx":
                t = mx.ones(batch['coords'].shape[0])
                out_feat = plddt_latent_module(
                    out_dict["denoised_coords"], t, batch
                )
                plddt_out_dict = plddt_out_module(
                    out_feat["latent"],
                    batch,
                )
            plddt_samples.append(plddt_out_dict["plddt"] * 100.0)
        
        out_dict = processor.postprocess(out_dict, batch)
        if args.backend == "torch":
            coord_samples.append(out_dict["denoised_coords"].detach())
        else:
            coord_samples.append(out_dict["denoised_coords"])
    
    if args.backend == "torch":
        sampled_coord = torch.cat(coord_samples, dim=0)
        pad_mask = pad_mask.detach().repeat_interleave(
            args.nsample_per_protein, dim=0
        )
        if compute_plddt:
            plddts = torch.cat(plddt_samples, dim=0).detach()
    else:
        sampled_coord = mx.concatenate(coord_samples, axis=0)
        pad_mask = mx.concatenate(
            [pad_mask] * args.nsample_per_protein, axis=0
        )
        if compute_plddt:
            plddts = mx.concatenate(plddt_samples, axis=0)
    
    if args.save_trajectory:
        return sampled_coord, pad_mask, plddts, trajectory_samples
    else:
        return sampled_coord, pad_mask, plddts


def predict_structures_from_fastas(args):
    # create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / f"predictions_{args.simplefold_model}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    if args.backend == "mlx" and not MLX_AVAILABLE:
        args.backend = "torch"
        print("MLX not available, switch to torch backend.")

    # initialize models
    model, device = initialize_folding_model(args)
    plddt_latent_module, plddt_out_module = initialize_plddt_module(args, device)
    esm_model, esm_dict, af2_to_esm = initialize_esm_model(args, device)

    # initialize other components
    tokenizer, featurizer, processor, flow, sampler = initialize_others(args, device)

    # process fasta files to input format
    download_fasta_utilities(cache)
    data = check_fasta_inputs(Path(args.fasta_path))
    if not data:
        raise ValueError("No valid input files found. Please check the input directory.")
    process_fastas(
        data=data,
        out_dir=output_dir,
        ccd_path=cache / "ccd.pkl",
    )

    for struct_file in output_dir.glob("structures/*.npz"):
        record_file = output_dir / "records" / f"{struct_file.stem}.json"

        # prepare the target protein data for inference
        batch, structure, record = process_one_inference_structure(
            struct_file, record_file,
            tokenizer, featurizer, processor,
            esm_model, esm_dict, af2_to_esm,
        )

        # Generate structures (with or without trajectory)
        if args.save_trajectory:
            sampled_coord, pad_mask, plddts, trajectory_samples = generate_structure(
                args, batch, sampler, flow, processor,
                model, plddt_latent_module, plddt_out_module, device
            )
        
        else:
            sampled_coord, pad_mask, plddts = generate_structure(
                args, batch, sampler, flow, processor,
                model, plddt_latent_module, plddt_out_module, device
            )

        for i in range(args.nsample_per_protein):
            sampled_coord_i = sampled_coord[i]
            pad_mask_i = pad_mask[i]

            # save the generated structure
            structure_save = process_structure(
                deepcopy(structure), sampled_coord_i, pad_mask_i, record, backend=args.backend
            )
            outname = f"{record.id}_sampled_{i}"
            save_structure(
                structure_save, prediction_dir, outname,
                output_format=args.output_format,
                plddts=plddts[i] if plddts is not None else None
            )

            # Save trajectory if requested
            if args.save_trajectory:
                save_trajectory(
                    trajectory_samples[i],
                    prediction_dir / outname,
                    processor,
                    batch,
                    args,
                    device
                )