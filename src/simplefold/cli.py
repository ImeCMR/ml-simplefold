#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import argparse
from simplefold import __version__
from simplefold.inference import predict_structures_from_fastas

def main():
    parser = argparse.ArgumentParser(
        prog="simplefold",
        description="Folding proteins with SimpleFold."
    )
    
    # === EXISTING SIMPLEFOLD ARGUMENTS ===
    parser.add_argument("--simplefold_model", type=str, default="simplefold_100M", 
                       help="Name of the model to load.")
    parser.add_argument("--ckpt_dir", type=str, default="artifacts", 
                       help="Directory to save the checkpoint.")
    parser.add_argument("--output_dir", type=str, default="artifacts/debug_samples", 
                       help="Directory to save the output structure.")
    parser.add_argument("--num_steps", type=int, default=500, 
                       help="Number of steps in inference.")
    parser.add_argument("--tau", type=float, default=0.1, 
                       help="Diffusion coefficient scaling factor.")
    parser.add_argument("--no_log_timesteps", action="store_true", 
                       help="Disable logarithmic timesteps.")
    parser.add_argument("--fasta_path", required=True, type=str, 
                       help="Path to the input FASTA file/directory.")
    parser.add_argument("--nsample_per_protein", type=int, default=1, 
                       help="Number of samples to generate per protein.")
    parser.add_argument("--plddt", action="store_true", 
                       help="Enable pLDDT prediction.")
    parser.add_argument("--output_format", type=str, default="mmcif", 
                       choices=["pdb", "mmcif"], help="Output file format.")
    parser.add_argument("--backend", type=str, default='torch', 
                       choices=['torch', 'mlx'], 
                       help="Backend to run inference either torch or mlx")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility.")
    
    # === NEW NMR GUIDANCE ARGUMENTS ===
    parser.add_argument("--noe_file", type=str, default=None,
                       help="Path to NOE.dat file (MELD format) containing distance restraints. Optional.")
    parser.add_argument("--talos_file", type=str, default=None,
                       help="Path to rotamers.dat file (MELD/TALOS format) containing torsion restraints. Optional.")
    parser.add_argument("--nmr_guidance_scale", type=float, default=1.0,
                       help="Strength of NMR guidance (lambda). Higher values = stronger enforcement of restraints. "
                            "Recommended range: 0.5-3.0. Default: 1.0")
    parser.add_argument("--nmr_activation_fraction", type=float, default=0.85,
                       help="Fraction of restraint groups to activate (MELD-style selective activation). "
                            "Range: 0.0-1.0. Default: 0.85 (85%% of restraints active)")
    parser.add_argument("--nmr_start_t", type=float, default=0.1,
                       help="Start applying NMR guidance at this timestep in the diffusion process. "
                            "Range: 0.0-1.0. Default: 0.1 (early structure formation)")
    parser.add_argument("--nmr_end_t", type=float, default=0.8,
                       help="Stop applying NMR guidance at this timestep. "
                            "Range: 0.0-1.0. Default: 0.8 (before final refinement)")
    parser.add_argument("--nmr_schedule", type=str, default='cosine',
                       choices=['linear', 'cosine', 'constant'],
                       help="Schedule for ramping guidance weight over time. "
                            "'cosine': smooth ramp (recommended), 'linear': linear ramp, 'constant': fixed weight")
    # === NEW: MELD DYNAMIC RESELECTION ===
    parser.add_argument("--nmr_reselect_every", type=int, default=1,
                        help="Re-select active restraints every N steps (MELD dynamic selection). "
                        "Default: 1 steps")
    parser.add_argument("--save_trajectory", action="store_true",
                    help="Save flow matching trajectory at each timestep")
    parser.add_argument("--trajectory_stride", type=int, default=1,
                    help="Save every N-th frame (default: 1 = all frames)")
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    args = parser.parse_args()
    print(f"Running protein folding with SimpleFold ...")
    
    # Print NMR settings if provided
    if args.noe_file or args.talos_file:
        print(f"\n=== NMR Guidance Enabled ===")
        if args.noe_file:
            print(f"  NOE file: {args.noe_file}")
        if args.talos_file:
            print(f"  TALOS file: {args.talos_file}")
        print(f"  Guidance scale: {args.nmr_guidance_scale}")
        print(f"  Activation fraction: {args.nmr_activation_fraction}")
        print(f"  Reselection frequency: every {args.nmr_reselect_every} steps")
        print(f"  Time window: [{args.nmr_start_t}, {args.nmr_end_t}]")
        print(f"  Schedule: {args.nmr_schedule}")
        print("="*30 + "\n")
    
    predict_structures_from_fastas(args)

if __name__ == "__main__":
    main()