#!/bin/bash
#SBATCH --job-name=simpleNOEfold_inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --qos=alberto.perezant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Load modules and activate environment
module load conda cuda/12.9.1
conda activate /orange/alberto.perezant/imesh.ranaweera/softwares/simpleNOEfold/ml-simplefold/simplefold_env

# Set PYTHONPATH
export PYTHONPATH=src/simplefold:src

# Parameters
SEQUENCE="MKTIIALSYIFCLVFADYKDDDDG" # Replace with your sequence
NEF_PATH="/path/to/your/restraints.nef" # Replace with your NEF path
OUTPUT_DIR="outputs/inference_run"
STEERING_WEIGHT=5.0
NSAMPLES=10

# Run inference
python src/simplefold/inference_nef.py \
    --sequence "$SEQUENCE" \
    --nef_path "$NEF_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --steering_weight $STEERING_WEIGHT \
    --nsamples $NSAMPLES \
    --plddt \
    --ckpt_path "/orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/artifacts/simplefold_100M.ckpt"
