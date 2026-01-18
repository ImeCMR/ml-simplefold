#!/bin/bash
#SBATCH --job-name=simpleNOEfold_finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --qos=alberto.perezant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=24:00:00

# Load modules and activate environment
module load conda cuda/12.9.1
conda activate /orange/alberto.perezant/imesh.ranaweera/softwares/simpleNOEfold/ml-simplefold/simplefold_env

# Distributed training environment variables for FSDP
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS

# Define output directory
OUTPUT_DIR="outputs/finetune_bayesian"
mkdir -p $OUTPUT_DIR

# Run training
srun python src/simplefold/train_fsdp.py \
    --config-name finetune_bayesian \
    data=nef_train \
    trainer.devices=auto \
    +trainer.num_nodes=1 \
    model.bayesian_beta_start=0.1 \
    model.bayesian_beta_end=0.5 \
    trainer.max_steps=50000 \
    paths.output_dir=$OUTPUT_DIR \
    load_ckpt_path="/orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/artifacts/simplefold_100M.ckpt"
