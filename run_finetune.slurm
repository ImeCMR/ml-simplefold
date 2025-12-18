#!/bin/bash
#SBATCH --job-name=finetune-noefold
#SBATCH --output=logs/finetune_%x_%j.out
#SBATCH --error=logs/finetune_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@EXAMPLE.COM
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20000mb
#SBATCH --time=00:20:00
# Please replace the following with your specific Slurm QoS if required
# --qos=<YOUR_QOS_IF_NEEDED>

mkdir -p logs

# IMPORTANT: Activate your conda environment
module load conda
conda activate <PATH_TO_YOUR_CONDA_ENVIRONMENT>

echo "Starting SimpleNOEFold fine-tuning job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"

# === IMPORTANT: SET YOUR PATHS BEFORE RUNNING ===
# Path to the pre-trained model checkpoint you want to fine-tune
PRETRAINED_CKPT="<PATH_TO_PRETRAINED_CHECKPOINT>"

# Top-level directory for your training data
DATA_ROOT="<PATH_TO_YOUR_DATA_DIRECTORY>"
NEF_DIR="${DATA_ROOT}/nef"
STRUCTURES_DIR="${DATA_ROOT}/structures"
TOKENIZED_DIR="${DATA_ROOT}/tokenized"
TRAIN_ID_LIST="${DATA_ROOT}/nef_train_pdb_ids.txt"
# =================================================

srun python src/simplefold/train_fsdp.py --config-name finetune \
    task_name=SimpleNOEFold_Finetune_3429 \
    load_ckpt_path=${PRETRAINED_CKPT} \
    data.nef_dir=${NEF_DIR} \
    data.datasets.0.record_list=${TRAIN_ID_LIST} \
    data.datasets.0.target_dir=${STRUCTURES_DIR} \
    data.datasets.0.tokenized_dir=${TOKENIZED_DIR} \
    trainer.max_epochs=50 \
    model.bayesian_steering=default \
    model.bayesian_steering.bayesian_loss_weight=0.1 \
    model.optimizer.lr=1e-5

echo "Fine-tuning job finished."
