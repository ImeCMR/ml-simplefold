#!/bin/bash
#SBATCH --job-name=finetune-noefold
#SBATCH --output=logs/finetune_%x_%j.out
#SBATCH --error=logs/finetune_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=imeshr150@gmail.com
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20000mb
#SBATCH --time=00:20:00
#SBATCH --qos=alberto.perezant

mkdir -p logs

module load conda
conda activate /orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/simplefold

echo "Starting SimpleNOEFold fine-tuning job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"

# Set up environment variables for torch.distributed
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500 # A high port that is likely free
export WORLD_SIZE=$SLURM_NTASKS
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"


# --- This is the Final, Definitive srun Command ---
# It uses the correct 'model.bayesian_loss_weight' parameter.
# Each task launched by srun will have SLURM_PROCID set, which PyTorch Lightning
# will automatically use to set the RANK.
srun --export=ALL --gpu-bind=single:1 python src/simplefold/train_fsdp.py --config-name finetune \
    task_name=SimpleNOEFold_Finetune_3429 \
    trainer.ckpt_path=/orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/artifacts/simplefold_100M.ckpt \
    trainer.max_epochs=50 \
    model.optimizer.lr=1e-5 \
    model.bayesian_steering=default \
    model.bayesian_loss_weight=0.1 \
    data.nef_dir=/orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/data/data_train/nef \
    data.datasets="[{name: 'nmr_finetune_set', target_dir: /orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/data/data_train/structures, tokenized_dir: /orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/data/data_train/tokenized, record_list: /orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/data/data_train/nef_train_pdb_ids.txt, manifest_path: /orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/data/data_train/manifest.json, filters: null, esm_dir: null, connectivity_dir: null, cheap_dir: null, cropper: null, cluster: null}]"

echo "Fine-tuning job finished."
