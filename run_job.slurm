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

# --- This is the Final, Definitive srun Command ---
# It uses the correct 'model.bayesian_loss_weight' parameter.
srun --export=ALL --gpu-bind=single:1 python src/simplefold/train_fsdp.py --config-name finetune \
    task_name=SimpleNOEFold_Finetune_3429 \
    trainer.ckpt_path=/orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/artifacts/simplefold_100M.ckpt \
    data.nef_dir=/orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/data/data_train/nef \
    data.datasets.0.record_list=/orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/data/data_train/nef_train_pdb_ids.txt \
    data.datasets.0.target_dir=/orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/data/data_train/structures \
    data.datasets.0.tokenized_dir=/orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/data/data_train/tokenized \
    trainer.max_epochs=50 \
    model.bayesian_steering=default \
    model.bayesian_loss_weight=0.1 \
    model.optimizer.lr=1e-5

echo "Fine-tuning job finished."
