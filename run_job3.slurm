#!/bin/bash
#SBATCH --job-name=finetune-noefold
#SBATCH --output=logs/finetune_%x_%j.out
#SBATCH --error=logs/finetune_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=imeshr150@gmail.com
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20000mb
#SBATCH --time=04:00:00
#SBATCH --qos=alberto.perezant

mkdir -p logs

module load conda cuda/12.9.1
conda activate /orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/simplefold

nvidia-smi

echo "Starting SimpleNOEFold fine-tuning job..."

# Set correct environment variables
export NCCL_DEBUG=INFO

srun --ntasks=2 --ntasks-per-node=2 python src/simplefold/train_fsdp.py --config-name finetune \
    task_name="SimpleNOEFold_Finetune_3429" \
    trainer.ckpt_path=/orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/artifacts/simplefold_100M.ckpt \
    trainer.max_epochs=50 \
    trainer.num_nodes=1 \
    trainer.devices=2 \
    model.optimizer.lr=1e-5 \
    data.nef_dir=/orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/data/data_train/nef \
    'data.datasets=[{name: "nmr_finetune_set", target_dir: /orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/data/data_train/structures, tokenized_dir: /orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/data/data_train/tokenized, record_list: /orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/data/data_train/nef_train_pdb_ids.txt, manifest_path: /orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/data/data_train/manifest.json, filters: null, esm_dir: null,connectivity_dir: null, cheap_dir: null, cropper: null, cluster: null}]'

echo "Fine-tuning job finished."