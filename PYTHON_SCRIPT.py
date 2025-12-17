# This is the code for your Python notebook.

import hydra
from hydra import compose, initialize
import torch
from omegaconf import OmegaConf
import os

# --- IMPORTANT: Set the path to your project's root directory ---
project_root = "/orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold" # Change this if it's different

# --- This is the key fix ---
# Change the current working directory to the project root.
# This makes all relative paths (here and in the configs) work correctly.
os.chdir(project_root)

print(f"Successfully changed working directory to: {os.getcwd()}")

# --- Initialize Hydra Using a Relative Path ---
# Now that we are in the project root, the config path is simply "configs".
with initialize(config_path="configs", version_base=None):

    # --- THIS IS THE KEY CHANGE ---
    # Use the new 'debug_finetune' config. It's designed for a quick, lightweight test.
    cfg = compose(config_name="experiment/debug_finetune")

print("Hydra configuration loaded successfully!")
# You can uncomment the line below to see the full configuration that was loaded.
# print(OmegaConf.to_yaml(cfg))

# --- Instantiate the Dataloader ---
print("Instantiating datamodel...")
datamodule = hydra.utils.instantiate(cfg.data)
datamodule.setup()
dataloader = datamodule.train_dataloader()
print("Dataloader created.")

# --- Get a Single Batch ---
print("Fetching one batch of data...")
batch = next(iter(dataloader))
print("Batch fetched successfully!")

# --- Inspect the Batch ---
print("\n--- Inspecting Batch Contents ---")
for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        print(f"Key: '{key}', Type: Tensor, Shape: {value.shape}, Device: {value.device}")
    elif isinstance(value, list) and value:
        print(f"Key: '{key}', Type: List, Length: {len(value)}")
        print(f"  - First element's type: {type(value[0])}")
