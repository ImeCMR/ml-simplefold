#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import hydra
import torch
import time
import functools
from omegaconf import OmegaConf

import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from model.torch.blocks import DiTBlock
from utils.utils import (
    extras,
    create_folders,
    task_wrapper,
)
from utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
)
from utils.logging_utils import log_hyperparameters
from utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)
torch.set_float32_matmul_precision("medium")


@task_wrapper
def train(cfg):
    seed = cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    # Use a file-based lock to prevent race conditions during model download
    # Get rank from Slurm environment variable, default to 0 if not set
    try:
        rank = int(os.environ.get("SLURM_PROCID", 0))
    except ValueError:
        rank = 0

    lock_file = os.path.join(cfg.paths.output_dir, "model_init.lock")

    if rank == 0:
        # Rank 0 creates the lock, instantiates the model (triggering download), and removes the lock
        log.info("Rank 0: Creating model initialization lock.")
        # Ensure the directory exists before creating the lock file
        os.makedirs(os.path.dirname(lock_file), exist_ok=True)
        with open(lock_file, "w") as f:
            f.write("locked")

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info("Rank 0: Removing model initialization lock.")
        os.remove(lock_file)
    else:
        # Other ranks wait for the lock file to be removed
        log.info(f"Rank {rank}: Waiting for model initialization lock to be released by rank 0.")
        wait_time = 0
        while os.path.exists(lock_file):
            time.sleep(5)
            wait_time += 5
            if wait_time > 600: # Add a timeout to prevent hanging forever
                raise TimeoutError("Waited too long for the model initialization lock file.")
        log.info(f"Rank {rank}: Lock released. Instantiating model from cache.")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Handle checkpoint path from both root and trainer configs for robustness
    # We temporarily set struct to False to allow popping the ckpt_path key
    OmegaConf.set_struct(cfg.trainer, False)
    ckpt_path = cfg.trainer.pop("ckpt_path", None) or cfg.get("load_ckpt_path", None)
    OmegaConf.set_struct(cfg.trainer, True)

    if ckpt_path:
        # load existing ckpt
        log.info(f"Resuming from checkpoint <{ckpt_path}>...")
        model.strict_loading = False

        # manually reset these variables in case of fine-tuning
        model.lddt_weight_schedule = cfg.model.get("lddt_weight_schedule", False)
        model.plddt_training = cfg.model.get("plddt_training", False)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    OmegaConf.set_struct(cfg.logger, True)
    loggers = instantiate_loggers(cfg.get("logger"))

    # When using FSDP, we need to manually specify the wrap policy
    # and activation checkpointing policy for transformer layers.

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    tmp_esm_model, _ = torch.hub.load("facebookresearch/esm:main", "esm2_t6_8M_UR50D")
    esm_layer_class = tmp_esm_model.layers[0].__class__
    del tmp_esm_model

    transformer_auto_wrapper_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={DiTBlock, esm_layer_class},
    )
    strategy = FSDPStrategy(
        auto_wrap_policy=transformer_auto_wrapper_policy,
        activation_checkpointing_policy={DiTBlock, esm_layer_class},
        use_orig_params=True,
        state_dict_type="sharded",
        limit_all_gathers=True,
        cpu_offload=False
    )
    trainer = hydra.utils.instantiate(
        cfg.trainer, 
        strategy=strategy,
        callbacks=callbacks, 
        logger=loggers, 
        plugins=None
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }

    if log:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting training!")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="base_train.yaml")
def submit_run(cfg):
    OmegaConf.resolve(cfg)
    extras(cfg)
    create_folders(cfg)
    train(cfg)
    return


if __name__ == "__main__":
    submit_run()
