# SimpleNOEFold Fine-Tuning Guide

This guide explains how to run a fine-tuning job for SimpleNOEFold using the provided configuration and Slurm script.

## 1. Understanding the Configuration

The fine-tuning process is controlled by a combination of configuration files and command-line arguments, all managed by Hydra. Here are the key files and their roles:

-   **`configs/finetune.yaml`**: This is the **top-level configuration** for your fine-tuning job. When you run the training script, you will specify `--config-name finetune` to load this file as the base.

-   **`configs/model/simplefold.yaml`**: This is the primary configuration for the `SimpleFold` model. We have added `bayesian_steering: null` to this file. This makes the model "aware" of the Bayesian steering module, allowing us to configure it from the command line.

-   **`configs/model/bayesian_steering/default.yaml`**: This file defines the default parameters for the `BayesianSteering` module, including weights for the different energy terms and physical constants.

## 2. The Fine-Tuning Command

The `run_finetune.slurm` script contains the definitive `srun` command to launch your job. Let's break down how it works:

```bash
srun python src/simplefold/train_fsdp.py --config-name finetune \
    # ... (other parameters)
    model.bayesian_steering=default \
    # ... (other parameters)
```

-   **`--config-name finetune`**: This tells Hydra to load `configs/finetune.yaml` as the main configuration.
-   **`model.bayesian_steering=default`**: This is the crucial override. It instructs Hydra to:
    1.  Look at the `bayesian_steering` parameter inside the loaded model configuration (`simplefold.yaml`).
    2.  Instead of `null`, it loads the configuration from `configs/model/bayesian_steering/default.yaml`.
-   **Other Overrides**: All the other arguments (e.g., `data.nef_dir`, `trainer.max_epochs`, `model.optimizer.lr`) are Hydra overrides that set the specific parameters for this experiment, similar to how they were defined in the non-runnable `finetune_noesy.yaml` experiment file.

## 3. How the Code Uses This Configuration

The `SimpleFold` model in `src/simplefold/model/simplefold.py` is already written to accept and use the `BayesianSteering` module.

-   The `__init__` method of the `SimpleFold` class takes an optional `bayesian_steering` argument. When you run the command above, Hydra automatically instantiates the `BayesianSteering` object (using the `default.yaml` config) and passes it to the model.
-   The `training_step` method checks if `self.bayesian_steering` is not `None`. If it exists, it calculates the Bayesian energy term and adds it to the main loss function, enabling the fine-tuning process you designed.

## 4. How to Run the Fine-Tuning Job

1.  **Customize `run_finetune.slurm`**:
    -   Open the `run_finetune.slurm` script.
    -   Replace the placeholder values (e.g., `YOUR_EMAIL@EXAMPLE.COM`, `<PATH_TO_YOUR_CONDA_ENVIRONMENT>`, `<PATH_TO_PRETRAINED_CHECKPOINT>`, `<PATH_TO_YOUR_DATA_DIRECTORY>`) with your actual information and paths.

2.  **Submit the Job**:
    -   Once the script is customized, submit it to the Slurm scheduler using `sbatch`:
        ```bash
        sbatch run_finetune.slurm
        ```

This will launch the fine-tuning job with the correct configuration, enabling the Bayesian steering mechanism to guide the model's training.
