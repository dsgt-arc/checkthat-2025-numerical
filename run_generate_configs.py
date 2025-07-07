"""Script to generate multiple configuration files for hyperparameter tuning.
Configs can than be used to run multiple experiments in parallel via Slurm Array-Jobs.
"""

import itertools
import copy
import yaml
from pathlib import Path
import logging
from numerical.helper.logger import set_up_log


def main():
    set_up_log()
    logging.info("Start Config-Generation")

    # Base configuration dictionary
    base_config = {
        "data": {
            "dir": "/storage/coda1/p-dsgt_clef2025/0/shared/checkthat-2025-numerical-data",
            "reranked_results_train": "reranking/reranked_results_train.csv",
            "reranked_results_val": "reranking/reranked_results_val.csv",
            "fine_tuned_model_path": "fine_tuned_models",
        },
        "encoder_model": {
            "name": "answerdotai/ModernBERT-large",  # (uf-aice-lab/math-roberta, modernbert-large, roberta-large)
            "TOKENIZERS_PARALLELISM": False,
            "add_special_tokens": True,
            "padding": "max_length",
            "truncation": True,
            "return_attention_mask": True,
            "return_tensors": "pt",
            "pad_to_max_length": True,
            "max_length": 1024,
            "r2l": False,
            "batch_size": 20,  # only used for 00-eda-claims
            "hidden_dim": 1024,  # depends on the model
            "freeze_encoder": None,  # (None, "first_5_layers", "all")
            "dropout_ratio": 0.1,
            "mlp_dim": 768,
            "lora_rank": 4,  # (recommended: 4 - 8)
            "lora_alpha": 8,  # (recommended: 2x lora_rank)
        },
        "train": {
            "train_test_split": 0.8,
            "top_n_evidences": 5,
            "batch_size": 16,  # used for 02-nli
            "dataloader_num_workers": 0,
            "seed": 42,
            "epochs": 10,
            "learning_rate": 0.00002,
            "eps": 1e-8,
            "early_stopping_patience": 2,
            "rounding_metrices": 4,
            "warmup_steps": 0,
            "step_per": "epoch",
        },
    }

    # Define the lists of hyperparameter values to try
    learning_rate_vals = [0.00002, 0.0001]
    top_n_evidences_vals = [1, 3]
    train_batch_size_vals = [16, 32]
    mlp_dim_vals = [768, 1024]
    lora_rank_vals = [4, 8]
    lora_alpha_vals = [8, 16]  # typically 2 x lora_rank
    max_length_vals = [1024, 2048]

    # Create a directory to save the generated config files
    output_dir = Path("config/generated_configs")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Clear the directory if it already exists
        for file in output_dir.glob("*.yaml"):
            file.unlink()

    # Counter for naming each config file
    config_count = 0

    # Iterate over the Cartesian product of all hyperparameter lists
    for lr, top_n, train_bs, mlp_dim, lora_rank, lora_alpha, max_length in itertools.product(
        learning_rate_vals,
        top_n_evidences_vals,
        train_batch_size_vals,
        mlp_dim_vals,
        lora_rank_vals,
        lora_alpha_vals,
        max_length_vals,
    ):
        # Make a deep copy of the base config so that changes don't affect others
        config_variant = copy.deepcopy(base_config)

        # Update hyperparameters in the train section
        config_variant["train"]["learning_rate"] = lr
        config_variant["train"]["top_n_evidences"] = top_n
        config_variant["train"]["batch_size"] = train_bs

        # Update hyperparameters in the encoder_model section
        config_variant["encoder_model"]["mlp_dim"] = mlp_dim
        config_variant["encoder_model"]["lora_rank"] = lora_rank
        config_variant["encoder_model"]["lora_alpha"] = lora_alpha
        config_variant["encoder_model"]["max_length"] = max_length

        # Optionally, you can also update other parameters if needed

        # Save the configuration variant to a YAML file
        config_filename = output_dir / Path(f"config_{config_count}.yaml")
        with open(config_filename, "w") as f:
            yaml.dump(config_variant, f, sort_keys=False)

        config_count += 1

    logging.info(f"Generated {config_count} configuration files in '{output_dir}' directory.")


main()
