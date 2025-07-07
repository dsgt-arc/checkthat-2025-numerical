import logging
from pathlib import Path

import yaml


class RunConfig:
    """Singleton class to store config data."""

    data = {}
    encoder_model = {}
    dim_red_model = {}
    visualization = {}
    reranking = {}
    train = {}

    @classmethod
    def load_config(cls, path: Path = Path("config/run_config.yml")) -> None:
        """Load YAML config into a RunConfig object."""
        if not isinstance(path, Path):
            path = Path(path)

        with open(path, "r") as f:
            config_data = yaml.safe_load(f)

        cls.data = config_data.get("data")
        cls.encoder_model = config_data.get("encoder_model")
        cls.dim_red_model = config_data.get("dim_red_model")
        cls.visualization = config_data.get("visualization")
        cls.reranking = config_data.get("reranking")
        cls.train = config_data.get("train")


def overwrite_config(
    dropout_ratio: float | None = None,
    hidden_dim: int | None = None,
    mlp_dim: int | None = None,
    max_length: int | None = None,
    freeze_encoder: bool | None = None,
    top_n_evidences: int | None = None,
    batch_size: int | None = None,
    learning_rate: int | None = None,
    early_stopping_patience: int | None = None,
) -> None:
    """Overwrite config with optional arguments."""
    overwrite_config = False

    if dropout_ratio is not None:
        RunConfig.encoder_model["dropout_ratio"] = dropout_ratio
        overwrite_config = True
    if hidden_dim is not None:
        RunConfig.encoder_model["hidden_dim"] = hidden_dim
        overwrite_config = True
    if mlp_dim is not None:
        RunConfig.encoder_model["mlp_dim"] = mlp_dim
        overwrite_config = True
    if max_length is not None:
        RunConfig.encoder_model["max_length"] = max_length
        overwrite_config = True
    if freeze_encoder is not None:
        RunConfig.encoder_model["freeze_encoder"] = freeze_encoder
        overwrite_config = True
    if top_n_evidences is not None:
        RunConfig.train["top_n_evidences"] = top_n_evidences
        overwrite_config = True
    if batch_size is not None:
        RunConfig.train["batch_size"] = batch_size
        overwrite_config = True
    if learning_rate is not None:
        RunConfig.train["learning_rate"] = learning_rate
        overwrite_config = True
    if early_stopping_patience is not None:
        RunConfig.train["early_stopping_patience"] = early_stopping_patience
        overwrite_config = True

    if overwrite_config:
        logging.info("Overwritten config with optional arguments.")
        logging.info(f"Encoder Model: {RunConfig.encoder_model}")
        logging.info(f"Training: {RunConfig.train}")
