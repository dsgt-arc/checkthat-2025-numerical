import logging
import torch
import polars as pl
import numpy as np

# Explicitly encode labels given mapping dict to avoid any issues with orginal data vs our data
LABEL_ENCODING = {"False": 0, "Conflicting": 1, "Half True/False": 1, "True": 2}
LABEL_DECODING = {value: key for key, value in LABEL_ENCODING.items()}


def encode_labels(
    df_train: pl.DataFrame, df_val: pl.DataFrame, mapping: dict[str, str] = LABEL_ENCODING
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Encode labels."""

    logging.info("Encoding labels.")

    df_train = df_train.with_columns(pl.col("label").replace_strict(mapping).alias("label_encoded"))
    df_val = df_val.with_columns(pl.col("label").replace_strict(mapping).alias("label_encoded"))

    return df_train, df_val


def decode_labels(input: torch.Tensor | np.ndarray, mapping: dict[str, str] = LABEL_DECODING) -> torch.Tensor:
    """Decode labels."""
    logging.info("Decoding labels.")
    if isinstance(input, torch.Tensor):
        # Convert to numpy array if tensor
        input = input.cpu().numpy()
    encoded_strings = np.vectorize(LABEL_DECODING.get)(input)
    return encoded_strings
