import os
from typing import Callable
from pathlib import Path
from time import sleep

import torch
import wandb
import logging

import polars as pl

from numerical.helper.run_config import RunConfig
from numerical.helper.label import decode_labels


def login_wandb() -> None:
    # Login with API key from environment
    wandb.login(key=os.environ.get("WANDB_API_KEY"), relogin=True)


def track_with_wandb(
    train_fn: Callable, predict_fn: Callable, *, run_name: str, **kwargs
) -> tuple[torch.nn.Module, float, pl.DataFrame | None]:
    login_wandb()

    with wandb.init(
        project="checkthat-numerical",
        name=run_name,
        config={"encoder_model": RunConfig.encoder_model, "train": RunConfig.train, "data": RunConfig.data},
    ):
        wandb.watch(kwargs["model"], log="all", log_freq=100)
        kwargs["log_fn"] = wandb.log

        model, best_macro_f1, best_val_preds = train_fn(**kwargs)
        wandb.log({"final_macro_f1": best_macro_f1})

        # Prediction on validation data
        if "val_dataloader" in kwargs and "device" in kwargs and "df_val" in kwargs:
            # VALIDATION SET
            df_val_with_predicted_labels = predict_fn(
                kwargs["val_dataloader"], model, kwargs["device"], kwargs["df_val"], kwargs["log_fn"]
            )
            assert all(
                df_val_with_predicted_labels["predicted_label"].to_numpy() == best_val_preds.argmax(dim=1).cpu().numpy()
            ), "Predictions do not match best_val_preds"

            # Send results to wandb
            # As Artifact
            logging.info("Upload Predictions & Submission Results as Artifact")
            df_val_preds_upload = df_val_with_predicted_labels[["claim", "sequence", "label", "label_encoded", "predicted_label"]]
            df_submission_dev = df_val_preds_upload.with_columns(
                verdict=decode_labels(df_val_preds_upload["predicted_label"].to_numpy())
            ).select(["claim", "verdict"])
            wandb.log({"best_val/val_preds": wandb.Table(dataframe=df_val_preds_upload.to_pandas())})
            wandb.log({"best_val/val_preds_submission": wandb.Table(dataframe=df_submission_dev.to_pandas())})

            # As File
            logging.info("Upload Predictions & Submission Results as Files")
            temp_path = Path("df_val_with_preds.txt")
            df_val_preds_upload.write_csv(temp_path, separator="\t")
            wandb.save(str(temp_path))
            sleep(10)
            temp_path.unlink()

            temp_path = Path("df_submission_dev.txt")
            df_submission_dev.write_csv(temp_path, separator="\t")
            wandb.save(str(temp_path))
            sleep(10)
            temp_path.unlink()

        else:
            df_submission_dev = None

        # TEST SET
        if kwargs["test_wo_labels"] is not True and "test_dataloader" in kwargs and "device" in kwargs and "df_test" in kwargs:
            df_test_with_predicted_labels = predict_fn(
                kwargs["test_dataloader"], model, kwargs["device"], kwargs["df_test"], kwargs["log_fn"], test_only=True
            )
            df_submission_test = df_test_with_predicted_labels.with_columns(
                verdict=decode_labels(df_test_with_predicted_labels["predicted_label"].to_numpy())
            ).select(["claim", "verdict"])
            wandb.log({"test/test_preds_submission": wandb.Table(dataframe=df_submission_test.to_pandas())})

            temp_path = Path("df_submission_test.txt")
            df_submission_test.write_csv(temp_path, separator="\t")
            wandb.save(str(temp_path))
            sleep(10)
            temp_path.unlink()
        else:
            df_submission_test = None

    return (model, best_macro_f1, df_submission_dev, df_submission_test)
