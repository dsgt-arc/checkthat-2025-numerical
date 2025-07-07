import logging
import torch
import polars as pl
from torch.amp import autocast
from numerical.helper.metrics import compute_class_report
from typing import Callable


def predict_val_data(
    val_dataloader: torch.utils.data.DataLoader,
    best_model: torch.nn.Module,
    device: torch.device,
    df: pl.DataFrame,
    log_fn: Callable[[dict], None] | None = None,
    test_only: bool = False,
) -> pl.DataFrame:
    # Predict val data given best model
    # This is only possible because I use sequential sampler for val_dataloader
    logging.info("Predicting val data with best model.")
    predicted_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)

            # Inference
            with autocast(device.type):
                outputs = best_model(input_ids, attention_masks)
            _, predicted_lab = torch.max(outputs, dim=1)
            predicted_labels.append(predicted_lab)

    predicted_labels = torch.cat(predicted_labels)

    # Merge to df_val
    df_with_preds = df.with_columns(pl.Series(predicted_labels.cpu().numpy()).alias("predicted_label"))
    if not test_only:
        classification_report = compute_class_report(
            df_with_preds["label_encoded"].to_numpy(), df_with_preds["predicted_label"].to_numpy()
        )
        logging.info(f"Inference classification report: {classification_report}")

        # Logging with ML-experiment tracker (wandb)
        if log_fn:
            false_class_f1 = classification_report.filter(pl.col("category") == "False")["f1-score"]
            half_class_f1 = classification_report.filter(pl.col("category") == "Half True/False")["f1-score"]
            true_class_f1 = classification_report.filter(pl.col("category") == "True")["f1-score"]

            log_fn(
                {
                    "best_val/macro_f1": classification_report.filter(pl.col("category") == "macro avg")["f1-score"][0],
                    "best_val/accuracy": classification_report.filter(pl.col("category") == "accuracy")["f1-score"][
                        0
                    ],  # col does not matter
                    "best_val/false_class_f1": false_class_f1[0] if false_class_f1.shape[0] > 0 else None,
                    "best_val/conflicting_class_f1": half_class_f1[0] if half_class_f1.shape[0] > 0 else None,
                    "best_val/true_class_f1": true_class_f1[0] if true_class_f1.shape[0] > 0 else None,
                }
            )

    return df_with_preds
