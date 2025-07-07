import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from numerical.helper.data_manipulation import transform_organizer_data_to_fc, transform_to_factcheck
from numerical.helper.data_store import read_train_and_val_data, read_organizer_correct_labels
from numerical.helper.inference import predict_val_data
from numerical.helper.logger import set_up_log
from numerical.helper.metrics import FocalLoss, accuracy_fn, macro_f1_fn, reweight, compute_class_report
from numerical.helper.run_config import RunConfig, overwrite_config
from numerical.helper.tracking import track_with_wandb
from numerical.helper.train import (
    compute_token_stats,
    get_step_size,
    set_up_dataloaders,
    set_up_optimizer,
    tokenize_data,
    train,
    train_val_split,
)
from numerical.helper.label import encode_labels, decode_labels
from numerical.helper.util import check_data_availability
from numerical.models.encoder_classifier import BaseClassifier, Tokenizer


def init_args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-Tune Encoder")

    parser.add_argument(
        "--config_path", type=str, default="config/encoder_classifier/run_config_modernbert-base_local.yml", help="Config name"
    )
    parser.add_argument("--force", "-f", default=False, action="store_true", help="Force recomputation of embedding")
    parser.add_argument("--test-wo-labels", "-two", dest="two", default=False, action="store_true", help="Test without labels")

    # Optional arguments
    parser.add_argument("--dropout_ratio", type=float, default=None, help="Dropout ratio for encoder classifier")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Hidden dimension for encoder classifier")
    parser.add_argument("--mlp_dim", type=int, default=None, help="MLP dimension for encoder classifier")
    parser.add_argument("--max_length", type=int, default=None, help="Max length of input sequence")
    parser.add_argument("--freeze_encoder", default=None, action="store_true", help="Freeze encoder weights")
    parser.add_argument("--top_n_evidences", type=int, default=None, help="Top n evidences")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--early_stopping_patience", type=int, default=None, help="Early stopping patience")

    return parser.parse_args()


def main() -> int:
    set_up_log()
    logging.info("Start Fine-Tuning Encoder")
    try:
        args = init_args_parser()
        logging.info(f"Fine-Tuning Encoder with following Arguments: {args.__dict__}")
        logging.info(f"Reading config {args.config_path}")
        RunConfig.load_config(Path(args.config_path))
        logging.info(f"Config data: {RunConfig.data}, encoder: {RunConfig.encoder_model}, train: {RunConfig.train}")

        # Overwrite config with optional arguments
        overwrite_config(
            args.dropout_ratio,
            args.hidden_dim,
            args.mlp_dim,
            args.max_length,
            args.freeze_encoder,
            args.top_n_evidences,
            args.batch_size,
            args.learning_rate,
            args.early_stopping_patience,
        )

        logging.info("Setting seeds for reproducibility.")
        torch.manual_seed(RunConfig.train["seed"])
        torch.cuda.manual_seed_all(RunConfig.train["seed"])
        random.seed(RunConfig.train["seed"])
        np.random.seed(RunConfig.train["seed"])
        torch.mps.manual_seed(RunConfig.train["seed"])
        # Not using torch.use_deterministic_algorithms(True) as it slows the process down and requires cublas config (os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8")

        # Getting device
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        df_train, df_val, df_test = read_train_and_val_data()
        if df_val is None:
            # Split train data
            split_ratio = RunConfig.train["train_test_split"]
            logging.info(f"Splitting train data with ratio '{split_ratio}'.")
            df_train, df_val = train_val_split(df_train)

        if "organizer" not in RunConfig.data["reranked_results_train"]:
            # Sort by claim_id like original dataset
            df_train, df_val = df_train.sort(by="claim_id"), df_val.sort(by="claim_id")
        elif "organizer" in RunConfig.data["reranked_results_train"]:
            # Load correct labels from organizer data
            train_labels, val_labels = read_organizer_correct_labels()
            # Drop duplicates and join the correct labels
            df_train = (
                df_train.drop("label")
                .unique(keep="first", subset="claim", maintain_order=True)
                .join(train_labels[["claim", "label"]], on="claim")
            )
            df_val = (
                df_val.drop("label")
                .unique(keep="first", subset="claim", maintain_order=True)
                .join(val_labels[["claim", "label"]], on="claim")
            )

        df_train, df_val = encode_labels(df_train, df_val)
        num_classes = len(df_train["label"].unique())
        logging.info(f"Number of classes: {num_classes}")

        logging.info("Transforming train and val data to factcheck squences '[Claim]:...[Questions]:...[Evidences]:...'.")
        if "organizer" in RunConfig.data["reranked_results_train"]:
            # Load correct labels from organizer data
            train_labels, val_labels = read_organizer_correct_labels()
            # If the data is from the quanttemp author
            df_train = transform_organizer_data_to_fc(df_train)
            df_val = transform_organizer_data_to_fc(df_val)
            if not args.two:
                df_test = transform_to_factcheck(
                    df_test,
                    RunConfig.train["top_n_evidences"],
                    RunConfig.train["top_n_evidences_refined"],
                    RunConfig.train["top_percentile_evidences"],
                    test=True
                )
        else:
            # our data
            df_train = transform_to_factcheck(
                df_train,
                RunConfig.train["top_n_evidences"],
                RunConfig.train["top_n_evidences_refined"],
                RunConfig.train["top_percentile_evidences"],
            )
            df_val = transform_to_factcheck(
                df_val,
                RunConfig.train["top_n_evidences"],
                RunConfig.train["top_n_evidences_refined"],
                RunConfig.train["top_percentile_evidences"],
            )

        # Load encoder classifer to fine-tune if not already fine-tuned
        encoder_model_name = RunConfig.encoder_model["name"]
        run_name = (
            f"{encoder_model_name.replace('/', '-')}_{'org_data' if 'organizer' in RunConfig.data['reranked_results_train'] else 'our_data'}"
            + f"_max_length{RunConfig.encoder_model['max_length']}_bs{RunConfig.train['batch_size']}_lr{RunConfig.train['learning_rate']}{'_r2l' if RunConfig.encoder_model['r2l'] else ''}"
            + f"_top{RunConfig.train['top_n_evidences'] if RunConfig.train['top_n_evidences'] else RunConfig.train['top_percentile_evidences']}"
            + f"_loss{RunConfig.train['loss']}"
        )
        model_path = Path(RunConfig.data["dir"]) / Path(RunConfig.data["fine_tuned_model_path"] / Path(run_name) / Path("model"))
        tokenizer_path = (
            Path(RunConfig.data["dir"]) / Path(RunConfig.data["fine_tuned_model_path"]) / Path(run_name) / Path("tokenizer")
        )

        if check_data_availability(model_path) & (not args.force):
            logging.info(f"Weights of model '{encoder_model_name}' already exist in '{model_path}'. Loading.")
            model = BaseClassifier.load(model_path)
            tokenizer = Tokenizer.load(tokenizer_path)
        else:
            tokenizer = Tokenizer(model_name=encoder_model_name, r2l=RunConfig.encoder_model["r2l"])
            model = BaseClassifier.create(
                model_name=encoder_model_name,
                labels_count=num_classes,
                freeze_encoder=RunConfig.encoder_model["freeze_encoder"],
                dropout_ratio=RunConfig.encoder_model["dropout_ratio"],
                hidden_dim=RunConfig.encoder_model["hidden_dim"],
                mlp_dim=RunConfig.encoder_model["mlp_dim"],
                lora_rank=RunConfig.encoder_model["lora_rank"],
                lora_alpha=RunConfig.encoder_model["lora_alpha"],
            )

        if device.type == "cuda":
            if torch.cuda.device_count() > 1:
                logging.info(f"Let's use {torch.cuda.device_count()} GPUs.")
                model = torch.nn.DataParallel(model)
                parallel = True  # Flag for storage
            else:
                parallel = False
                logging.info("Using single GPU.")
        else:
            parallel = False
            logging.info("Using single GPU.")

        # Use tokenizer of model to prepare data further and load into dataloaders
        train_input_ids, val_input_ids, test_input_ids, train_masks, val_masks, test_masks = tokenize_data(
            tokenizer, df_train, df_val, df_test, RunConfig.encoder_model["max_length"], args.two
        )

        compute_token_stats(train_input_ids, val_input_ids, test_input_ids, args.two)

        if device.type == "cuda":
            # Record CUDA GPU VRAM Allocation -> pkl can be visualzied @ https://pytorch.org/memory_viz
            torch.cuda.memory._record_memory_history(max_entries=100000)

        train_dataloader, val_dataloader, test_dataloader = set_up_dataloaders(
            RunConfig.train["batch_size"],
            RunConfig.train["seed"],
            RunConfig.train["dataloader_num_workers"],
            train_input_ids,
            val_input_ids,
            test_input_ids,
            train_masks,
            val_masks,
            test_masks,
            df_train,
            df_val,
            args.two,
        )

        num_batchs = len(train_dataloader)
        # Venktesh: sets up the scheduler with total_steps = num_batchs * epochs (warumup_steps = 0). However, does not step at all in training loop
        step_size = get_step_size(RunConfig.train["epochs"], num_batchs, RunConfig.train["warmup_steps"])
        logging.info(
            f"With {RunConfig.train['epochs']} epochs, {num_batchs} num_batches, {RunConfig.train['warmup_steps']} warumup_steps, "
            + f"determined a step size of: {step_size}."
        )

        # Set up optimizer
        optimizer, scheduler = set_up_optimizer(
            model, RunConfig.train["learning_rate"], RunConfig.train["eps"], RunConfig.train["warmup_steps"], step_size
        )

        # Set up loss function
        if RunConfig.train["loss"] == "cross_entropy":
            logging.info("Using CrossEntropyLoss.")
            loss = torch.nn.CrossEntropyLoss()
        elif RunConfig.train["loss"] == "focal":
            logging.info("Using FocalLoss.")
            # Reweighting
            cls_num_count = df_train.group_by("label_encoded").len()["len"].to_list()
            per_cls_weights = reweight(cls_num_count)
            # Move weights to device
            per_cls_weights = per_cls_weights.to("cuda")
            loss = FocalLoss(weight=per_cls_weights, gamma=RunConfig.train["focal_loss_gamma"])

        else:
            logging.error(f"Loss function '{RunConfig.train['loss']}' not supported.")
            raise ValueError(f"Loss function '{RunConfig.train['loss']}' not supported.")

        # Train and track with wandb
        best_model, best_score, df_submission_dev, df_submission_test = track_with_wandb(
            train_fn=train,
            predict_fn=predict_val_data,
            run_name=run_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            metric_fns={"accuracy": accuracy_fn, "macro_f1": macro_f1_fn},
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            device=device,
            epochs=RunConfig.train["epochs"],
            early_stopping_patience=RunConfig.train["early_stopping_patience"],
            df_val=df_val,
            df_test=df_test,
            test_wo_labels=args.two,
        )

        if device.type == "cuda":
            # Dump memory snapshot history to a file and stop recording
            torch.cuda.memory._dump_snapshot(f"{encoder_model_name.replace('/', '-') + '-classifier'}_finetune_memory.pkl")
            torch.cuda.memory._record_memory_history(enabled=None)

        # Save weights of fine-tuned model
        logging.info(f"Saving model to {model_path}.")
        if parallel:
            best_model.module.save(model_path)
        else:
            best_model.save(model_path)
        logging.info(f"Saving tokenizer to {tokenizer_path}.")
        tokenizer.save(tokenizer_path)
        logging.info(f"Finished fine-tuning with best macro-avg f1: {best_score}.")

        if df_submission_dev is not None:
            # Save df_submission
            df_submission_dev_path = (
                Path(RunConfig.data["dir"])
                / Path(RunConfig.data["fine_tuned_model_path"])
                / Path(run_name)
                / Path("df_submission_dev.tsv")
            )
            logging.info(f"Saving df_submission_dev to {df_submission_dev_path}.")
            df_submission_dev.write_csv(df_submission_dev_path, separator="\t")

        if df_submission_test is not None:
            # Save df_submission
            df_submission_test_path = (
                Path(RunConfig.data["dir"])
                / Path(RunConfig.data["fine_tuned_model_path"])
                / Path(run_name)
                / Path("df_submission_test.tsv")
            )
            logging.info(f"Saving df_submission_test to {df_submission_test_path}.")
            df_submission_test.write_csv(df_submission_test_path, separator="\t")

        logging.info("Finished!")
        return best_score

    except Exception:
        if torch.cuda.is_available():
            # Dump memory snapshot history to a file and stop recording
            torch.cuda.memory._dump_snapshot(f"{encoder_model_name.replace('/', '-') + '-classifier'}_finetune_memory.pkl")
            torch.cuda.memory._record_memory_history(enabled=None)

        logging.exception("Fine-Tuning failed", stack_info=True)
        return 1


if __name__ == "__main__":
    # Protect script entry point for DataLoader subprocesses on Win/macOS ("spawn" method)
    main()
