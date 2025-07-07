import logging
from pathlib import Path
from typing import Optional

import polars as pl
from numerical.helper.run_config import RunConfig


class DataStore:
    def __init__(self, location: str = "data") -> None:
        self.location = Path(location)
        self.data = None

    def read_json_data(self, file: str, schema_overrides: Optional[dict[str, str]] = None) -> None:
        """Reads data from a file and stores it in the data attribute as pl.DataFrame."""
        path = self.location / Path(file)
        logging.info(f"Reading {str(path)}")
        self.data = pl.read_json(path, schema_overrides=schema_overrides)

    def read_csv_data(self, file: str, separator: str = ",", schema_overrides: Optional[dict[str, str]] = None) -> None:
        """Reads data from a file and stores it in the data attribute as pl.DataFrame."""
        path = self.location / Path(file)
        logging.info(f"Reading {str(path)}")
        self.data = pl.read_csv(path, separator=separator, schema_overrides=schema_overrides)

    def get_data(self) -> pl.DataFrame:
        if self.data is None:
            raise ValueError("Data not read yet. Call read_data() first.")
        return self.data


def read_train_and_val_data() -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None]:
    """Read train and validation data."""
    logging.info(
        f"Reading train, validation and test data from '{RunConfig.data['reranked_results_train']}'"
        + f", '{RunConfig.data['reranked_results_val']}' and '{RunConfig.data['reranked_results_test']}'."
    )

    ds = DataStore(location=RunConfig.data["dir"])
    train_path = Path(RunConfig.data["reranked_results_train"])
    if train_path.suffix == ".json":
        ds.read_json_data(train_path)
    elif train_path.suffix == ".csv":
        ds.read_csv_data(train_path, schema_overrides={"label": pl.String})
    else:
        raise NotImplementedError(f"File format '{train_path.suffix}' not supported.")
    df_train = ds.get_data()

    if RunConfig.data["reranked_results_val"] is None:
        logging.info("No validation data available. Need to split train.")
        df_val = None
    else:
        val_path = Path(RunConfig.data["reranked_results_val"])
        if val_path.suffix == ".json":
            ds.read_json_data(val_path)
        elif val_path.suffix == ".csv":
            ds.read_csv_data(val_path, schema_overrides={"label": pl.String})
        else:
            raise NotImplementedError(f"File format '{val_path.suffix}' not supported.")
        df_val = ds.get_data()

    if RunConfig.data["reranked_results_test"] is None:
        logging.info("No test data available. Need to split train.")
        df_test = None
    else:
        test_path = Path(RunConfig.data["reranked_results_test"])
        if test_path.suffix == ".json":
            ds.read_json_data(test_path)
        elif test_path.suffix == ".csv":
            ds.read_csv_data(test_path, schema_overrides={"label": pl.String})
        else:
            raise NotImplementedError(f"File format '{test_path.suffix}' not supported.")
        df_test = ds.get_data()

    return df_train, df_val, df_test


def read_organizer_correct_labels() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read organizer correct labels."""
    logging.info(
        f"Reading organizer correct labels from '{RunConfig.data['organizer_correct_labels_train']}' and '{RunConfig.data['organizer_correct_labels_val']}'."
    )

    ds = DataStore(location=RunConfig.data["dir"])
    train_path = Path(RunConfig.data["organizer_correct_labels_train"])
    if train_path.suffix == ".json":
        ds.read_json_data(train_path)
    elif train_path.suffix == ".csv":
        ds.read_csv_data(train_path, schema_overrides={"label": pl.String})
    else:
        raise NotImplementedError(f"File format '{train_path.suffix}' not supported.")
    train_correct_labels = ds.get_data()

    val_path = Path(RunConfig.data["organizer_correct_labels_val"])
    if val_path.suffix == ".json":
        ds.read_json_data(val_path)
    elif val_path.suffix == ".csv":
        ds.read_csv_data(val_path, schema_overrides={"label": pl.String})
    else:
        raise NotImplementedError(f"File format '{val_path.suffix}' not supported.")
    val_correct_labels = ds.get_data()

    return train_correct_labels, val_correct_labels
