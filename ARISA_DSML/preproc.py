"""Functions for preprocessing the diabetes dataset."""

import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger
import pandas as pd

from ARISA_DSML.config import DATASET, PROCESSED_DATA_DIR, RAW_DATA_DIR


def get_raw_data(dataset: str = DATASET) -> None:
    """Download and unzip the diabetes dataset from Kaggle."""
    api = KaggleApi()
    api.authenticate()

    download_folder = Path(RAW_DATA_DIR)
    logger.info(f"RAW_DATA_DIR is: {RAW_DATA_DIR}")
    api.dataset_download_files(dataset, path=str(download_folder), unzip=True)


def preprocess_df(file: str | Path) -> str | Path:
    """Preprocess the diabetes dataset for ML pipeline."""
    _, file_name = os.path.split(file)
    df_data = pd.read_csv(file)

    # List of columns in the diabetes dataset that should not have zeros
    cols_with_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_missing:
        if col in df_data.columns:
            df_data[col] = df_data[col].replace(0, pd.NA)
            df_data[col] = df_data[col].fillna(df_data[col].median())

    # Optionally drop any irrelevant columns (uncomment if needed)
    # if 'id' in df_data.columns:
    #     df_data = df_data.drop(columns=['id'])

    # Save preprocessed data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    outfile_path = PROCESSED_DATA_DIR / file_name
    df_data.to_csv(outfile_path, index=False)

    return outfile_path


if __name__ == "__main__":
    # Download the dataset
    logger.info("Getting diabetes dataset")
    get_raw_data()

    # Find the CSV file downloaded
    files = list(Path(RAW_DATA_DIR).glob("*.csv"))
    if not files:
        logger.error("No CSV file found in the raw data directory.")
        raise FileNotFoundError("Expected a diabetes CSV file in raw data directory.")

    # Preprocess the file
    for file in files:
        logger.info(f"Preprocessing {file.name}")
        preprocess_df(file)
    logger.info("Preprocessing complete.")
    logger.info("All files processed and saved in the processed data directory.")
