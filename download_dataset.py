import logging
import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml


def download_kaggle_dataset(dataset_name, download_path):
    from kaggle.api.kaggle_api_extended import KaggleApi

    """
    Download dataset from Kaggle.

    Args:
        dataset_name (str): Kaggle dataset name in 'owner/dataset' format.
        download_path (str): Path to download the dataset.

    """
    api = KaggleApi()
    api.authenticate()
    logging.info(f"Downloading dataset {dataset_name} to {download_path}...")
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    logging.info("Download completed.")


def organize_dataset(raw_path, processed_path):
    """
    Organize the dataset into processed directories.

    Args:
        raw_path (str): Path where raw dataset is downloaded.
        processed_path (str): Path to save processed dataset.

    """
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)
    processed_path.mkdir(parents=True, exist_ok=True)

    for folder in raw_path.iterdir():
        if folder.is_dir():
            label = folder.name
            label_dir = processed_path / label
            label_dir.mkdir(parents=True, exist_ok=True)
            for img in folder.glob("*.*"):
                if img.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
                    shutil.move(str(img), label_dir / img.name)
    logging.info("Dataset organized.")


def main(dataset_name=None, raw_path=None, processed_path=None, extra_path=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load environment variables
    load_dotenv(".env")

    # Create directories if they don't exist
    Path(raw_path).mkdir(parents=True, exist_ok=True)
    Path(processed_path).mkdir(parents=True, exist_ok=True)

    # Download dataset
    download_kaggle_dataset(dataset_name, raw_path)

    raw_dataset_path = raw_path + "/" + extra_path
    # Organize dataset
    organize_dataset(raw_dataset_path, processed_path)

    logging.info("Dataset downloaded and organized.")


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load the YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Dict[str, Any]: Configuration parameters.

    """
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config("src/config/config.yaml")
    use_both_datasets = config["data"]["use_both_datasets"]
    if use_both_datasets:
        main(
            dataset_name="bhawks/pokemon-generation-one-22k",
            raw_path="data/raw",
            processed_path="data/processed",
            extra_path="PokemonData",
        )
        main(
            dataset_name="thedagger/pokemon-generation-one",
            raw_path="data/raw",
            processed_path="data/processed",
            extra_path="dataset",
        )
