from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Union
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure dual handlers: stdout console stream + persistent log file
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOG_DIR / "feature_engineering.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike[str], Path]


def load_params(param_path: PathLike = "params.yaml") -> dict[str, Any]:
    """Load configuration parameters from a YAML file."""
    logger.debug("Attempting to load configuration file from: %s", param_path)
    try:
        with open(param_path, "r", encoding="utf-8") as file:
            data: Any = yaml.safe_load(file)
            if not isinstance(data, dict):
                raise ValueError(
                    f"YAML content at '{param_path}' must be a key-value mapping."
                )
            logger.info("Successfully loaded parameters from '%s'.", param_path)
            return data
    except FileNotFoundError:
        logger.error("Configuration file not found: '%s'.", param_path)
        raise
    except yaml.YAMLError:
        logger.exception("Malformed YAML file: '%s'.", param_path)
        raise
    except PermissionError:
        logger.error("Permission denied reading file: '%s'.", param_path)
        raise


def load_data(
    train_path: PathLike, test_path: PathLike
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load interim train and test datasets."""
    logger.info("Loading training data from: %s", train_path)
    logger.info("Loading testing data from: %s", test_path)
    try:
        train_df: pd.DataFrame = pd.read_csv(train_path)
        test_df: pd.DataFrame = pd.read_csv(test_path)

        if train_df.empty:
            logger.warning("Training dataset at '%s' contains 0 records.", train_path)
        if test_df.empty:
            logger.warning("Test dataset at '%s' contains 0 records.", test_path)

        logger.info(
            "Dataset shapes loaded -> Train: %s, Test: %s",
            train_df.shape,
            test_df.shape,
        )
        return train_df, test_df
    except FileNotFoundError as err:
        logger.error("Data file not found: %s", err.filename)
        raise
    except pd.errors.EmptyDataError:
        logger.exception("Encountered empty CSV file during ingestion.")
        raise
    except pd.errors.ParserError:
        logger.exception("Failed to parse CSV file content.")
        raise


def apply_tfidf_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int,
    text_col: str = "content",
    label_col: str = "sentiment",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply TF-IDF feature extraction and reattach labels."""
    logger.info(
        "Initializing TF-IDF extraction: max_features=%d, text_col='%s', label_col='%s'",
        max_features,
        text_col,
        label_col,
    )

    # Validate column presence
    for name, df in [("train", train_df), ("test", test_df)]:
        missing = [col for col in (text_col, label_col) if col not in df.columns]
        if missing:
            logger.error(
                "Missing column(s) %s in %s dataset. Available: %s",
                missing,
                name,
                list(df.columns),
            )
            raise KeyError(
                f"Missing required column(s) {missing} in {name} dataset. Available: {list(df.columns)}"
            )

    if max_features <= 0:
        logger.error("Invalid max_features value: %d (must be > 0)", max_features)
        raise ValueError(
            f"'max_features' must be a positive integer, got {max_features}."
        )

    try:
        train_missing_count = train_df[text_col].isna().sum()
        test_missing_count = test_df[text_col].isna().sum()
        if train_missing_count > 0:
            logger.warning(
                "Imputing %d missing text values in train['%s'] with empty strings.",
                train_missing_count,
                text_col,
            )
        if test_missing_count > 0:
            logger.warning(
                "Imputing %d missing text values in test['%s'] with empty strings.",
                test_missing_count,
                text_col,
            )

        X_train: pd.Series[str] = train_df[text_col].fillna("").astype(str)
        X_test: pd.Series[str] = test_df[text_col].fillna("").astype(str)

        start_time = time.perf_counter()
        vectorizer: TfidfVectorizer = TfidfVectorizer(max_features=max_features)

        logger.debug("Fitting TfidfVectorizer on training text...")
        X_train_bow = vectorizer.fit_transform(X_train)
        logger.debug("Transforming test text using fitted vectorizer...")
        X_test_bow = vectorizer.transform(X_test)

        duration = time.perf_counter() - start_time
        logger.info(
            "Vectorization complete in %.2f seconds. Train matrix: %s, Test matrix: %s",
            duration,
            X_train_bow.shape,
            X_test_bow.shape,
        )

        feature_names = vectorizer.get_feature_names_out()
        if len(feature_names) == 0:
            logger.error("Vocabulary extraction yielded 0 features.")
            raise ValueError(
                "Vocabulary is empty. Check that the text columns contain non-stopword tokens."
            )

        # Convert sparse matrices to DataFrames
        logger.debug("Converting sparse matrices to dense arrays for CSV storage...")
        train_features: pd.DataFrame = pd.DataFrame(
            X_train_bow.toarray(), columns=feature_names
        )
        test_features: pd.DataFrame = pd.DataFrame(
            X_test_bow.toarray(), columns=feature_names
        )

        # Attach labels
        train_features["label"] = train_df[label_col].values
        test_features["label"] = test_df[label_col].values

        logger.info(
            "Feature extraction finalized: %d features across %d train and %d test rows.",
            len(feature_names),
            len(train_features),
            len(test_features),
        )
        return train_features, test_features
    except MemoryError:
        logger.critical(
            "Out of memory creating dense DataFrame via .toarray(). Consider saving sparse matrices directly."
        )
        raise
    except Exception:
        logger.exception("Unexpected error encountered during TF-IDF feature extraction.")
        raise


def save_processed_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: PathLike
) -> None:
    """Save processed feature datasets to disk."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_out = os.path.join(output_dir, "train_tfidf.csv")
        test_out = os.path.join(output_dir, "test_tfidf.csv")

        logger.info("Writing train features to: %s", train_out)
        train_df.to_csv(train_out, index=False)

        logger.info("Writing test features to: %s", test_out)
        test_df.to_csv(test_out, index=False)

        logger.info(
            "Files written successfully. Sizes -> Train: %.2f MB, Test: %.2f MB",
            os.path.getsize(train_out) / (1024 * 1024),
            os.path.getsize(test_out) / (1024 * 1024),
        )
    except PermissionError:
        logger.error("Permission denied when writing output to: %s", output_dir)
        raise
    except OSError:
        logger.exception("Operating system error writing output files.")
        raise


def main() -> None:
    pipeline_start = time.perf_counter()
    logger.info("=== Starting Feature Engineering Pipeline ===")

    try:
        params: dict[str, Any] = load_params("params.yaml")

        try:
            max_features: int = int(params["feature_engineering"]["max_features"])
        except (KeyError, TypeError) as err:
            logger.error(
                "Missing or invalid 'feature_engineering.max_features' parameter: %s",
                err,
            )
            sys.exit(1)

        train_data, test_data = load_data(
            train_path=os.path.join("data", "interim", "train_processed.csv"),
            test_path=os.path.join("data", "interim", "test_processed.csv"),
        )

        train_tfidf_df, test_tfidf_df = apply_tfidf_features(
            train_df=train_data,
            test_df=test_data,
            max_features=max_features,
        )

        save_processed_data(
            train_df=train_tfidf_df,
            test_df=test_tfidf_df,
            output_dir=os.path.join("data", "processed"),
        )

        total_time = time.perf_counter() - pipeline_start
        logger.info(
            "=== Feature Engineering Pipeline Completed Successfully in %.2f seconds ===",
            total_time,
        )
    except Exception:
        logger.critical(
            "Fatal pipeline failure occurred. Check traceback above.", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()