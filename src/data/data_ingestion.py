import logging
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Logging Setup (FileHandler Only) ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "pipeline.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Formatter defining structure and timing
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# FileHandler directing records exclusively to disk
file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Attach handler to logger (ensuring no stream handler is added)
logger.addHandler(file_handler)
logger.propagate = False  # Prevents passing logs up to the root console logger


def get_test_size() -> float:
    try:
        with open('params.yaml', 'r') as file:
            config = yaml.safe_load(file)
            test_size = float(config['data_ingestion']['test_size'])
            logger.info(f"Retrieved test_size={test_size} from params.yaml")
            return test_size
            
    except FileNotFoundError:
        logger.warning("'params.yaml' file not found. Falling back to default test_size 0.2")
        return 0.2
    except KeyError:
        logger.warning("'test_size' key missing in YAML. Falling back to default test_size 0.2")
        return 0.2


def load_data() -> pd.DataFrame:
    url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
    logger.info("Initiating dataset download from remote URL...")
    try:
        df = pd.read_csv(url)
        logger.info(f"Dataset successfully fetched with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Critical Error downloading data: {e}", exc_info=True)
        raise 


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data preprocessing...")
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'] = final_df['sentiment'].map({'happiness': 1, 'sadness': 0})
        logger.info(f"Data preprocessing finished. Filtered dataset shape: {final_df.shape}")
        return final_df
    except KeyError as e:
        logger.error(f"Error during preprocessing. A required column is missing: {e}", exc_info=True)
        raise


def split_and_save_data(final_df: pd.DataFrame, test_size: float) -> None:
    logger.info(f"Splitting data with test_size={test_size}...")
    try:
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        
        data_path = os.path.join("data", "raw")
        os.makedirs(data_path, exist_ok=True)
        
        train_path = os.path.join(data_path, "train.csv")
        test_path = os.path.join(data_path, "test.csv")
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        logger.info(f"Train split saved to: {train_path} ({len(train_data)} rows)")
        logger.info(f"Test split saved to: {test_path} ({len(test_data)} rows)")
    except OSError as e:
        logger.error(f"File System Error while creating directories or saving CSVs: {e}", exc_info=True)
        raise


def main() -> None:
    logger.info("Starting pipeline execution...")
    try:
        test_size = get_test_size()
        df = load_data()
        final_df = preprocess_data(df)
        split_and_save_data(final_df, test_size)
        logger.info("Data ingestion pipeline completed successfully!")
    except Exception:
        logger.critical("Pipeline execution halted due to an unhandled error.")


if __name__ == '__main__':
    main()