"""Load the raw Ames Housing train/test CSVs."""
import pandas as pd

from . import config


def load_raw_data():
    """Return (train_df, test_df) exactly as provided, with Id set aside."""
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    return train_df, test_df


def basic_info(df: pd.DataFrame, name: str = "dataset") -> None:
    """Print a quick sanity-check summary of a dataframe."""
    print(f"--- {name} ---")
    print(f"shape: {df.shape}")
    n_missing = df.isnull().sum()
    n_missing = n_missing[n_missing > 0].sort_values(ascending=False)
    print(f"columns with missing values: {len(n_missing)}")
    if len(n_missing):
        print(n_missing.head(10))
    print()


if __name__ == "__main__":
    train_df, test_df = load_raw_data()
    basic_info(train_df, "train")
    basic_info(test_df, "test")
