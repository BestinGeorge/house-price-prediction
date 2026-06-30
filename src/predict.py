"""
Generate predictions on the official (unlabeled) test set and write
outputs/submission.csv in the exact format the Kaggle competition expects:
    Id,SalePrice
"""
import joblib
import numpy as np
import pandas as pd

from . import config
from .data_loader import load_raw_data
from .feature_engineering import build_features


def main():
    print("Loading data and rebuilding features...")
    train_df, test_df = load_raw_data()
    X_train, y_train, X_test, test_ids, scaler = build_features(train_df, test_df)

    model = joblib.load(config.MODELS_DIR / "best_model.joblib")
    with open(config.MODELS_DIR / "best_model_name.txt") as f:
        model_name = f.read().strip()

    print(f"Fitting {model_name} on full training data...")
    model.fit(X_train, y_train)

    print("Predicting on test set...")
    log_preds = model.predict(X_test)
    preds = np.expm1(log_preds)  # invert the log1p transform used on the target
    preds = np.clip(preds, a_min=1000, a_max=None)  # sale prices can't be ~0 or negative

    submission = pd.DataFrame({config.ID_COL: test_ids, config.TARGET: preds})
    out_path = config.OUTPUTS_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)

    print(f"Saved {out_path}")
    print(submission.head())
    print(f"\nPredicted SalePrice summary:\n{submission[config.TARGET].describe()}")


if __name__ == "__main__":
    main()
