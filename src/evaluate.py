"""
Evaluate the saved best model: cross-validated metrics, residual diagnostics,
predicted-vs-actual plot, and (for tree models) feature importance.
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from . import config
from .data_loader import load_raw_data
from .feature_engineering import build_features

sns.set_theme(style="whitegrid")


def load_artifacts():
    model = joblib.load(config.MODELS_DIR / "best_model.joblib")
    with open(config.MODELS_DIR / "best_model_name.txt") as f:
        model_name = f.read().strip()
    return model, model_name


def main():
    print("Loading data and rebuilding features...")
    train_df, test_df = load_raw_data()
    X_train, y_train, X_test, test_ids, scaler = build_features(train_df, test_df)

    model, model_name = load_artifacts()
    print(f"Evaluating: {model_name}")

    # Out-of-fold predictions give an honest estimate of generalization error
    kf = KFold(config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    oof_preds_log = cross_val_predict(model, X_train, y_train, cv=kf, n_jobs=-1)

    rmse_log = np.sqrt(mean_squared_error(y_train, oof_preds_log))
    mae_log = mean_absolute_error(y_train, oof_preds_log)
    r2 = r2_score(y_train, oof_preds_log)

    # Also report in actual dollars (inverse the log1p transform) for interpretability
    y_actual = np.expm1(y_train)
    preds_actual = np.expm1(oof_preds_log)
    rmse_dollars = np.sqrt(mean_squared_error(y_actual, preds_actual))
    mae_dollars = mean_absolute_error(y_actual, preds_actual)

    metrics = {
        "model": model_name,
        "cv_rmse_log_saleprice": rmse_log,
        "cv_mae_log_saleprice": mae_log,
        "cv_r2": r2,
        "cv_rmse_dollars": rmse_dollars,
        "cv_mae_dollars": mae_dollars,
    }
    print("\n=== Out-of-fold metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    pd.Series(metrics).to_csv(config.REPORTS_DIR / "final_model_metrics.csv")

    # --- Predicted vs actual plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(y_train, oof_preds_log, alpha=0.4, s=18, color="#2c7fb8")
    lims = [min(y_train.min(), oof_preds_log.min()), max(y_train.max(), oof_preds_log.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1.5)
    axes[0].set_xlabel("Actual log(SalePrice)")
    axes[0].set_ylabel("Predicted log(SalePrice)")
    axes[0].set_title(f"Predicted vs Actual ({model_name})")

    residuals = y_train - oof_preds_log
    axes[1].scatter(oof_preds_log, residuals, alpha=0.4, s=18, color="#d95f0e")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Predicted log(SalePrice)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual Plot")
    plt.tight_layout()
    plt.savefig(config.OUTPUTS_DIR / "diagnostics_pred_vs_actual.png", dpi=140)
    plt.close()
    print("Saved outputs/diagnostics_pred_vs_actual.png")

    # --- Residual distribution ---
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color="#756bb1")
    plt.title("Residual Distribution")
    plt.xlabel("Residual (log scale)")
    plt.tight_layout()
    plt.savefig(config.OUTPUTS_DIR / "diagnostics_residual_distribution.png", dpi=140)
    plt.close()
    print("Saved outputs/diagnostics_residual_distribution.png")

    # --- Feature importance (if the model supports it) ---
    fitted = model
    fitted.fit(X_train, y_train)
    importances = None
    if hasattr(fitted, "feature_importances_"):
        importances = fitted.feature_importances_
    elif hasattr(fitted, "coef_"):
        importances = np.abs(fitted.coef_)

    if importances is not None:
        imp_df = pd.DataFrame({"feature": X_train.columns, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
        plt.figure(figsize=(8, 8))
        sns.barplot(data=imp_df, y="feature", x="importance", color="#31a354")
        plt.title(f"Top 20 Feature Importances ({model_name})")
        plt.tight_layout()
        plt.savefig(config.OUTPUTS_DIR / "feature_importance.png", dpi=140)
        plt.close()
        imp_df.to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)
        print("Saved outputs/feature_importance.png and reports/feature_importance.csv")
    else:
        print(f"{model_name} does not expose feature importances/coefficients directly "
              "(e.g. it's a stacking ensemble) -- skipping importance plot.")

    print("\nEvaluation complete.")
    return metrics


if __name__ == "__main__":
    main()
