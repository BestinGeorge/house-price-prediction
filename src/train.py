"""
Train, tune, and compare multiple regression models on the engineered
Ames Housing features. Saves the best single model and a stacked ensemble,
plus a CV comparison report.

Models trained:
- Linear Regression (baseline, no regularization)
- Ridge (L2 regularized linear)
- Lasso (L1 regularized linear, also does feature selection)
- ElasticNet (L1 + L2)
- Random Forest
- Gradient Boosting (sklearn)
- XGBoost
- LightGBM
- Stacked ensemble (Ridge/Lasso/GBM/XGB -> Ridge meta-learner)

Everything is scored with 5-fold cross-validated RMSE on log(SalePrice),
matching the actual Kaggle competition metric.
"""
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from . import config
from .data_loader import load_raw_data
from .feature_engineering import build_features

warnings.filterwarnings("ignore")


def rmse_cv(model, X, y, n_folds=config.N_FOLDS):
    kf = KFold(n_folds, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf, n_jobs=-1)
    return np.sqrt(-scores)


def tune_ridge(X, y):
    model = Ridge(random_state=config.RANDOM_STATE)
    param_dist = {"alpha": np.logspace(-2, 2.5, 50)}
    search = RandomizedSearchCV(
        model, param_dist, n_iter=20, scoring="neg_mean_squared_error",
        cv=config.N_FOLDS, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def tune_lasso(X, y):
    model = Lasso(random_state=config.RANDOM_STATE, max_iter=50000)
    param_dist = {"alpha": np.logspace(-4, 0, 50)}
    search = RandomizedSearchCV(
        model, param_dist, n_iter=20, scoring="neg_mean_squared_error",
        cv=config.N_FOLDS, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def tune_elasticnet(X, y):
    model = ElasticNet(random_state=config.RANDOM_STATE, max_iter=50000)
    param_dist = {
        "alpha": np.logspace(-4, 0, 30),
        "l1_ratio": np.linspace(0.05, 0.95, 19),
    }
    search = RandomizedSearchCV(
        model, param_dist, n_iter=25, scoring="neg_mean_squared_error",
        cv=config.N_FOLDS, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def tune_random_forest(X, y):
    model = RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1)
    param_dist = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [None, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5, 0.8],
    }
    search = RandomizedSearchCV(
        model, param_dist, n_iter=20, scoring="neg_mean_squared_error",
        cv=3, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def tune_gradient_boosting(X, y):
    model = GradientBoostingRegressor(random_state=config.RANDOM_STATE)
    param_dist = {
        "n_estimators": [300, 500, 800, 1200],
        "learning_rate": [0.01, 0.02, 0.05, 0.08],
        "max_depth": [2, 3, 4, 5],
        "min_samples_split": [2, 5, 10],
        "subsample": [0.6, 0.8, 1.0],
        "max_features": ["sqrt", "log2", None],
    }
    search = RandomizedSearchCV(
        model, param_dist, n_iter=25, scoring="neg_mean_squared_error",
        cv=3, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def tune_xgboost(X, y):
    model = XGBRegressor(
        random_state=config.RANDOM_STATE, objective="reg:squarederror", n_jobs=-1,
    )
    param_dist = {
        "n_estimators": [400, 600, 800, 1200],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
        "reg_alpha": [0, 0.001, 0.01, 0.1],
        "reg_lambda": [0.5, 1, 1.5, 2],
    }
    search = RandomizedSearchCV(
        model, param_dist, n_iter=25, scoring="neg_mean_squared_error",
        cv=3, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def tune_lightgbm(X, y):
    model = LGBMRegressor(random_state=config.RANDOM_STATE, n_jobs=-1, verbosity=-1)
    param_dist = {
        "n_estimators": [400, 600, 800, 1200],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "num_leaves": [15, 31, 50, 70],
        "max_depth": [-1, 5, 8, 12],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
        "reg_alpha": [0, 0.001, 0.01, 0.1],
        "reg_lambda": [0.5, 1, 1.5, 2],
    }
    search = RandomizedSearchCV(
        model, param_dist, n_iter=25, scoring="neg_mean_squared_error",
        cv=3, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def main():
    print("Loading raw data...")
    train_df, test_df = load_raw_data()

    print("Engineering features...")
    X_train, y_train, X_test, test_ids, scaler = build_features(train_df, test_df)
    print(f"X_train shape: {X_train.shape}")

    results = []
    fitted_models = {}

    tuning_steps = [
        ("LinearRegression", lambda X, y: (LinearRegression().fit(X, y), {})),
        ("Ridge", tune_ridge),
        ("Lasso", tune_lasso),
        ("ElasticNet", tune_elasticnet),
        ("RandomForest", tune_random_forest),
        ("GradientBoosting", tune_gradient_boosting),
        ("XGBoost", tune_xgboost),
        ("LightGBM", tune_lightgbm),
    ]

    for name, tune_fn in tuning_steps:
        print(f"\nTuning {name}...")
        t0 = time.time()
        best_model, best_params = tune_fn(X_train, y_train)
        scores = rmse_cv(best_model, X_train, y_train)
        elapsed = time.time() - t0
        print(
            f"{name}: CV RMSE = {scores.mean():.5f} (+/- {scores.std():.5f}) "
            f"[{elapsed:.1f}s] params={best_params}"
        )
        results.append({
            "model": name,
            "cv_rmse_mean": scores.mean(),
            "cv_rmse_std": scores.std(),
            "best_params": str(best_params),
            "train_seconds": elapsed,
        })
        fitted_models[name] = best_model

    # --- Stacked ensemble of the strongest base learners ---
    print("\nBuilding stacked ensemble...")
    t0 = time.time()
    stack = StackingRegressor(
        estimators=[
            ("ridge", fitted_models["Ridge"]),
            ("lasso", fitted_models["Lasso"]),
            ("gbm", fitted_models["GradientBoosting"]),
            ("xgb", fitted_models["XGBoost"]),
            ("lgbm", fitted_models["LightGBM"]),
        ],
        final_estimator=Ridge(alpha=1.0, random_state=config.RANDOM_STATE),
        cv=config.N_FOLDS,
        n_jobs=-1,
    )
    scores = rmse_cv(stack, X_train, y_train)
    elapsed = time.time() - t0
    print(f"StackedEnsemble: CV RMSE = {scores.mean():.5f} (+/- {scores.std():.5f}) [{elapsed:.1f}s]")
    results.append({
        "model": "StackedEnsemble",
        "cv_rmse_mean": scores.mean(),
        "cv_rmse_std": scores.std(),
        "best_params": "ridge+lasso+gbm+xgb+lgbm -> ridge meta",
        "train_seconds": elapsed,
    })
    fitted_models["StackedEnsemble"] = stack

    # --- Report & select best model ---
    results_df = pd.DataFrame(results).sort_values("cv_rmse_mean").reset_index(drop=True)
    results_df.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    print("\n=== Model comparison (sorted by CV RMSE, lower is better) ===")
    print(results_df[["model", "cv_rmse_mean", "cv_rmse_std", "train_seconds"]].to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    print(f"\nBest model: {best_name}")
    best_model = fitted_models[best_name]
    best_model.fit(X_train, y_train)  # final fit on all training data

    joblib.dump(best_model, config.MODELS_DIR / "best_model.joblib")
    joblib.dump(scaler, config.MODELS_DIR / "scaler.joblib")
    joblib.dump(list(X_train.columns), config.MODELS_DIR / "feature_columns.joblib")
    with open(config.MODELS_DIR / "best_model_name.txt", "w") as f:
        f.write(best_name)

    # Also persist all fitted models, useful for evaluate.py comparisons
    joblib.dump(fitted_models, config.MODELS_DIR / "all_models.joblib")

    print(f"\nSaved best model ({best_name}) to models/best_model.joblib")
    return results_df, best_name


if __name__ == "__main__":
    main()
