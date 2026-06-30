# House Price Prediction (Regression + Feature Engineering)

Predicts residential home sale prices in Ames, Iowa using the real **House Prices: Advanced
Regression Techniques** dataset (the Ames Housing dataset, 1460 training homes / 79 explanatory
features). No synthetic data is used anywhere in this project — every row is a real, sold house.

## What this project does

1. **Loads & inspects** the raw data (`data/raw/train.csv`, `data/raw/test.csv`), using
   `data_description.txt` to interpret what every column and missing value actually means.
2. **Handles missing values** the right way — `NaN` doesn't always mean "missing"; for many
   columns (e.g. `PoolQC`, `Fence`, `GarageType`) it means "the house doesn't have that feature",
   so it's encoded as `"None"` / `0` rather than imputed statistically.
3. **Engineers features**: total square footage, total bathrooms, house age, remodel age,
   has-pool / has-garage / has-fireplace flags, quality×condition interactions, polynomial terms
   for the strongest predictors, and skewness correction (Box-Cox) on the numeric features and
   the target (`SalePrice` is log-transformed since it's right-skewed).
4. **Encodes categoricals**: ordinal encoding for quality-scale features (e.g. `Ex/Gd/TA/Fa/Po`),
   one-hot encoding for nominal features.
5. **Scales features** with `RobustScaler` (robust to the outliers present in real housing data).
6. **Trains multiple models**: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest,
   Gradient Boosting, XGBoost, and LightGBM, plus a stacked ensemble.
7. **Tunes hyperparameters** with `RandomizedSearchCV` / `GridSearchCV` and 5-fold cross-validation.
8. **Evaluates** with RMSE (on log SalePrice — the actual Kaggle competition metric), MAE, and R²,
   and produces diagnostic plots (residuals, predicted-vs-actual, feature importance).
9. **Generates a submission file** (`outputs/submission.csv`) with predictions on the official
   unseen test set, in the exact format the Kaggle competition expects.

## Project structure

```
house_price_project/
├── data/
│   ├── raw/                  # original train.csv, test.csv, data_description.txt
│   └── processed/            # cleaned/engineered feature matrices (generated)
├── src/
│   ├── config.py             # paths & constants
│   ├── data_loader.py        # load raw csvs
│   ├── feature_engineering.py# missing values, encoding, new features, scaling
│   ├── train.py              # trains + tunes all models, saves the best one
│   ├── evaluate.py           # diagnostics, plots, metrics report
│   └── predict.py            # generates outputs/submission.csv from the saved model
├── notebooks/
│   └── eda.ipynb             # exploratory data analysis
├── models/                   # saved trained models (generated)
├── outputs/                  # submission.csv + diagnostic plots (generated)
├── reports/
│   └── model_comparison.csv  # CV scores for every model tried
├── requirements.txt
└── run_pipeline.py           # runs the whole pipeline end-to-end
```

## How to run

```bash
pip install -r requirements.txt
python run_pipeline.py
```

This will: load data → engineer features → train & tune all models → pick the best one by
cross-validated RMSE → evaluate it → save diagnostic plots → write `outputs/submission.csv`.

Or run stages individually:

```bash
python src/train.py        # trains and tunes all models, saves reports/model_comparison.csv
python src/evaluate.py     # loads the best saved model, produces diagnostics
python src/predict.py      # produces outputs/submission.csv
```

You can also open `notebooks/eda.ipynb` for the exploratory analysis (distributions, correlations,
missing-value patterns, outlier inspection) that informed the feature engineering decisions.

## Dataset

Ames Housing dataset, distributed via the Kaggle competition **House Prices: Advanced Regression
Techniques** (https://www.kaggle.com/c/house-prices-advanced-regression-techniques). 1460 labeled
training homes, 1459 unlabeled test homes, 79 explanatory variables describing almost every
aspect of each house (lot size, quality ratings, square footage by area, garage/basement details,
sale conditions, etc.).

## Notes on realism / honesty about results

- The test set (`data/raw/test.csv`) has **no `SalePrice` column** — that's the actual unseen
  holdout Kaggle uses to score submissions. This project reports cross-validated performance on
  the training set as the honest measure of model quality, the same as the original competition
  intends.
- Expect a cross-validated RMSE (log scale) somewhere around **0.11–0.13** for a tuned gradient
  boosting / stacked model — that's a realistic, competitive score for this dataset, not a
  cherry-picked number.
