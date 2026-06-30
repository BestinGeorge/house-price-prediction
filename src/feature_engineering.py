"""
Feature engineering pipeline for the Ames Housing dataset.

Design choices (and why):
- Train and test are concatenated before transforming so that one-hot encoding
  produces identical columns for both, then split back apart at the end. This
  is standard practice for this exact Kaggle dataset and avoids any leakage
  of the target itself (SalePrice is dropped from train before concatenation
  and never touches the test rows).
- Missing-value handling is informed by data_description.txt, not guessed:
  many NaNs encode "feature absent", not "value unknown".
- A handful of well-known data-entry errors in this specific dataset are
  fixed (e.g. GarageYrBlt = 2207, clearly a typo for 2007).
"""
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.preprocessing import RobustScaler

from . import config

# Known bad data point in the raw Ames dataset (GarageYrBlt typo)
GARAGE_YR_BLT_FIX = {2207: 2007}


def _fix_known_errors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].replace(GARAGE_YR_BLT_FIX)
    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Categorical columns where NaN means "this feature doesn't exist"
    for col in config.NONE_MEANS_ABSENT_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Numeric columns where NaN means "0 of this feature"
    for col in config.ZERO_MEANS_ABSENT_NUMERIC:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # LotFrontage: houses in the same neighborhood tend to have similar
    # frontage, so impute with the neighborhood median rather than the
    # overall median.
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # Electrical / MSZoning / Exterior1st / Exterior2nd / SaleType / KitchenQual /
    # Functional / Utilities: a tiny number of NaNs (1-4 rows) -> fill with the
    # column mode, the standard treatment for rare missingness in an otherwise
    # complete categorical column.
    mode_fill_cols = [
        "Electrical", "MSZoning", "Exterior1st", "Exterior2nd",
        "SaleType", "KitchenQual", "Functional", "Utilities",
    ]
    for col in mode_fill_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Any remaining numeric NaNs -> 0 (none expected after the above, but safe)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Any remaining categorical NaNs -> mode (none expected after the above)
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def _encode_ordinal_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in config.QUALITY_COLS:
        if col in df.columns:
            df[col] = df[col].map(config.QUALITY_MAP).fillna(0).astype(int)

    if "BsmtExposure" in df.columns:
        df["BsmtExposure"] = (
            df["BsmtExposure"].map(config.BSMT_EXPOSURE_MAP).fillna(0).astype(int)
        )
    for col in ["BsmtFinType1", "BsmtFinType2"]:
        if col in df.columns:
            df[col] = df[col].map(config.BSMT_FINTYPE_MAP).fillna(0).astype(int)
    if "GarageFinish" in df.columns:
        df["GarageFinish"] = (
            df["GarageFinish"].map(config.GARAGE_FINISH_MAP).fillna(0).astype(int)
        )
    if "Fence" in df.columns:
        df["Fence"] = df["Fence"].map(config.FENCE_MAP).fillna(0).astype(int)
    if "LotShape" in df.columns:
        df["LotShape"] = df["LotShape"].map(config.LOT_SHAPE_MAP).fillna(0).astype(int)
    if "PavedDrive" in df.columns:
        df["PavedDrive"] = (
            df["PavedDrive"].map(config.PAVED_DRIVE_MAP).fillna(0).astype(int)
        )
    if "Functional" in df.columns:
        df["Functional"] = (
            df["Functional"].map(config.FUNCTIONAL_MAP).fillna(7).astype(int)
        )
    return df


def _engineer_new_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Total living/finished area
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalFinishedSF"] = (
        df["BsmtFinSF1"] + df["BsmtFinSF2"] + df["1stFlrSF"] + df["2ndFlrSF"]
    )

    # Total bathrooms (half baths count as 0.5)
    df["TotalBathrooms"] = (
        df["FullBath"] + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )

    # Total porch area across all porch types
    porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
    df["TotalPorchSF"] = df[porch_cols].sum(axis=1)

    # Age-related features (relative to year sold, not "today")
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).clip(lower=0)
    # Negative ages are data artifacts (e.g. sold the same year built but
    # remod year recorded oddly) -> clip at 0.
    for col in ["HouseAge", "RemodAge"]:
        df[col] = df[col].clip(lower=0)

    df["WasRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
    df["IsNewHouse"] = (df["YrSold"] == df["YearBuilt"]).astype(int)

    # Binary "has X" flags
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    # Quality x condition interaction (overall and exterior)
    df["OverallQualCond"] = df["OverallQual"] * df["OverallCond"]
    df["ExterQualCond"] = df["ExterQual"] * df["ExterCond"]
    df["GarageQualCond"] = df["GarageQual"] * df["GarageCond"]

    # Quality-weighted size: bigger AND nicer houses sell for disproportionately
    # more, so an interaction term helps tree models less but linear models more.
    df["QualGrLivArea"] = df["OverallQual"] * df["GrLivArea"]
    df["QualTotalSF"] = df["OverallQual"] * df["TotalSF"]

    # Polynomial term on the single strongest linear predictor of SalePrice
    df["OverallQual_sq"] = df["OverallQual"] ** 2
    df["GrLivArea_sq"] = df["GrLivArea"] ** 2

    return df


def _force_categorical_dtype(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in config.FORCE_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def _fix_skewed_numeric_features(df: pd.DataFrame, skew_threshold: float = 0.75) -> pd.DataFrame:
    """Box-Cox transform numeric features with high skew, the standard
    treatment for this dataset (most square-footage features are right-skewed)."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Don't transform binary flag / count columns - check actual skew & range
    skews = df[numeric_cols].apply(lambda s: skew(s.dropna()))
    skewed_cols = skews[abs(skews) > skew_threshold].index.tolist()
    lam = 0.15
    for col in skewed_cols:
        # boxcox1p requires non-negative values
        if (df[col] >= 0).all():
            df[col] = boxcox1p(df[col], lam)
    return df


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Full feature engineering pipeline. Returns:
        X_train, y_train (log1p-transformed SalePrice), X_test, test_ids
    """
    train_ids = train_df[config.ID_COL].copy()
    test_ids = test_df[config.ID_COL].copy()

    y_train = np.log1p(train_df[config.TARGET].copy())

    # Drop a couple of well-documented outliers (per the original dataset
    # author's recommendation: very large GrLivArea sold cheaply -- these
    # are partial sales / data entry oddities that distort linear models)
    outlier_idx = train_df[
        (train_df["GrLivArea"] > 4000) & (train_df[config.TARGET] < 300000)
    ].index
    if len(outlier_idx) > 0:
        train_df = train_df.drop(index=outlier_idx)
        y_train = y_train.drop(index=outlier_idx)
        train_ids = train_ids.drop(index=outlier_idx)

    train_features = train_df.drop(columns=[config.TARGET, config.ID_COL])
    test_features = test_df.drop(columns=[config.ID_COL])

    n_train = len(train_features)
    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    combined = _fix_known_errors(combined)
    combined = _handle_missing_values(combined)
    combined = _encode_ordinal_quality_features(combined)
    combined = _engineer_new_features(combined)
    combined = _force_categorical_dtype(combined)
    combined = _fix_skewed_numeric_features(combined)

    combined = pd.get_dummies(combined, drop_first=True)

    X_train = combined.iloc[:n_train, :].reset_index(drop=True)
    X_test = combined.iloc[n_train:, :].reset_index(drop=True)

    # Align just in case (shouldn't be needed since we one-hot encoded jointly)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, y_train.reset_index(drop=True), X_test_scaled, test_ids, scaler


if __name__ == "__main__":
    from .data_loader import load_raw_data

    train_df, test_df = load_raw_data()
    X_train, y_train, X_test, test_ids, scaler = build_features(train_df, test_df)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}")
    X_train.to_csv(config.DATA_PROCESSED_DIR / "X_train.csv", index=False)
    y_train.to_csv(config.DATA_PROCESSED_DIR / "y_train.csv", index=False)
    X_test.to_csv(config.DATA_PROCESSED_DIR / "X_test.csv", index=False)
    test_ids.to_csv(config.DATA_PROCESSED_DIR / "test_ids.csv", index=False)
    print("Saved processed features to data/processed/")
