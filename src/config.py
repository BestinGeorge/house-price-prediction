"""
Project-wide configuration: paths, constants, and the ordinal-quality maps
used for encoding the many "quality / condition" columns in this dataset.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = ROOT_DIR / "reports"

TRAIN_PATH = DATA_RAW_DIR / "train.csv"
TEST_PATH = DATA_RAW_DIR / "test.csv"
DATA_DESCRIPTION_PATH = DATA_RAW_DIR / "data_description.txt"

for d in (DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "SalePrice"
ID_COL = "Id"
RANDOM_STATE = 42
N_FOLDS = 5

# ---------------------------------------------------------------------------
# Columns where NaN genuinely means "this feature does not exist on the
# house" rather than "value unknown". Per data_description.txt these should
# be filled with "None" (categorical) or 0 (numeric), NOT imputed.
# ---------------------------------------------------------------------------
NONE_MEANS_ABSENT_CATEGORICAL = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

ZERO_MEANS_ABSENT_NUMERIC = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]

# ---------------------------------------------------------------------------
# Ordinal "quality scale" columns -> numeric mapping
# Ex=Excellent, Gd=Good, TA=Average/Typical, Fa=Fair, Po=Poor, None=absent
# ---------------------------------------------------------------------------
QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}

QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]

BSMT_EXPOSURE_MAP = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
BSMT_FINTYPE_MAP = {
    "GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0,
}
GARAGE_FINISH_MAP = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}
FENCE_MAP = {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "None": 0}
LOT_SHAPE_MAP = {"Reg": 3, "IR1": 2, "IR2": 1, "IR3": 0}
PAVED_DRIVE_MAP = {"Y": 2, "P": 1, "N": 0}
FUNCTIONAL_MAP = {
    "Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4,
    "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0,
}

# A handful of columns are actually categorical codes even though they're
# stored as integers (e.g. MSSubClass is a class code, not a magnitude).
FORCE_CATEGORICAL = ["MSSubClass", "MoSold", "YrSold"]
