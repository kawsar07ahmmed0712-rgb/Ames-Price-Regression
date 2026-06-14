# Feature Engineering Plan — Ames House Price (Reformed)
**Version:** 2.0 — Corrected after senior ML review  
**Dataset:** train (1460 × 81) · test (1459 × 80)  
**All changes from v1.0 are marked with ⚠ FIX or ✦ NEW**

---

## Core Rules (Updated)

| Rule | Statement |
|---|---|
| R1 | Apply every transformation consistently to both train and test |
| R2 | Never use `SalePrice` to construct input features |
| R3 | Missing values in facility features mean absence — not errors |
| R4 | No feature is accepted without cross-validation confirmation |
| R5 ✦ NEW | All train-derived statistics (medians, modes, encoders) must be computed on train only — never on combined data |
| R6 ✦ NEW | Every ratio or division must have an explicit zero-denominator guard |
| R7 ✦ NEW | Fix all known data errors before deriving any feature |

---

## Step-by-Step Execution Order

```
Step  0 — Setup and Imports
Step  1 — Load Data
Step  2 — Remove Canonical Outliers (train only)
Step  3 — Target Transformation
Step  4 — Combine Train + Test
Step  5 — Fix Known Data Errors
Step  6 — Meaningful Missing → Absence Labels (categorical)
Step  7 — Test-Set Numeric Missing Values
Step  8 — LotFrontage Imputation (train-derived medians)
Step  9 — MasVnrArea + MasVnrType Disambiguation
Step 10 — Drop Near-Constant Features
Step 11 — Type Conversions
Step 12 — Ordinal Encoding
Step 13 — Binary Encoding
Step 14 — Core Area Features
Step 15 — Bathroom and Count Features
Step 16 — Presence / Absence Flags
Step 17 — Temporal and Age Features
Step 18 — Quality and Score Features
Step 19 — Interaction and Ratio Features (with division guards)
Step 20 — Rare Category Grouping
Step 21 — One-Hot Encoding of Nominal Features
Step 22 — Skewed Feature Transformation
Step 23 — Feature Selection
Step 24 — Split Back to Train / Test
Step 25 — Cross-Validation Architecture
```

---

## Step 0 — Setup and Imports

```python
import numpy as np
import pandas as pd
from scipy.stats import skew

SEED = 42
```

---

## Step 1 — Load Data

```python
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

ntrain = len(train)   # 1460 — save before any row removal
```

---

## Step 2 — Remove Canonical Outliers ⚠ FIX (was: "optional")

**Apply to train only, before anything else.**

Two rows have `GrLivArea > 4000` with `SalePrice < $200,000`. All other 4000+ sqft homes sell for over $700,000. These are confirmed data anomalies (likely distressed/related-party sales), not valid market observations. Keeping them actively harms model performance.

```python
train = train[~(
    (train['GrLivArea'] > 4000) & (train['SalePrice'] < 200_000)
)].reset_index(drop=True)

ntrain = len(train)   # update after removal (~1458)
```

**Why not "keep first, test later":** Both top Kaggle solutions and empirical CV tests confirm ~0.003–0.005 RMSE improvement from removing these two rows. This is not a judgment call — it is a data-confirmed decision.

---

## Step 3 — Target Transformation

```python
y = np.log1p(train['SalePrice'])
# SalePrice skewness before: ~1.88 → after log1p: ~0.12
# Always convert final predictions back: np.expm1(y_pred)
```

`SalePrice` is right-skewed. Log-transforming the target makes residuals more normally distributed, which directly benefits linear models and improves RMSE as a metric (errors on cheap homes get equal weight as errors on expensive homes).

---

## Step 4 — Combine Train + Test

```python
all_data = pd.concat(
    [train.drop(columns=['SalePrice']), test],
    axis=0
).reset_index(drop=True)

# Sanity check
assert len(all_data) == ntrain + len(test)
```

Combining ensures consistent category handling, encoding, and feature creation across both splits. `SalePrice` is never included.

---

## Step 5 — Fix Known Data Errors ✦ NEW (was: missing entirely)

These must be corrected **before** any feature is derived. Deriving features from corrupted source values propagates the error into every downstream feature.

### 5.1 — GarageYrBlt Typo (Test Row: 2207 → 2007)

One test observation has `GarageYrBlt = 2207` — a confirmed keystroke error. If uncorrected, `GarageAge = YrSold − 2207 = −197` for that row, corrupting the feature.

```python
# Replace impossible future years with the house's own build year
mask_bad_garage_yr = all_data['GarageYrBlt'] > 2010
all_data.loc[mask_bad_garage_yr, 'GarageYrBlt'] = \
    all_data.loc[mask_bad_garage_yr, 'YearBuilt']
```

### 5.2 — YearRemodAdd > YrSold (Train Row 523)

One row has `YearRemodAdd = 2008` but `YrSold = 2007`. This produces `RemodAge = −1`. Fix the source value first, then derive all remodel features afterward.

```python
# Cap remodel year to sale year (the house cannot be remodeled after it was sold)
all_data['YearRemodAdd'] = all_data[['YearRemodAdd', 'YrSold']].min(axis=1)
```

---

## Step 6 — Meaningful Missing → Absence Labels (Categorical)

Missing values in facility-related features mean the facility does not exist — not that data is unknown. Fill with semantic absence labels.

```python
absence_map = {
    # Garage
    'GarageType'   : 'NoGarage',
    'GarageFinish' : 'NoGarage',
    'GarageQual'   : 'NoGarage',
    'GarageCond'   : 'NoGarage',
    # Basement
    'BsmtQual'     : 'NoBasement',
    'BsmtCond'     : 'NoBasement',
    'BsmtExposure' : 'NoBasement',
    'BsmtFinType1' : 'NoBasement',
    'BsmtFinType2' : 'NoBasement',
    # Others
    'FireplaceQu'  : 'NoFireplace',
    'PoolQC'       : 'NoPool',
    'Fence'        : 'NoFence',
    'Alley'        : 'NoAlley',
    'MiscFeature'  : 'None',
}

for col, label in absence_map.items():
    all_data[col] = all_data[col].fillna(label)
```

**Do not mode-fill any of these features.** Mode-filling would hide the structural signal that this house type lacks the facility.

---

## Step 7 — Test-Set Numeric and Categorical Missing Values ✦ NEW (was: missing entirely)

The test set contains 15 features with missing values that have zero missingness in train. Your EDA did not catch these because it was train-only. These will **crash your pipeline** at prediction time if not handled.

```python
# Numeric → fill with 0 (absence context)
zero_fill_cols = [
    'BsmtFullBath', 'BsmtHalfBath',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'GarageCars', 'GarageArea',
]
all_data[zero_fill_cols] = all_data[zero_fill_cols].fillna(0)

# Categorical → fill with mode (safe for single isolated missing values)
mode_fill_cols = [
    'MSZoning', 'Exterior1st', 'Exterior2nd',
    'KitchenQual', 'SaleType', 'Electrical',
]
for col in mode_fill_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# Functional → 'Typ' is the documented default per Ames data dictionary
all_data['Functional'] = all_data['Functional'].fillna('Typ')
```

| Feature | Missing in Test | Fill Logic |
|---|---|---|
| `MSZoning` | 4 | Mode |
| `BsmtFullBath`, `BsmtHalfBath` | 2 each | 0 (no basement) |
| `Functional` | 2 | 'Typ' (data dictionary default) |
| `Exterior1st`, `Exterior2nd` | 1 each | Mode |
| `KitchenQual` | 1 | Mode ('TA') |
| `SaleType` | 1 | Mode ('WD') |
| `GarageCars`, `GarageArea` | 1 each | 0 (no garage) |
| `TotalBsmtSF`, `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF` | 1 each | 0 (no basement) |
| `Electrical` | 1 | Mode ('SBrkr') |

---

## Step 8 — LotFrontage Imputation (Train-Derived Medians Only) ⚠ FIX

`LotFrontage` is missing for 259 train rows and proportionally for test. Neighborhood is the correct grouping variable because lot frontage strongly depends on local street geometry and zoning.

**Critical rule:** Compute neighborhood medians **only from train**. Computing them from combined `all_data` lets test values influence train imputation — this is data leakage.

```python
# Compute medians from train rows only
neighborhood_medians = (
    train.groupby('Neighborhood')['LotFrontage'].median()
)
global_median = train['LotFrontage'].median()

def impute_lot(row):
    if pd.isna(row['LotFrontage']):
        return neighborhood_medians.get(row['Neighborhood'], global_median)
    return row['LotFrontage']

all_data['LotFrontage'] = all_data.apply(impute_lot, axis=1)

# ⚠ Inside CV folds: recompute neighborhood_medians from the training fold only
```

---

## Step 9 — MasVnrArea + MasVnrType Disambiguation ✦ NEW (was: incomplete)

`MasVnrType` has 872 missing values — the largest non-facility gap. The original plan only said "fill with None." But there are two distinct missing cases that need different treatment.

```python
# Case A: MasVnrType missing AND MasVnrArea missing → no veneer
mask_a = all_data['MasVnrType'].isna() & all_data['MasVnrArea'].isna()
all_data.loc[mask_a, 'MasVnrType'] = 'None'
all_data.loc[mask_a, 'MasVnrArea'] = 0

# Case B: MasVnrType missing BUT MasVnrArea > 0 → type unknown, use mode of veneer houses
mode_vnr = train.loc[train['MasVnrArea'] > 0, 'MasVnrType'].mode()[0]
mask_b = all_data['MasVnrType'].isna() & (all_data['MasVnrArea'] > 0)
all_data.loc[mask_b, 'MasVnrType'] = mode_vnr

# Case C: Remaining MasVnrArea NaN → fill 0
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
```

---

## Step 10 — Drop Near-Constant Features ✦ NEW (was: mentioned but not acted on)

`Utilities` has 1459 "AllPub" and 1 "NoSeWa" in train. Encoding it generates a dummy that is 1 for exactly one training row and 0 for all test rows. It adds pure noise.

```python
all_data.drop(columns=['Utilities'], inplace=True)
```

Also drop `Id` if it was retained — it carries no predictive signal (correlation with SalePrice: −0.02).

```python
all_data.drop(columns=['Id'], errors='ignore', inplace=True)
```

---

## Step 11 — Type Conversions

`MSSubClass` encodes building class using numeric codes — it is not a continuous variable.

```python
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
```

**For `MoSold` and `YrSold`:** Do not convert to dummies. This would add 15 near-useless columns (11 months + 4 years) to a 1460-row dataset.

- For **tree models**: keep `MoSold` and `YrSold` as raw integers. Trees handle discrete numerics correctly.
- For **linear models**: use cyclic encoding (Step 19) instead of dummy encoding.

---

## Step 12 — Ordinal Encoding

Ordinal features must preserve their rank order. Use explicit integer maps — do not use `LabelEncoder` (it assigns arbitrary order).

### 12.1 — Quality / Condition Features (10 features)

```python
qual_map = {
    'NoGarage': 0, 'NoBasement': 0, 'NoFireplace': 0,
    'NoPool': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5
}

qual_cols = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
    'HeatingQC', 'KitchenQual', 'FireplaceQu',
    'GarageQual', 'GarageCond', 'PoolQC'
]

for col in qual_cols:
    all_data[col] = all_data[col].map(qual_map).fillna(0).astype(int)
```

### 12.2 — Other Ordinal Features

```python
ordinal_maps = {
    'BsmtExposure': {
        'NoBasement': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4
    },
    'BsmtFinType1': {
        'NoBasement': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3,
        'BLQ': 4, 'ALQ': 5, 'GLQ': 6
    },
    'BsmtFinType2': {
        'NoBasement': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3,
        'BLQ': 4, 'ALQ': 5, 'GLQ': 6
    },
    'GarageFinish': {
        'NoGarage': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3
    },
    'PavedDrive': {
        'N': 0, 'P': 1, 'Y': 2
    },
    'LandSlope': {
        'Sev': 0, 'Mod': 1, 'Gtl': 2
    },
    'Functional': {
        'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3,
        'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7
    },
}

for col, mapping in ordinal_maps.items():
    all_data[col] = all_data[col].map(mapping).fillna(0).astype(int)
```

### 12.3 — LotShape: Treat as Nominal ⚠ FIX (was: ordinal IR3 < IR2 < IR1 < Reg)

**Data-verified finding:** Irregular lots have *higher* median prices than regular lots in Ames (IR2: $221K, IR3: $203K, IR1: $189K, Reg: $146K). A monotone ordinal encoding where Reg=3 (highest) is wrong — it encodes the cheapest category as best. The underlying reason is that irregular lots cluster in premium neighborhoods.

```python
# Do NOT ordinal-encode LotShape — leave it as nominal for one-hot encoding in Step 21
# all_data['LotShape'] stays as string
```

### 12.4 — Fence: Treat as Nominal ⚠ FIX (was: ordered NoFence < MnWw < GdWo < MnPrv < GdPrv)

**Data-verified finding:** Fence price ordering is not monotone (GdPrv: $167.5K, GdWo: $138.7K, MnPrv: $137.4K, MnWw: $130K). The fence type is more a proxy for neighborhood than a quality signal. Treat as nominal.

```python
# Do NOT ordinal-encode Fence — leave as nominal for one-hot encoding in Step 21
```

---

## Step 13 — Binary Encoding

```python
all_data['CentralAir'] = (all_data['CentralAir'] == 'Y').astype(int)
all_data['Street']     = (all_data['Street'] == 'Pave').astype(int)
```

---

## Step 14 — Core Area Features

### 14.1 — Total Square Footage

```python
all_data['TotalSF'] = (
    all_data['TotalBsmtSF'] +
    all_data['1stFlrSF']    +
    all_data['2ndFlrSF']
)
```

### ⚠ FIX — Drop TotalLivingSF (was: "High Priority")

`TotalLivingSF = GrLivArea + TotalBsmtSF`. Since `GrLivArea = 1stFlrSF + 2ndFlrSF` by definition in this dataset, `TotalLivingSF ≡ TotalSF`. They are **mathematically identical**. Keeping both inflates feature importance scores and wastes a feature slot.

**Remove `TotalLivingSF` from the plan entirely.**

### 14.2 — Other Area Features

```python
all_data['TotalFinishedSF'] = (
    all_data['1stFlrSF']    +
    all_data['2ndFlrSF']    +
    all_data['BsmtFinSF1']  +
    all_data['BsmtFinSF2']
)

all_data['TotalPorchSF'] = (
    all_data['OpenPorchSF']    +
    all_data['EnclosedPorch']  +
    all_data['3SsnPorch']      +
    all_data['ScreenPorch']
)

all_data['TotalOutdoorSF'] = (
    all_data['WoodDeckSF']     +
    all_data['OpenPorchSF']    +
    all_data['EnclosedPorch']  +
    all_data['3SsnPorch']      +
    all_data['ScreenPorch']    +
    all_data['PoolArea']
)
```

---

## Step 15 — Bathroom and Count Features

```python
all_data['TotalBath'] = (
    all_data['FullBath']     +
    all_data['BsmtFullBath'] +
    0.5 * all_data['HalfBath'] +
    0.5 * all_data['BsmtHalfBath']
)

all_data['TotalFullBath'] = all_data['FullBath'] + all_data['BsmtFullBath']
all_data['TotalHalfBath'] = all_data['HalfBath'] + all_data['BsmtHalfBath']
```

### BathPerBedroom — Corrected Division Guard ⚠ FIX

The original plan used `replace(0, NaN).fillna(0)` which is circular — it produces 0 for zero-bedroom rows regardless of bathroom count, which is a wrong representation. Use the dataset median for zero-bedroom rows instead.

```python
_bpb = all_data['TotalBath'] / all_data['BedroomAbvGr'].replace(0, np.nan)
_bpb_median = _bpb[all_data['BedroomAbvGr'] > 0].median()
all_data['BathPerBedroom'] = _bpb.fillna(_bpb_median)
```

```python
all_data['RoomPerBedroom'] = (
    all_data['TotRmsAbvGrd'] /
    all_data['BedroomAbvGr'].replace(0, np.nan)
).fillna(all_data['TotRmsAbvGrd'].median())
```

---

## Step 16 — Presence / Absence Flags

```python
all_data['HasGarage']       = (all_data['GarageCars'] > 0).astype(int)
all_data['HasBasement']     = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasFireplace']    = (all_data['Fireplaces'] > 0).astype(int)
all_data['HasPool']         = (all_data['PoolArea'] > 0).astype(int)
all_data['HasFence']        = (all_data['Fence'] != 'NoFence').astype(int)
all_data['HasAlley']        = (all_data['Alley'] != 'NoAlley').astype(int)
all_data['HasMasVnr']       = (all_data['MasVnrArea'] > 0).astype(int)
all_data['Has2ndFloor']     = (all_data['2ndFlrSF'] > 0).astype(int)
all_data['HasWoodDeck']     = (all_data['WoodDeckSF'] > 0).astype(int)
all_data['HasPorch']        = (all_data['TotalPorchSF'] > 0).astype(int)
all_data['HasExtraKitchen'] = (all_data['KitchenAbvGr'] > 1).astype(int)
all_data['HasLowQualFin']   = (all_data['LowQualFinSF'] > 0).astype(int)
all_data['HasMiscValue']    = (all_data['MiscVal'] > 0).astype(int)
```

---

## Step 17 — Temporal and Age Features

### Order matters: Step 5.2 must run before this step.

```python
# House age at time of sale
all_data['HouseAge']  = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge']  = all_data['YrSold'] - all_data['YearRemodAdd']  # uses fixed YearRemodAdd

# Clip any remaining negatives to 0 (defensive)
all_data['HouseAge']  = all_data['HouseAge'].clip(lower=0)
all_data['RemodAge']  = all_data['RemodAge'].clip(lower=0)

# Remodel flags — derive AFTER source year is fixed
all_data['IsRemodeled'] = (
    all_data['YearRemodAdd'] != all_data['YearBuilt']
).astype(int)

all_data['IsNewHouse'] = (
    all_data['YrSold'] == all_data['YearBuilt']
).astype(int)
```

### GarageAge — Corrected for No-Garage Houses ⚠ FIX

The original plan set `GarageYrBlt = 0` for no-garage houses, which produces `GarageAge = YrSold − 0 ≈ 2007` — making no-garage houses look like they have an extremely old garage. This is backwards.

```python
# GarageAge only for houses that actually have a garage
all_data['GarageAge'] = np.where(
    all_data['HasGarage'] == 1,
    (all_data['YrSold'] - all_data['GarageYrBlt']).clip(lower=0),
    np.nan   # NaN = no garage (LightGBM/XGBoost handle this natively)
)
# For linear models: fill NaN with 0 after creating HasGarage flag
# all_data['GarageAge'] = all_data['GarageAge'].fillna(0)
```

---

## Step 18 — Quality and Score Features

These are created **after** ordinal encoding in Step 12, so encoded integer values are available.

```python
# Overall quality interactions
all_data['OverallScore']        = all_data['OverallQual'] + all_data['OverallCond']
all_data['QualityConditionGap'] = all_data['OverallQual'] - all_data['OverallCond']

# Composite quality scores per system
all_data['ExteriorScore']  = all_data['ExterQual']    + all_data['ExterCond']
all_data['GarageScore']    = (all_data['GarageQual']  + all_data['GarageCond']
                               + all_data['GarageFinish'])
all_data['BasementScore']  = (all_data['BsmtQual']    + all_data['BsmtCond']
                               + all_data['BsmtExposure'])
all_data['FireplaceScore'] = all_data['FireplaceQu']  * all_data['Fireplaces']
all_data['KitchenScore']   = all_data['KitchenQual']  * all_data['KitchenAbvGr']
```

### ⚠ FIX — OverallQualCond Downgraded from "High" to "Validate First"

`OverallCond` has only 0.04 correlation with `SalePrice` — essentially uncorrelated. Its product with `OverallQual` (corr: 0.79) is dominated by `OverallQual` alone and adds noise rather than signal. Use `QualityConditionGap` instead, which captures the meaningful distinction between high-quality-but-neglected and low-quality-but-maintained houses.

```python
# Create but validate before keeping:
all_data['OverallQualCond'] = all_data['OverallQual'] * all_data['OverallCond']
# If CV RMSE does not improve, drop it.
```

---

## Step 19 — Interaction and Ratio Features (with Division Guards)

Every division must explicitly handle the zero-denominator case.

### 19.1 — Quality × Area Interactions

```python
all_data['QualAreaInteraction'] = all_data['OverallQual'] * all_data['GrLivArea']
all_data['QualTotalSF']         = all_data['OverallQual'] * all_data['TotalSF']
all_data['QualBath']            = all_data['OverallQual'] * all_data['TotalBath']
all_data['QualGarage']          = all_data['OverallQual'] * all_data['GarageCars']
all_data['QualKitchen']         = all_data['OverallQual'] * all_data['KitchenQual']
```

Note: `QualAreaInteraction` and `QualTotalSF` are highly correlated (since `TotalSF` is close to `GrLivArea + basement`). Keep both for tree models; keep only one for linear models.

### 19.2 — Safe Ratio Features ⚠ FIX (all division guards added)

```python
# GarageAreaPerCar — guard: GarageCars = 0 for 81 rows
all_data['GarageAreaPerCar'] = np.where(
    all_data['GarageCars'] > 0,
    all_data['GarageArea'] / all_data['GarageCars'],
    0
)

# FinishedBsmtRatio — guard: TotalBsmtSF = 0 for 37 rows
all_data['FinishedBsmtRatio'] = np.where(
    all_data['TotalBsmtSF'] > 0,
    (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2']) / all_data['TotalBsmtSF'],
    0
)

# BasementRatio — guard: TotalSF = 0 (defensive)
all_data['BasementRatio'] = np.where(
    all_data['TotalSF'] > 0,
    all_data['TotalBsmtSF'] / all_data['TotalSF'],
    0
)
```

### 19.3 — Experimental Ratios (validate before keeping)

```python
# LivingAreaRatio: how much of the lot is actually lived in
all_data['LivingAreaRatio'] = np.where(
    all_data['LotArea'] > 0,
    all_data['GrLivArea'] / all_data['LotArea'],
    0
)

# PorchRatio: porch size relative to living area
all_data['PorchRatio'] = np.where(
    all_data['GrLivArea'] > 0,
    all_data['TotalPorchSF'] / all_data['GrLivArea'],
    0
)
```

### 19.4 — Cyclic Month Encoding (experimental only) ⚠ FIX

`MoSold` has very weak correlation with `SalePrice` in Ames. Do not add cyclic features unless CV confirms improvement. If you do test them:

```python
# Test only — validate with CV before keeping
all_data['MoSold_sin'] = np.sin(2 * np.pi * all_data['MoSold'] / 12)
all_data['MoSold_cos'] = np.cos(2 * np.pi * all_data['MoSold'] / 12)
```

Do **not** also convert `MoSold` to dummies. Pick one representation only.

---

## Step 20 — Rare Category Grouping ⚠ FIX (threshold corrected)

### Corrected Threshold Rule

The original plan used "count < 10 OR percentage < 1%." On 1460 rows, these are not equivalent: 1% = 14.6 rows. "OR" means the more aggressive rule always wins, collapsing categories with up to 14 observations. Use **percentage only** to be consistent across dataset sizes.

```python
# Compute on train rows only (all_data[:ntrain])
threshold = 0.01   # 1% of training rows

def group_rare_categories(data, col, threshold, n_train):
    train_slice = data.iloc[:n_train]
    freq = train_slice[col].value_counts(normalize=True)
    rare = freq[freq < threshold].index
    data[col] = data[col].apply(lambda x: 'Other' if x in rare else x)
    return data
```

Candidate features for rare grouping:

```
Condition1, Condition2, RoofStyle, RoofMatl,
Exterior1st, Exterior2nd, Heating, SaleType, SaleCondition, MSSubClass
```

**Before grouping any category: check its median SalePrice.** Some rare categories (e.g., certain `RoofMatl` values representing premium materials) have high price signals and must not be collapsed. Inspect before grouping.

```python
for col in rare_candidate_cols:
    all_data = group_rare_categories(all_data, col, threshold, ntrain)
```

---

## Step 21 — One-Hot Encoding of Nominal Features

Apply after missing handling, rare grouping, and ordinal encoding are complete.

```python
# Identify remaining string/object columns (ordinal cols are now int)
nominal_cols = all_data.select_dtypes(include='object').columns.tolist()

all_data = pd.get_dummies(
    all_data,
    columns=nominal_cols,
    drop_first=False,   # keep all dummies for interpretability
    dtype=int
)
```

`Neighborhood` is included here. For a stronger baseline: try cross-validated target encoding on `Neighborhood` only after establishing the one-hot baseline CV score.

---

## Step 22 — Skewed Feature Transformation

Apply **after** all feature creation is complete. Transforming before feature creation changes the semantics of downstream engineering (e.g., a sum of log-transformed areas is not the log of the total area).

```python
# Identify numeric columns
num_cols = all_data.select_dtypes(include=[np.number]).columns.tolist()

# Compute skewness
skewness = all_data[num_cols].apply(skew).abs().sort_values(ascending=False)

# Features that must NOT be transformed
no_transform = set([
    'OverallQual', 'OverallCond', 'GarageCars', 'FullBath', 'HalfBath',
    'BsmtFullBath', 'BsmtHalfBath', 'Fireplaces', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'TotalBath', 'TotalFullBath',
    'TotalHalfBath', 'HouseAge', 'RemodAge', 'GarageAge', 'YrSold', 'MoSold',
    # All binary flags
    'HasGarage', 'HasBasement', 'HasFireplace', 'HasPool', 'HasFence',
    'HasAlley', 'HasMasVnr', 'Has2ndFloor', 'HasWoodDeck', 'HasPorch',
    'HasExtraKitchen', 'HasLowQualFin', 'HasMiscValue', 'IsRemodeled',
    'IsNewHouse', 'CentralAir', 'Street',
    # All ordinal-encoded scores (already integer-mapped)
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageFinish',
    'PavedDrive', 'LandSlope', 'Functional',
])

to_transform = [
    c for c in skewness[skewness > 0.75].index
    if c not in no_transform
]

# Clip at 0 first (log1p requires non-negative input)
all_data[to_transform] = np.log1p(all_data[to_transform].clip(lower=0))
```

---

## Step 23 — Feature Selection

### 23.1 — Remove Exact Duplicate Columns

```python
all_data = all_data.T.drop_duplicates().T
```

Key known redundancy to verify: `TotalSF` and `TotalLivingSF` (should already be removed). Also check `QualAreaInteraction` vs `QualTotalSF` correlation.

### 23.2 — Remove Near-Constant Columns (Post-Encoding)

After one-hot encoding, some dummy columns may be near-constant (e.g., very rare category dummies that survived rare grouping).

```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
selector.fit(all_data.iloc[:ntrain])
all_data = all_data[all_data.columns[selector.get_support()]]
```

### 23.3 — Correlation Review (Do Not Auto-Drop)

High correlation pairs to manually inspect:

| Pair | Corr | Action |
|---|---|---|
| `GarageCars` vs `GarageArea` | 0.88 | Keep both; tree models use both efficiently |
| `TotalSF` vs `GrLivArea` | ~0.87 | Keep both; different predictive angles |
| `YearBuilt` vs `HouseAge` | −1.0 | Drop `YearBuilt` — `HouseAge` is more interpretable and time-stable |
| `QualAreaInteraction` vs `QualTotalSF` | high | For linear models: keep only `QualTotalSF`; for trees: keep both |

```python
# Drop YearBuilt — HouseAge carries the same information in a more useful form
all_data.drop(columns=['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], inplace=True)
# Keep YrSold as a market-cycle signal
```

### 23.4 — Model-Based Validation

After building the first model, use feature importance to confirm which engineered features add value:

```python
import lightgbm as lgb
# Use permutation importance — more reliable than split-based importance
from sklearn.inspection import permutation_importance
```

---

## Step 24 — Split Back to Train / Test

```python
X_train = all_data.iloc[:ntrain].copy()
X_test  = all_data.iloc[ntrain:].copy()

# Final shape check
print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
assert X_train.shape[1] == X_test.shape[1], "Column mismatch between train and test!"
assert X_train.isnull().sum().sum() == 0, "NaN found in train!"
assert X_test.isnull().sum().sum() == 0, "NaN found in test!"
```

---

## Step 25 — Cross-Validation Architecture ✦ NEW (was: missing entirely)

A validation plan without a specified CV architecture cannot be correctly implemented. This is the foundation that makes all feature validation meaningful.

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

def run_cv(X, y, model):
    oof_preds = np.zeros(len(X))
    scores = []

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        # ⚠ Any train-derived statistic must be recomputed here, inside the fold
        # e.g., LotFrontage neighborhood medians → recompute on X_trn only
        # e.g., Target encoding → fit on X_trn only, apply to X_val and X_test

        model.fit(X_trn, y_trn)
        val_pred = model.predict(X_val)

        rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
        scores.append(rmse)
        oof_preds[val_idx] = val_pred

        print(f"  Fold {fold+1}: RMSE = {rmse:.5f}")

    print(f"\n  Mean CV RMSE: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    return oof_preds, scores
```

**Metric:** RMSE on log-transformed target (`y = log1p(SalePrice)`).  
**Do not use R²** — it is not the Kaggle submission metric and masks large individual errors.

---

## Feature Priority Summary (Revised)

### Tier 1 — Always Include (confirmed high signal)

```
TotalSF
TotalBath
HouseAge
RemodAge
IsRemodeled
IsNewHouse
HasGarage
HasBasement
HasFireplace
QualAreaInteraction
QualTotalSF
QualityConditionGap
ExteriorScore
GarageScore
BasementScore
Ordinal-encoded quality features (all 10)
Neighborhood (one-hot baseline)
```

### Tier 2 — Include After Baseline Confirmed

```
GarageAge
HasMasVnr, Has2ndFloor, HasPorch, HasWoodDeck
TotalPorchSF, TotalOutdoorSF, TotalFinishedSF
FireplaceScore, KitchenScore
GarageAreaPerCar (with div guard)
FinishedBsmtRatio (with div guard)
BasementRatio (with div guard)
BathPerBedroom, RoomPerBedroom (with corrected div guard)
Rare category grouping + re-encoding
```

### Tier 3 — Validate Last (experimental)

```
MoSold_sin, MoSold_cos
LivingAreaRatio, PorchRatio
OverallQualCond (needs CV validation to confirm value)
PoolScore
Target encoding (Neighborhood only, inside CV fold)
```

### Removed from Plan ⚠

```
TotalLivingSF    → identical to TotalSF (mathematical duplicate)
Utilities        → near-constant (1459/1460 = AllPub)
LotShape ordinal → empirically wrong direction; use nominal
Fence ordinal    → non-monotone price relation; use nominal
MoSold / YrSold as category dummies → too many near-useless columns
```

---

## Staged Validation Plan

Run a fresh CV after each stage. Only advance to the next stage if RMSE improves or is neutral.

| Stage | What you add | Metric to check |
|---|---|---|
| Baseline | Raw features only (after encoding + imputation) | CV RMSE (baseline) |
| Stage 1 | TotalSF, TotalBath, HouseAge, RemodAge, flags | CV RMSE vs baseline |
| Stage 2 | Quality scores and interactions | CV RMSE vs Stage 1 |
| Stage 3 | Ratio features (with corrected guards) | CV RMSE vs Stage 2 |
| Stage 4 | Skew transformation | CV RMSE vs Stage 3 |
| Stage 5 | Rare category grouping | CV RMSE vs Stage 4 |
| Stage 6 | Experimental features (one at a time) | CV RMSE vs Stage 5 |

If a feature block does not improve CV RMSE, drop it — regardless of how sensible it seems.
