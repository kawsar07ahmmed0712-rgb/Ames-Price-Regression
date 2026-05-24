# Feature Engineering Plan — Ames House Price (Corrected)

## 0. Objective

The goal of feature engineering is to convert the EDA findings into modeling-ready features.

Main objectives:

- fix all known data errors before deriving any feature,
- preserve important raw signals,
- handle meaningful missing values correctly — including test-only missing values,
- create meaningful combined features,
- apply all transformations using train-derived statistics only,
- reduce skewness where useful,
- encode categorical variables properly,
- guard every ratio and division against zero denominators,
- avoid leakage,
- validate every major feature block using cross-validation.

This plan is based on the EDA findings, the senior ML review findings, and direct data inspection of both train and test sets.

---

## 1. Notebook Structure

Recommended notebook structure:

```text
 0. Setup and Data Loading
 1. Remove Canonical Outliers (train only)
 2. Target Transformation
 3. Train-Test Combine
 4. Fix Known Data Errors
 5. Meaningful Missing Value Handling (categorical)
 6. Test-Set Missing Value Handling (numerical and categorical)
 7. LotFrontage Imputation
 8. MasVnrType and MasVnrArea Disambiguation
 9. Drop Near-Constant Features
10. Type Conversions
11. Ordinal Encoding
12. Binary Encoding
13. Core Area Features
14. Bathroom and Count Features
15. Presence / Absence Flags
16. Temporal and Age Features
17. Quality and Score Features
18. Interaction and Ratio Features
19. Rare Category Grouping
20. One-Hot Encoding
21. Skewed Feature Transformation
22. Feature Selection and Redundancy Check
23. Train-Test Split Back
24. Cross-Validation Architecture
25. Staged Validation Plan
```

---

## 2. Core Rules

### Rule 1: Apply every transformation consistently to both train and test

Feature engineering must produce the same columns, in the same form, for both splits.

### Rule 2: Never use `SalePrice` to construct input features

Target-based encoding must be done only inside cross-validation folds if used at all.

### Rule 3: Missing values in facility features mean absence — not unknown data

These features are missing because the house has no such facility:

- no garage
- no basement
- no fireplace
- no pool
- no fence
- no alley
- no miscellaneous feature

### Rule 4: Feature engineering decisions must be validated

A feature is not automatically good because it makes intuitive sense.  
Final acceptance depends on cross-validation score change.

### Rule 5: All train-derived statistics must be computed from train only

Neighborhood medians, mode values, rare category frequencies, and any encoder fit must come from the training set alone.  
Computing them from combined data leaks test information into train.

### Rule 6: Every division must have an explicit zero-denominator guard

Any ratio feature where the denominator can be zero must define what happens in that case before the feature is created.

### Rule 7: Fix all known data errors before deriving any feature

Source errors propagate into every feature derived from them.  
Data correction must be the first step after combining train and test.

---

## 3. Remove Canonical Outliers

### When

Train only. Before target transformation. Before combining.

### Which rows

Two rows have `GrLivArea` above 4000 with `SalePrice` below $200,000.  
All other houses with `GrLivArea` above 4000 sell for above $700,000.  
These rows are confirmed anomalies — not valid market observations.  
They are likely distressed sales or related-party transactions.

### Decision

Remove both rows from train before any other step.  
Do not apply this filter to test.

### Why not "keep and check later"

This is one of the most widely validated decisions in the Ames Housing Kaggle literature.  
Keeping these rows consistently hurts cross-validation RMSE.  
The decision is data-confirmed, not judgement-based.

---

## 4. Target Transformation

### Plan

Create a log-transformed version of `SalePrice`:

```text
SalePrice_log = log1p(SalePrice)
```

### Reason

`SalePrice` is right-skewed with skewness approximately 1.88.  
Log-transforming the target makes residuals more normally distributed.  
It gives equal weight to relative errors on cheap and expensive houses.

### Decision

- Use original `SalePrice` for final interpretation only.
- Use `SalePrice_log` for all modeling and cross-validation.
- Convert all predictions back using `expm1()` before submission.

---

## 5. Train-Test Combine

### Plan

Combine train (without `SalePrice`) and test into a single dataframe for all preprocessing steps.

### Reason

This ensures consistent:

- missing value handling,
- category handling,
- rare category grouping,
- one-hot encoding column alignment,
- feature creation.

### Important

- Never include `SalePrice` in the combined dataframe.
- Record the number of train rows before combining so the split can be reversed later.
- All statistics fitted during preprocessing must come from train rows only.

---

## 6. Fix Known Data Errors

These corrections must run before any feature derivation.

### 6.1 — GarageYrBlt Typo in Test Set

One test observation has `GarageYrBlt = 2207`.  
This is a confirmed keystroke error — the value should be 2007.  
If uncorrected, any age feature derived from this value will be wildly wrong.

| Feature | Problem | Correct action |
|---|---|---|
| `GarageYrBlt` | One test value = 2207 (impossible future year) | Replace with `YearBuilt` of the same row |

### 6.2 — YearRemodAdd After YrSold

One train row has `YearRemodAdd = 2008` but `YrSold = 2007`.  
A house cannot be remodeled after it was sold.  
This produces a negative `RemodAge` if left uncorrected.

| Feature | Problem | Correct action |
|---|---|---|
| `YearRemodAdd` | One row has value after `YrSold` | Cap `YearRemodAdd` to `YrSold` for all rows |

### Important ordering rule

`YearRemodAdd` must be corrected **before** deriving `IsRemodeled` and `RemodAge`.  
If the order is reversed, `IsRemodeled` will be computed from the wrong source value.

---

## 7. Meaningful Missing Value Handling — Categorical

### Rule

Do not mode-fill any of these features.  
Their missing values carry structural information: the house does not have the facility.

### Fill table

| Feature group | Features | Fill value |
|---|---|---|
| Garage-related | `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond` | `NoGarage` |
| Basement-related | `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2` | `NoBasement` |
| Fireplace | `FireplaceQu` | `NoFireplace` |
| Pool | `PoolQC` | `NoPool` |
| Fence | `Fence` | `NoFence` |
| Alley | `Alley` | `NoAlley` |
| Misc feature | `MiscFeature` | `None` |

---

## 8. Test-Set Missing Value Handling

### Why this section exists

The EDA was conducted on train only.  
The test set contains 15 features with missing values that have zero missingness in train.  
These are invisible during EDA and will crash the pipeline at prediction time if unhandled.

### Numeric features — fill with 0

These are all absence-type missings: a single row is missing basement or garage values because the house has no such facility.

| Feature | Fill value | Reason |
|---|---|---|
| `BsmtFullBath` | 0 | No basement |
| `BsmtHalfBath` | 0 | No basement |
| `BsmtFinSF1` | 0 | No basement |
| `BsmtFinSF2` | 0 | No basement |
| `BsmtUnfSF` | 0 | No basement |
| `TotalBsmtSF` | 0 | No basement |
| `GarageCars` | 0 | No garage |
| `GarageArea` | 0 | No garage |

### Categorical features — fill with mode or documented default

| Feature | Fill value | Reason |
|---|---|---|
| `MSZoning` | Mode from train | 4 missing; RL dominates train |
| `Exterior1st` | Mode from train | 1 missing |
| `Exterior2nd` | Mode from train | 1 missing |
| `KitchenQual` | Mode from train | 1 missing; TA dominates |
| `SaleType` | Mode from train | 1 missing; WD dominates |
| `Electrical` | Mode from train | 1 missing; SBrkr dominates |
| `Functional` | `Typ` | Ames data dictionary default for missing |

### Decision

All mode values must be computed from train rows only.  
Never compute them from the combined dataframe.

---

## 9. LotFrontage Imputation

### Plan

Impute missing `LotFrontage` using the median value of the same `Neighborhood`.

### Reason

Lot frontage depends on local street geometry and zoning.  
Neighborhood-wise medians produce more accurate imputations than a global median.

### Critical leakage rule

Neighborhood medians must be computed from train rows only.  
Computing them from the combined dataframe allows test-row values to influence train imputation.  
This is a data leakage that inflates cross-validation score.

### Decision

- Compute one median per `Neighborhood` from train rows.
- Apply those fixed medians to all rows in the combined dataframe.
- For any neighborhood that appears only in test: use the global train median as fallback.
- Inside cross-validation folds: recompute neighborhood medians from the training fold only.

---

## 10. MasVnrType and MasVnrArea Disambiguation

### Why this section exists

`MasVnrType` has 872 missing values — the largest non-facility gap in the dataset.  
The original plan only said "fill with None" without distinguishing two different missing cases.

### Two cases

| Case | Condition | Correct action |
|---|---|---|
| Case A | `MasVnrType` missing AND `MasVnrArea` missing | Both indicate no veneer — fill `MasVnrType` with `None`, fill `MasVnrArea` with 0 |
| Case B | `MasVnrType` missing BUT `MasVnrArea` is greater than 0 | Veneer exists but type is unknown — fill `MasVnrType` with mode of veneer houses (train only) |

### Decision

Handle the two cases separately.  
Do not apply the same fill to all missing `MasVnrType` rows.

---

## 11. Drop Near-Constant Features

### Features to drop

| Feature | Reason |
|---|---|
| `Utilities` | 1459 of 1460 train rows are `AllPub`. Encoding produces a dummy that is 1 for exactly one training row and 0 for all test rows. Pure noise. |
| `Id` | Identifier column. Correlation with `SalePrice` is −0.02. Carries no predictive information. |

### Decision

Drop both features from the combined dataframe before any encoding step.

---

## 12. Type Conversions

### MSSubClass

`MSSubClass` stores building class codes as integers.  
It is not a continuous numeric variable.  
Convert to string so it will be treated as categorical in encoding steps.

### MoSold and YrSold

Do not convert `MoSold` or `YrSold` to dummy variables.  
This would add approximately 15 near-useless columns to a 1460-row dataset.

| Model type | Recommended treatment |
|---|---|
| Tree models (XGBoost, LightGBM) | Keep as raw integers — trees handle discrete numerics correctly |
| Linear models (Lasso, Ridge) | Use cyclic encoding in the interaction features step |

---

## 13. Ordinal Encoding

Ordinal features must preserve their rank order.  
Use explicit integer maps — do not use automatic label encoders which assign arbitrary order.

### 13.1 — Quality / Condition Features

Standard quality/condition scale applies to 10 features:

| Level | Integer value |
|---|---|
| No facility / missing | 0 |
| Po (Poor) | 1 |
| Fa (Fair) | 2 |
| TA (Typical/Average) | 3 |
| Gd (Good) | 4 |
| Ex (Excellent) | 5 |

Apply this map to:

```text
ExterQual
ExterCond
BsmtQual
BsmtCond
HeatingQC
KitchenQual
FireplaceQu
GarageQual
GarageCond
PoolQC
```

### 13.2 — Other Ordinal Features

| Feature | Order (low to high) |
|---|---|
| `BsmtExposure` | `NoBasement < No < Mn < Av < Gd` |
| `BsmtFinType1` | `NoBasement < Unf < LwQ < Rec < BLQ < ALQ < GLQ` |
| `BsmtFinType2` | `NoBasement < Unf < LwQ < Rec < BLQ < ALQ < GLQ` |
| `GarageFinish` | `NoGarage < Unf < RFn < Fin` |
| `PavedDrive` | `N < P < Y` |
| `LandSlope` | `Sev < Mod < Gtl` |
| `Functional` | `Sal < Sev < Maj2 < Maj1 < Mod < Min2 < Min1 < Typ` |

### 13.3 — LotShape: Treat as Nominal

Original plan ordinal-encoded `LotShape` as `IR3 < IR2 < IR1 < Reg`.  
This is incorrect.

Data-verified median prices:

| LotShape | Median SalePrice |
|---|---|
| IR2 | $221,000 |
| IR3 | $203,570 |
| IR1 | $189,000 |
| Reg | $146,000 |

Irregular lots sell for more than regular lots in this dataset.  
The relationship is not monotone and not consistent with a single ordinal direction.  
The underlying cause is that irregular lots cluster in premium neighborhoods.

**Decision:** Treat `LotShape` as nominal. Leave it for one-hot encoding in step 20.

### 13.4 — Fence: Treat as Nominal

Original plan ordinal-encoded `Fence` as `NoFence < MnWw < GdWo < MnPrv < GdPrv`.  
This is incorrect.

Data-verified median prices:

| Fence | Median SalePrice |
|---|---|
| GdPrv | $167,500 |
| GdWo | $138,750 |
| MnPrv | $137,450 |
| MnWw | $130,000 |

The ordering is non-monotone.  
Fence type appears to be a neighborhood proxy, not a quality signal.

**Decision:** Treat `Fence` as nominal. Leave it for one-hot encoding in step 20.

### Decision

- Ordinal encode only features with a confirmed and correct directional order.
- Do not one-hot encode ordinal features — this wastes the rank information.
- After encoding, all ordinal columns become integers. Any unmapped value should become 0.

---

## 14. Binary Encoding

| Feature | Encoding |
|---|---|
| `CentralAir` | `Y = 1`, `N = 0` |
| `Street` | `Pave = 1`, `Grvl = 0` |
| All engineered flags | `True = 1`, `False = 0` |

---

## 15. Core Area Features

### 15.1 — Total Square Footage

| New feature | Formula | Priority |
|---|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | High |
| `TotalFinishedSF` | `1stFlrSF + 2ndFlrSF + BsmtFinSF1 + BsmtFinSF2` | Medium |
| `TotalPorchSF` | `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch` | Medium |
| `TotalOutdoorSF` | `WoodDeckSF + OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + PoolArea` | Medium |

### Important correction — TotalLivingSF removed

`TotalLivingSF` was defined as `GrLivArea + TotalBsmtSF`.  
Since `GrLivArea = 1stFlrSF + 2ndFlrSF` by definition in this dataset, `TotalLivingSF` is mathematically identical to `TotalSF`.  
Keeping both inflates feature importance, creates a perfect duplicate, and confuses model selection.

**Decision:** Remove `TotalLivingSF` from the plan entirely. Use `TotalSF` only.

---

## 16. Bathroom and Count Features

| New feature | Formula | Priority |
|---|---|---|
| `TotalBath` | `FullBath + 0.5 × HalfBath + BsmtFullBath + 0.5 × BsmtHalfBath` | High |
| `TotalFullBath` | `FullBath + BsmtFullBath` | Medium |
| `TotalHalfBath` | `HalfBath + BsmtHalfBath` | Medium |
| `BathPerBedroom` | `TotalBath / BedroomAbvGr` | Medium |
| `RoomPerBedroom` | `TotRmsAbvGrd / BedroomAbvGr` | Medium |

### Division guard — BathPerBedroom and RoomPerBedroom

6 rows in train have `BedroomAbvGr = 0`.  
The denominator is zero for these rows.

Original plan rule was:

```text
BathPerBedroom = TotalBath / BedroomAbvGr.replace(0, NaN).fillna(0)
```

This is circular: replacing 0 with NaN and then filling NaN with 0 gives the exact same result as doing nothing.  
A house with 0 bedrooms and 2 bathrooms would get `BathPerBedroom = 0`, which is a worse representation than the actual ratio.

**Correct decision:**  
For zero-bedroom rows, fill `BathPerBedroom` with the median of the ratio computed on all non-zero-bedroom rows.  
Apply the same logic to `RoomPerBedroom`.

---

## 17. Presence / Absence Flags

| New feature | Logic | Priority |
|---|---|---|
| `HasGarage` | `GarageCars > 0` | High |
| `HasBasement` | `TotalBsmtSF > 0` | High |
| `HasFireplace` | `Fireplaces > 0` | High |
| `HasPool` | `PoolArea > 0` | Medium |
| `HasFence` | `Fence != NoFence` | Medium |
| `HasAlley` | `Alley != NoAlley` | Low/Medium |
| `HasMasVnr` | `MasVnrArea > 0` | Medium |
| `Has2ndFloor` | `2ndFlrSF > 0` | Medium |
| `HasWoodDeck` | `WoodDeckSF > 0` | Medium |
| `HasPorch` | `TotalPorchSF > 0` | Medium |
| `HasExtraKitchen` | `KitchenAbvGr > 1` | Medium |
| `HasLowQualFin` | `LowQualFinSF > 0` | Low/Review |
| `HasMiscValue` | `MiscVal > 0` | Low/Review |

### Reason

Zero-heavy area and count features often behave better as existence signals.  
The presence of a facility (garage, basement, fireplace) is often more informative than its exact size.

---

## 18. Temporal and Age Features

### Important ordering rule

`YearRemodAdd` must be corrected in step 6 before any feature in this section is derived.  
`GarageYrBlt` must be corrected in step 6 before `GarageAge` is derived.

| New feature | Formula / logic | Priority |
|---|---|---|
| `HouseAge` | `YrSold - YearBuilt` | High |
| `RemodAge` | `YrSold - YearRemodAdd` | High |
| `IsRemodeled` | `YearRemodAdd != YearBuilt` | High |
| `IsNewHouse` | `YrSold == YearBuilt` | Medium |
| `GarageAge` | `YrSold - GarageYrBlt` | Medium |

### Age rule

All age features should be non-negative.  
Clip any result that is still negative to 0 as a defensive measure.

### GarageAge — corrected handling for no-garage houses

The original plan set `GarageYrBlt = 0` for no-garage houses, then computed `GarageAge = YrSold - 0 ≈ 2007`.  
This makes every no-garage house appear to have an extremely old garage, which is the opposite of the intended signal.

**Correct decision:**

| House type | GarageAge value |
|---|---|
| Has garage | `YrSold - GarageYrBlt` (clipped to 0 minimum) |
| No garage | Leave as missing / not applicable |

For tree models (LightGBM, XGBoost): leave as NaN — both handle missing natively.  
For linear models: fill with 0 after `HasGarage` flag is already created.

### Experimental temporal features

| New feature | Logic | Status |
|---|---|---|
| `MoSold_sin` | `sin(2π × MoSold / 12)` | Experimental — validate with CV first |
| `MoSold_cos` | `cos(2π × MoSold / 12)` | Experimental — validate with CV first |

`MoSold` has very weak correlation with `SalePrice` in Ames.  
Add these only if cross-validation confirms improvement.  
Do not add both cyclic encoding and dummy encoding for the same feature.

---

## 19. Quality and Score Features

These features must be created **after** ordinal encoding in step 13.  
The encoded integer values are required for these calculations.

| New feature | Formula | Priority |
|---|---|---|
| `OverallScore` | `OverallQual + OverallCond` | Medium |
| `QualityConditionGap` | `OverallQual - OverallCond` | Medium/High |
| `ExteriorScore` | `ExterQual_enc + ExterCond_enc` | Medium |
| `GarageScore` | `GarageQual_enc + GarageCond_enc + GarageFinish_enc` | Medium |
| `BasementScore` | `BsmtQual_enc + BsmtCond_enc + BsmtExposure_enc` | Medium |
| `FireplaceScore` | `FireplaceQu_enc × Fireplaces` | Medium |
| `KitchenScore` | `KitchenQual_enc × KitchenAbvGr` | Medium |
| `PoolScore` | `PoolQC_enc × HasPool` | Low/Review |

### Note on OverallQualCond (priority corrected)

Original plan listed `OverallQual × OverallCond` as high priority.  
This is incorrect.

`OverallCond` has a correlation of only 0.04 with `SalePrice`.  
Its product with `OverallQual` (correlation: 0.79) is almost entirely dominated by `OverallQual`.  
The product adds noise, not signal.

`QualityConditionGap = OverallQual - OverallCond` is more meaningful.  
It captures the difference between high-quality-but-neglected and low-quality-but-maintained houses.

**Decision:** Create `QualityConditionGap` as medium/high priority.  
Create `OverallQual × OverallCond` as experimental only, and validate before keeping.

---

## 20. Interaction and Ratio Features

### 20.1 — Quality × Area Interactions

| New feature | Formula | Priority |
|---|---|---|
| `QualAreaInteraction` | `OverallQual × GrLivArea` | High |
| `QualTotalSF` | `OverallQual × TotalSF` | High |
| `QualBath` | `OverallQual × TotalBath` | Medium |
| `QualGarage` | `OverallQual × GarageCars` | Medium |
| `QualKitchen` | `OverallQual × KitchenQual_enc` | Medium |

Note: `QualAreaInteraction` and `QualTotalSF` will be highly correlated because `TotalSF` contains `GrLivArea`.  
For tree models: keep both — trees tolerate correlated features.  
For linear models: keep only `QualTotalSF`.

### 20.2 — Ratio Features (all require division guards)

| New feature | Formula | Division guard | Priority |
|---|---|---|---|
| `GarageAreaPerCar` | `GarageArea / GarageCars` | If `GarageCars = 0`: result = 0 | Medium |
| `FinishedBsmtRatio` | `(BsmtFinSF1 + BsmtFinSF2) / TotalBsmtSF` | If `TotalBsmtSF = 0`: result = 0 | Medium |
| `BasementRatio` | `TotalBsmtSF / TotalSF` | If `TotalSF = 0`: result = 0 | Medium |
| `LivingAreaRatio` | `GrLivArea / LotArea` | If `LotArea = 0`: result = 0 | Experimental |
| `PorchRatio` | `TotalPorchSF / GrLivArea` | If `GrLivArea = 0`: result = 0 | Experimental |

### Division guard — how many rows are affected

| Feature | Rows with zero denominator |
|---|---|
| `GarageAreaPerCar` | 81 rows (`GarageCars = 0`) |
| `FinishedBsmtRatio` | 37 rows (`TotalBsmtSF = 0`) |

These are not edge cases — they are a significant portion of the dataset.  
A missing guard will produce `inf` or `NaN` for these rows and corrupt model training.

---

## 21. Rare Category Grouping

### Threshold rule (corrected)

Original plan used either `count < 10` OR `percentage < 1%`.  
These are not equivalent on a 1460-row dataset (1% = 14.6 rows).  
Using OR means the more aggressive rule always wins.

**Correct decision:** Use percentage only.

```text
Rare threshold: category appears in fewer than 1% of training rows
```

Rare categories are grouped as `Other`.

### Candidate features for rare grouping

```text
Condition1
Condition2
RoofStyle
RoofMatl
Exterior1st
Exterior2nd
Heating
SaleType
SaleCondition
MSSubClass
```

### Important check before grouping

Do not automate rare category grouping blindly.  
Before collapsing any category, inspect its median `SalePrice`.  
Some rare categories represent premium materials or special conditions and carry meaningful price signals.  
If a rare category has a distinctly different median price from its neighbors, consider keeping it separately.

### Leakage rule

Rare frequency thresholds must be computed from train rows only.  
Apply the result (which categories are rare) consistently to the full combined dataframe.

---

## 22. One-Hot Encoding

Apply after:

- all missing value steps are complete,
- all ordinal and binary encoding is complete,
- all rare category grouping is complete.

Apply to all remaining string / object columns.  
This includes: all nominal categoricals, `LotShape`, `Fence`, `MSSubClass`, `Neighborhood`, and all other unencoded string features.

### High-cardinality features

| Feature | Recommended strategy |
|---|---|
| `Neighborhood` | One-hot encoding as baseline; cross-validated target encoding as advanced option |
| `Exterior1st`, `Exterior2nd` | Rare grouping first, then one-hot |
| `MSSubClass` | One-hot after conversion to string in step 10 |

### Decision

Start with one-hot encoding plus rare grouping as the baseline.  
Use target encoding only if it improves cross-validation score over the one-hot baseline.  
If target encoding is used, it must be applied inside cross-validation folds — never fit on the full training set and applied to the same training set.

---

## 23. Skewed Feature Transformation

Apply only after all feature creation steps are complete.  
Transforming features before engineering changes the semantics of sums, ratios, and interactions built from them.

### Target

Already transformed in step 3. Do not re-transform.

### Numeric predictors

```text
1. Identify all numeric columns.
2. Compute absolute skewness for each.
3. Apply log1p to features with absolute skewness above 0.75.
4. Clip all values to zero minimum before applying log1p — the function requires non-negative input.
5. Do not transform the exclusion list below.
```

### Do not transform

```text
OverallQual, OverallCond
GarageCars
FullBath, HalfBath, BsmtFullBath, BsmtHalfBath
Fireplaces, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd
TotalBath, TotalFullBath, TotalHalfBath
HouseAge, RemodAge, GarageAge
YrSold, MoSold
All binary flags (HasGarage, HasBasement, etc.)
All ordinal-encoded features (now integers 0–5 or similar)
All engineered score features
```

### Reason

Ordinal scores, count variables, age features, and binary flags are already on meaningful bounded scales.  
Log-transforming them destroys their interpretability and their semantic meaning.

---

## 24. Feature Selection

### 24.1 — Remove Exact Duplicate Columns

Check whether any two columns are identical after all engineering steps.

Known redundancy confirmed by data inspection:

| Pair | Status |
|---|---|
| `TotalSF` vs `TotalLivingSF` | Mathematically identical — `TotalLivingSF` must already be removed in step 15 |

### 24.2 — Remove Near-Constant Columns

After one-hot encoding, some dummy columns may be near-constant because the category they represent is extremely rare.  
Use a variance threshold to identify and remove these.

Already removed before encoding:

```text
Utilities  — near-constant (1459/1460 = AllPub)
```

### 24.3 — Correlation Review (do not auto-drop)

| Pair | Approximate correlation | Recommendation |
|---|---|---|
| `GarageCars` vs `GarageArea` | 0.88 | Keep both for tree models |
| `TotalSF` vs `GrLivArea` | ~0.87 | Keep both — different angles |
| `YearBuilt` vs `HouseAge` | −1.0 | Drop `YearBuilt` — `HouseAge` is more interpretable |
| `YearRemodAdd` vs `RemodAge` | −1.0 | Drop `YearRemodAdd` — `RemodAge` is more interpretable |
| `GarageYrBlt` vs `GarageAge` | −1.0 | Drop `GarageYrBlt` — `GarageAge` is more interpretable |
| `QualAreaInteraction` vs `QualTotalSF` | High | For linear models: keep one; for trees: keep both |

### Decision for year columns

After age features are created, the original year columns become redundant.  
Drop `YearBuilt`, `YearRemodAdd`, and `GarageYrBlt` from the final feature set.  
Keep `YrSold` — it captures the market cycle and is not redundant with any derived feature.

### 24.4 — Model-Based Validation

After building the first model, use feature importance to confirm which engineered features contribute.

Recommended methods:

```text
Lasso / Ridge coefficients (for linear models)
LightGBM / XGBoost split-based feature importance
Permutation importance (more reliable than split-based)
Cross-validation RMSE change from removing the feature
```

---

## 25. Train-Test Split Back

After all preprocessing is complete, split the combined dataframe back into train and test using the saved row count.

### Sanity checks before proceeding

```text
1. Train and test must have identical column counts.
2. Train must have zero missing values.
3. Test must have zero missing values.
4. No column in train or test should be all-zero (check for dead dummies).
```

If any check fails, trace the problem back to the step that caused it before modeling.

---

## 26. Cross-Validation Architecture

### Specification

```text
Method:   K-Fold
Folds:    5
Shuffle:  Yes
Seed:     42 (fixed for reproducibility)
Metric:   RMSE on log-transformed target
```

### Why K-Fold and not Stratified K-Fold

`SalePrice` is continuous. Stratified K-Fold is for classification.  
Standard K-Fold with shuffle is the correct choice for a regression problem.

### What must happen inside each fold

The following statistics must be recomputed on the training fold only:

```text
LotFrontage neighborhood medians   — fit on training fold, apply to validation fold
Rare category frequencies          — determined from training fold
Target encoding (if used)          — fit on training fold, apply to validation fold and test
```

This is not optional. Computing these statistics outside the fold leaks validation information into training and makes your CV score optimistic.

### Metric

```text
RMSE on log(1 + SalePrice)
```

Do not use R² as the primary metric.  
It is not the Kaggle submission metric and it masks large individual errors.

---

## 27. Staged Validation Plan

Feature engineering should be added and validated incrementally.  
Do not add all features at once and tune from there.

| Stage | Feature block added | What to check |
|---|---|---|
| Baseline | Raw cleaned features only (encoded, imputed) | Baseline CV RMSE |
| Stage 1 | TotalSF, TotalBath, HouseAge, RemodAge, IsRemodeled, IsNewHouse | CV change vs baseline |
| Stage 2 | Presence / absence flags | CV change vs Stage 1 |
| Stage 3 | Quality scores and interactions | CV change vs Stage 2 |
| Stage 4 | Ratio features (with division guards) | CV change vs Stage 3 |
| Stage 5 | Skew transformation | CV change vs Stage 4 |
| Stage 6 | Rare category grouping | CV change vs Stage 5 |
| Stage 7 | Experimental features (one at a time) | CV change vs Stage 6 |

### Decision rule

If a feature block does not improve CV RMSE, remove it.  
Do not keep features because they seem reasonable.  
Keep only what the cross-validation confirms.

---

## 28. Priority Plan (Corrected)

### High Priority — Create First

```text
SalePrice_log
TotalSF                           ← replaces both TotalSF and TotalLivingSF
TotalBath
HouseAge
RemodAge
IsRemodeled
HasGarage
HasBasement
HasFireplace
QualityConditionGap               ← replaces OverallQualCond as high priority
QualAreaInteraction
QualTotalSF
Ordinal-encoded quality features (all 10)
Neighborhood one-hot encoding
```

### Medium Priority — Create Second

```text
GarageAge                         ← with corrected no-garage handling
IsNewHouse
Has2ndFloor
HasMasVnr
HasPorch
HasWoodDeck
TotalPorchSF
TotalOutdoorSF
TotalFinishedSF
ExteriorScore
GarageScore
BasementScore
FireplaceScore
KitchenScore
GarageAreaPerCar                  ← with division guard
FinishedBsmtRatio                 ← with division guard
BasementRatio                     ← with division guard
BathPerBedroom                    ← with corrected division guard
RoomPerBedroom                    ← with corrected division guard
Rare category grouping
```

### Experimental — Validate Last

```text
MoSold_sin, MoSold_cos
LivingAreaRatio, PorchRatio
OverallQual × OverallCond         ← validate before keeping; likely dominated by OverallQual
PoolScore
Target encoding on Neighborhood   ← only if one-hot baseline CV is established first
```

### Removed from Plan

| Feature | Reason |
|---|---|
| `TotalLivingSF` | Mathematically identical to `TotalSF` — duplicate |
| `Utilities` encoding | Near-constant — dropped before encoding |
| `LotShape` ordinal encoding | Empirically wrong direction — now treated as nominal |
| `Fence` ordinal encoding | Non-monotone price relation — now treated as nominal |
| `MoSold` / `YrSold` dummy encoding | Too many near-useless columns for a 1460-row dataset |

---

## 29. Final Feature Engineering Decision

The correct feature engineering direction for this dataset is:

```text
1. Fix all known data errors first, before deriving any feature.
2. Handle test-set missing values explicitly — they are different from train.
3. Impute LotFrontage using train-only neighborhood medians.
4. Preserve strong raw predictors.
5. Handle meaningful missing values as structural absence.
6. Create TotalSF (not TotalLivingSF), TotalBath, age features, and facility flags.
7. Encode ordinal quality features using the correct directional rank.
8. Treat LotShape and Fence as nominal — their price ordering is not monotone.
9. Apply division guards to every ratio feature before creating it.
10. Encode nominal features carefully — one-hot baseline first.
11. Transform only selected skewed numeric features, after all engineering is complete.
12. Validate every feature block with cross-validation before finalizing.
```

Final feature selection must not depend only on EDA or intuition.  
It must be confirmed by cross-validation score.
