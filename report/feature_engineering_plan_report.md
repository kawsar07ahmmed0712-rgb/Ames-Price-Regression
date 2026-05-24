# Feature Engineering Plan — Ames House Price

## 0. Objective

The goal of feature engineering is to convert the EDA findings into modeling-ready features.

Main objectives:

- preserve important raw signals,
- create meaningful combined features,
- handle meaningful missing values correctly,
- reduce skewness where useful,
- encode categorical variables properly,
- avoid leakage,
- validate every major feature block using cross-validation.

This plan is based on the current univariate, bivariate, and multivariate EDA findings.

---

## 1. Feature Engineering Notebook Structure

Recommended notebook structure:

```text
1. Setup and Data Loading
2. Feature Group Import
3. Target Transformation
4. Train-Test Combine for Preprocessing
5. Missing Value Feature Engineering
6. Type Conversion and Category Preparation
7. Ordinal Encoding
8. Core Feature Creation
9. Ratio and Interaction Features
10. Skewed Feature Transformation
11. Rare Category Handling
12. Final Encoding
13. Feature Selection / Low-Variance Check
14. Train-Test Split Back
15. Validation Plan
16. Final Feature Engineering Summary
```

---

## 2. Core Rules

### Rule 1: Do not engineer features only from train data if the same logic must apply to test data

Feature engineering should be applied consistently to both train and test.

### Rule 2: Do not use `SalePrice` to create model input features

Target-based encoding must be done only inside cross-validation if used.

### Rule 3: Missing values are not always errors

Many missing values mean absence of a facility:

- no garage
- no basement
- no fireplace
- no pool
- no fence
- no alley
- no miscellaneous feature

### Rule 4: Feature engineering decisions should be validated

A feature is not automatically good because it makes sense.  
Final acceptance should depend on cross-validation.

---

## 3. Target Transformation

### Plan

Create:

```python
SalePrice_log = np.log1p(SalePrice)
```

### Reason

`SalePrice` is right-skewed.  
Using `log1p(SalePrice)` makes the target more modeling-friendly.

### Decision

- Use original `SalePrice` for interpretation.
- Use `SalePrice_log` for modeling and validation.
- Convert predictions back using `np.expm1()`.

---

## 4. Train-Test Combine

### Plan

Combine train and test before preprocessing feature transformations:

```python
all_data = pd.concat([train.drop(columns=["SalePrice"]), test], axis=0)
```

### Reason

This ensures consistent:

- missing value handling,
- category handling,
- feature creation,
- one-hot encoding,
- rare category grouping.

### Important

Never include `SalePrice` in `all_data`.

---

## 5. Missing Value Feature Engineering

## 5.1 Meaningful Missing Categories

Fill the following missing categorical values with meaningful labels:

| Feature group | Features | Fill value |
|---|---|---|
| Garage-related | `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond` | `NoGarage` |
| Basement-related | `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2` | `NoBasement` |
| Fireplace | `FireplaceQu` | `NoFireplace` |
| Pool | `PoolQC` | `NoPool` |
| Fence | `Fence` | `NoFence` |
| Alley | `Alley` | `NoAlley` |
| Misc feature | `MiscFeature` | `NoMiscFeature` |
| Masonry veneer type | `MasVnrType` | `None` |

### Decision

Do not mode-fill these features.

---

## 5.2 Numerical Missing Values

| Feature | Handling plan | Reason |
|---|---|---|
| `LotFrontage` | Neighborhood-wise median | Lot frontage depends on location/neighborhood. |
| `MasVnrArea` | Fill `0` when `MasVnrType` is `None`/missing | No veneer should have zero area. |
| `GarageYrBlt` | Handle based on garage presence | Missing usually means no garage. |
| Other numeric missing in test | Use group-aware or safe median/zero depending on feature meaning | Test may contain extra missing values. |

### GarageYrBlt plan

```text
If no garage:
    GarageYrBlt = 0
    GarageAge = 0
Else if garage exists but GarageYrBlt missing:
    fill GarageYrBlt with YearBuilt or median garage year
```

Recommended default:

```text
Use YearBuilt for garage year if garage exists but GarageYrBlt is missing.
Use 0 only for no-garage houses.
```

---

## 6. Type Conversion

Some numeric-looking variables are categorical.

### Convert to string/category

```text
MSSubClass
MoSold
YrSold
```

### Reason

- `MSSubClass` is a building class code, not a continuous numeric variable.
- `MoSold` and `YrSold` have limited discrete values and may behave like categories.
- For tree models, raw numeric versions can also be tested, but categorical versions are safer for linear models.

### Recommended approach

Keep both options if needed:

```text
MSSubClass_cat
MoSold_cat
YrSold_cat
```

Then validate.

---

## 7. Ordinal Encoding Plan

Ordinal categorical features should preserve their order.

## 7.1 Quality / Condition Features

Use this order:

```text
NoFacility / Missing = 0
Po = 1
Fa = 2
TA = 3
Gd = 4
Ex = 5
```

Apply to:

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

## 7.2 Other Ordinal Features

| Feature | Order |
|---|---|
| `BsmtExposure` | `NoBasement < No < Mn < Av < Gd` |
| `BsmtFinType1`, `BsmtFinType2` | `NoBasement < Unf < LwQ < Rec < BLQ < ALQ < GLQ` |
| `GarageFinish` | `NoGarage < Unf < RFn < Fin` |
| `PavedDrive` | `N < P < Y` |
| `LotShape` | `IR3 < IR2 < IR1 < Reg` |
| `LandSlope` | `Sev < Mod < Gtl` |
| `Functional` | `Sal < Sev < Maj2 < Maj1 < Mod < Min2 < Min1 < Typ` |
| `Fence` | `NoFence < MnWw < GdWo < MnPrv < GdPrv` |
| `Utilities` | `ELO < NoSeWa < NoSewr < AllPub` |

### Decision

- Ordinal encode these features.
- Do not one-hot encode all ordinal features blindly.
- Keep original category columns only if needed for comparison.

---

## 8. Core Feature Creation

## 8.1 Size / Area Features

| New feature | Formula | Priority |
|---|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | High |
| `TotalLivingSF` | `GrLivArea + TotalBsmtSF` | High |
| `TotalFinishedSF` | `1stFlrSF + 2ndFlrSF + BsmtFinSF1 + BsmtFinSF2` | Medium |
| `TotalPorchSF` | `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch` | Medium |
| `TotalOutdoorSF` | `WoodDeckSF + OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + PoolArea` | Medium |

### Reason

Size features are among the strongest target-related signals.

---

## 8.2 Bathroom / Count Features

| New feature | Formula | Priority |
|---|---|---|
| `TotalBath` | `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath` | High |
| `TotalFullBath` | `FullBath + BsmtFullBath` | Medium |
| `TotalHalfBath` | `HalfBath + BsmtHalfBath` | Medium |
| `BathPerBedroom` | `TotalBath / BedroomAbvGr` | Medium |
| `RoomPerBedroom` | `TotRmsAbvGrd / BedroomAbvGr` | Medium |

### Safe division rule

Use a safe denominator:

```text
If denominator is 0, use NaN or 0, then handle later.
```

Recommended:

```python
BathPerBedroom = TotalBath / BedroomAbvGr.replace(0, np.nan)
BathPerBedroom = BathPerBedroom.fillna(0)
```

---

## 8.3 Presence / Absence Flags

| New feature | Logic | Priority |
|---|---|---|
| `HasGarage` | `GarageCars > 0` or `GarageArea > 0` | High |
| `HasBasement` | `TotalBsmtSF > 0` | High |
| `HasFireplace` | `Fireplaces > 0` | High |
| `HasPool` | `PoolArea > 0` or `PoolQC != NoPool` | Medium |
| `HasFence` | `Fence != NoFence` | Medium |
| `HasAlley` | `Alley != NoAlley` | Low/Medium |
| `HasMasVnr` | `MasVnrArea > 0` or `MasVnrType != None` | Medium |
| `Has2ndFloor` | `2ndFlrSF > 0` | Medium |
| `HasWoodDeck` | `WoodDeckSF > 0` | Medium |
| `HasPorch` | `TotalPorchSF > 0` | Medium |
| `HasExtraKitchen` | `KitchenAbvGr > 1` | Medium |
| `HasLowQualFinSF` | `LowQualFinSF > 0` | Low/Review |
| `HasMiscValue` | `MiscVal > 0` | Low/Review |

### Reason

Zero-heavy features often behave better as existence signals.

---

## 8.4 Temporal / Age Features

| New feature | Formula / logic | Priority |
|---|---|---|
| `HouseAge` | `YrSold - YearBuilt` | High |
| `RemodAge` | `YrSold - YearRemodAdd` | High |
| `GarageAge` | `YrSold - GarageYrBlt` | Medium |
| `IsRemodeled` | `YearRemodAdd != YearBuilt` | High |
| `IsNewHouse` | `YrSold == YearBuilt` | Medium |
| `SoldSeason` | derived from `MoSold` | Experimental |
| `MoSold_sin` | `sin(2*pi*MoSold/12)` | Experimental |
| `MoSold_cos` | `cos(2*pi*MoSold/12)` | Experimental |

### Important consistency check

Before creating `RemodAge`:

```text
If YearRemodAdd > YrSold:
    cap YearRemodAdd to YrSold
```

This is allowed in feature engineering because the EDA identified a logical inconsistency.

### Age rule

Age features should not be negative.

```text
Any negative age should be set to 0 after investigation.
```

---

## 8.5 Quality / Score Features

| New feature | Formula | Priority |
|---|---|---|
| `OverallScore` | `OverallQual + OverallCond` | Medium |
| `OverallQualCond` | `OverallQual * OverallCond` | High |
| `QualityConditionGap` | `OverallQual - OverallCond` | Medium |
| `ExteriorScore` | `ExterQual_encoded + ExterCond_encoded` | Medium |
| `KitchenScore` | `KitchenQual_encoded * KitchenAbvGr` | Medium |
| `GarageScore` | `GarageQual_encoded + GarageCond_encoded + GarageFinish_encoded` | Medium |
| `BasementScore` | `BsmtQual_encoded + BsmtCond_encoded + BsmtExposure_encoded` | Medium |
| `FireplaceScore` | `FireplaceQu_encoded * Fireplaces` | Medium |
| `PoolScore` | `PoolQC_encoded * HasPool` | Low/Review |

### Reason

Quality features show strong target separation.  
Combining quality with existence/count can capture richer signals.

---

## 8.6 Interaction Features

| New feature | Formula | Priority |
|---|---|---|
| `QualAreaInteraction` | `OverallQual * GrLivArea` | High |
| `QualTotalSF` | `OverallQual * TotalSF` | High |
| `QualGarage` | `OverallQual * GarageCars` | Medium |
| `QualBath` | `OverallQual * TotalBath` | Medium |
| `QualKitchen` | `OverallQual * KitchenQual_encoded` | Medium |
| `GarageAreaPerCar` | `GarageArea / GarageCars` | Medium |
| `FinishedBsmtRatio` | `(BsmtFinSF1 + BsmtFinSF2) / TotalBsmtSF` | Medium |
| `BasementRatio` | `TotalBsmtSF / TotalSF` | Medium |
| `LivingAreaRatio` | `GrLivArea / LotArea` | Experimental |
| `PorchRatio` | `TotalPorchSF / GrLivArea` | Experimental |

### Important

Interactions and ratios can help but can also create noise.  
They must be validated.

---

## 9. Skewed Feature Transformation

## 9.1 Target

Use:

```python
SalePrice_log = np.log1p(SalePrice)
```

## 9.2 Numeric Predictors

Apply `log1p` or Box-Cox/Yeo-Johnson only after feature creation.

Recommended approach:

```text
1. Create engineered numeric features.
2. Calculate skewness.
3. Transform features with strong positive skew.
4. Do not transform ordinal scores, count variables, or binary flags.
```

### Candidate features for log1p

| Feature type | Examples |
|---|---|
| Original skewed features | `LotArea`, `GrLivArea`, `1stFlrSF`, `TotalBsmtSF`, `OpenPorchSF` |
| Zero-heavy features | `MasVnrArea`, `WoodDeckSF`, `ScreenPorch`, `PoolArea`, `MiscVal` |
| Engineered area features | `TotalSF`, `TotalLivingSF`, `TotalPorchSF`, `TotalOutdoorSF` |

### Do not transform blindly

Do not transform:

```text
OverallQual
OverallCond
GarageCars
FullBath
HalfBath
Fireplaces
binary flags
ordinal encoded categories
```

---

## 10. Rare Category Handling

## 10.1 Rare Category Grouping

Rare categories can be grouped as `Other`.

Candidate rare-heavy features:

```text
Condition2
RoofMatl
SaleType
Exterior1st
Exterior2nd
Condition1
Heating
RoofStyle
Functional
GarageQual
GarageCond
PoolQC
```

### Rule

Use either:

```text
category count < 10
```

or

```text
category percentage < 1%
```

### Important

Do not group rare categories before checking whether they are meaningful.  
Some rare categories may be high-value property signals.

---

## 10.2 High-Cardinality Features

High-cardinality features:

```text
Neighborhood
Exterior1st
Exterior2nd
MSSubClass
```

Recommended strategies:

| Strategy | Use case |
|---|---|
| One-hot encoding | Simple and safe baseline |
| Rare grouping + one-hot | Reduces sparse columns |
| Cross-validated target encoding | Advanced option; avoid leakage |
| Frequency encoding | Optional for tree models |

### Decision

Start with one-hot + rare grouping.  
Use target encoding only later if needed.

---

## 11. Encoding Strategy

## 11.1 Binary Encoding

| Feature | Encoding |
|---|---|
| `CentralAir` | `Y = 1`, `N = 0` |
| `Street` | `Pave = 1`, `Grvl = 0` |
| engineered flags | `True = 1`, `False = 0` |

## 11.2 Ordinal Encoding

Use predefined ordinal maps.

## 11.3 Nominal Encoding

Use one-hot encoding after missing handling and rare category grouping.

---

## 12. Optional Outlier Review

Do not remove outliers blindly.

Optional train-only review:

```text
Very large GrLivArea with unusually low SalePrice
Very large TotalBsmtSF with unusual SalePrice
Very large LotArea with unusual SalePrice
```

Possible rule:

```text
Only remove if the row is clearly inconsistent and hurts cross-validation.
```

Recommended:

- keep first,
- test model,
- then compare with outlier removal.

---

## 13. Feature Selection Plan

After feature engineering:

### 13.1 Remove exact duplicate columns

Check if two engineered columns are identical.

### 13.2 Remove constant / near-constant features

Examples to review:

```text
Utilities
PoolQC-related encoded columns
very rare category dummies
```

### 13.3 Correlation-based review

Highly correlated features are not removed automatically.

Review:

```text
GarageCars vs GarageArea
TotalSF vs GrLivArea
TotalSF vs TotalLivingSF
YearBuilt vs HouseAge
```

### 13.4 Model-based validation

Use:

- Lasso/Ridge coefficients
- Random Forest / XGBoost / LightGBM feature importance
- permutation importance
- cross-validation score change

---

## 14. Validation Plan

Feature engineering should be added step by step.

| Stage | Feature block | Validate |
|---|---|---|
| Baseline | cleaned raw features | baseline CV RMSE |
| Stage 1 | missing handling + encoding | CV change |
| Stage 2 | target log + skew transformation | CV change |
| Stage 3 | area/count/age features | CV change |
| Stage 4 | flags | CV change |
| Stage 5 | quality interactions | CV change |
| Stage 6 | ratios | CV change |
| Stage 7 | rare grouping / advanced encoding | CV change |

### Main validation metric

Use RMSE on log target.

---

## 15. Priority Plan

## High Priority

Create first:

```text
SalePrice_log
TotalSF
TotalLivingSF
TotalBath
HouseAge
RemodAge
HasGarage
HasBasement
HasFireplace
OverallQualCond
QualAreaInteraction
QualTotalSF
Ordinal encoded quality features
Neighborhood encoding
```

## Medium Priority

Create second:

```text
GarageAge
IsRemodeled
IsNewHouse
HasMasVnr
Has2ndFloor
TotalPorchSF
TotalOutdoorSF
GarageScore
BasementScore
FireplaceScore
GarageAreaPerCar
FinishedBsmtRatio
Rare category grouping
```

## Experimental

Try only after strong baseline:

```text
MoSold_sin
MoSold_cos
SoldSeason
LivingAreaRatio
PorchRatio
PoolScore
target encoding
polynomial features
aggressive outlier removal
```

---

## 16. Final Feature Engineering Decision

The best feature engineering direction for this dataset is:

```text
1. Preserve strong raw predictors.
2. Handle meaningful missing values as absence.
3. Create total size, total bath, age, and facility flag features.
4. Encode ordinal quality features using meaningful rank.
5. Encode nominal features carefully, especially high-cardinality features.
6. Transform only selected skewed numeric features.
7. Validate every feature block with cross-validation.
```

Final feature selection should not depend only on EDA.  
It should be confirmed by model performance.
