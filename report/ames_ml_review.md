# Ames House Price — Senior ML Review
**Reviewed by:** Senior ML Engineer / Kaggle Grandmaster Perspective  
**Dataset:** Ames Housing (train: 1460 × 81, test: 1459 × 80)  
**Scope:** EDA reports + Feature Engineering plan + raw data audit

---

## Executive Summary

Your work is structurally sound and shows genuine ML awareness — meaningful-absence imputation, ordinal mapping, train-test combine, and log-target transformation are all correct instincts. However, there are **seven confirmed data bugs**, **three data-leakage risks**, **multiple incorrect ordinal directions**, and **a systematic gap in test-data missing-value coverage** that will silently break your pipeline at prediction time. Several engineered features as specified will produce `NaN` or `inf` at runtime. The feature selection plan is also missing a critical redundancy analysis.

Each issue below includes: what it is, why it hurts, and exactly how to fix it.

---

## Part 1 — Confirmed Data Bugs (Data-Verified)

### Bug 1 — GarageYrBlt = 2207 in the Test Set (Not Caught in EDA)

**What:** Test observation with `GarageYrBlt = 2207` — a typographical data entry error (should be 2007).  
**Where:** Test set only; not present in train. Your EDA was train-only, so you missed it.  
**Why it hurts:** When you compute `GarageAge = YrSold - GarageYrBlt`, this row yields `GarageAge = 2010 - 2207 = -197`. Your FE plan says "age features should not be negative" and clips to 0, but that silently discards valid information for this observation rather than correcting the source error.  
**Fix:**
```python
all_data.loc[all_data['GarageYrBlt'] > 2010, 'GarageYrBlt'] = all_data['YearBuilt']
# Then compute GarageAge after this correction
```

---

### Bug 2 — YearRemodAdd > YrSold (Row 523 in Train)

**What:** One row has `YearRemodAdd = 2008` but `YrSold = 2007`. Your EDA correctly identified this but classified it as "note only."  
**Why it hurts:** `RemodAge = YrSold - YearRemodAdd = -1` for this row. Your FE plan says to cap `YearRemodAdd` to `YrSold`, but your code should do this *before* computing `IsRemodeled`, otherwise `IsRemodeled = (YearRemodAdd != YearBuilt) = True` which is correct — but `RemodAge` is wrong. The fix must be ordered.  
**Fix (ordered correctly):**
```python
# Step 1: fix the source
all_data['YearRemodAdd'] = all_data[['YearRemodAdd', 'YrSold']].min(axis=1)
# Step 2: then derive features
all_data['IsRemodeled'] = (all_data['YearRemodAdd'] != all_data['YearBuilt']).astype(int)
all_data['RemodAge']    = all_data['YrSold'] - all_data['YearRemodAdd']
```

---

### Bug 3 — BedroomAbvGr = 0 Creates Silent NaN (6 Train Rows)

**What:** 6 rows have `BedroomAbvGr = 0`. Your FE plan specifies:  
```python
BathPerBedroom = TotalBath / BedroomAbvGr.replace(0, np.nan).fillna(0)
```
This is circular — you replace 0 with NaN then fill NaN with 0, giving exactly the same result as dividing by 0 and catching the error. The feature becomes 0 for all no-bedroom rows regardless of bathroom count.  
**Why it hurts:** A studio apartment with 2 bathrooms gets `BathPerBedroom = 0`, which is a worse representation than leaving the feature as NaN or using a sentinel.  
**Fix:**
```python
# For 0-bedroom rows, BathPerBedroom is undefined — fill with the dataset median
bpb = train['TotalBath'] / train['BedroomAbvGr']
median_bpb = bpb[train['BedroomAbvGr'] > 0].median()
all_data['BathPerBedroom'] = (all_data['TotalBath'] / all_data['BedroomAbvGr']
                               .replace(0, np.nan)).fillna(median_bpb)
```

---

### Bug 4 — FinishedBsmtRatio Divides by Zero (37 Rows)

**What:** 37 rows have `TotalBsmtSF = 0` (no basement). Your FE plan computes:  
```python
FinishedBsmtRatio = (BsmtFinSF1 + BsmtFinSF2) / TotalBsmtSF
```
This produces `inf` or `NaN` for no-basement houses. There is no mention of this edge case in the FE plan.  
**Fix:**
```python
all_data['FinishedBsmtRatio'] = np.where(
    all_data['TotalBsmtSF'] > 0,
    (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2']) / all_data['TotalBsmtSF'],
    0  # no basement → ratio is 0, which is semantically correct
)
```

---

### Bug 5 — GarageAreaPerCar Divides by Zero (81 Rows)

**What:** 81 rows have `GarageCars = 0`. Your FE plan specifies `GarageArea / GarageCars` with no edge-case handling for this feature (the safe-division comment only appears under `BathPerBedroom`).  
**Fix:**
```python
all_data['GarageAreaPerCar'] = np.where(
    all_data['GarageCars'] > 0,
    all_data['GarageArea'] / all_data['GarageCars'],
    0
)
```

---

### Bug 6 — BasementRatio Uses Wrong Denominator (37 Rows)

**What:** `BasementRatio = TotalBsmtSF / TotalSF`. For no-basement houses, `TotalBsmtSF = 0` so the ratio is 0, which is fine. But `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF` — if somehow `TotalSF = 0` (edge case in test), you get `inf`. More importantly, this ratio is nearly redundant with `HasBasement` for the 37 no-basement rows and with `FinishedBsmtRatio` for the rest. Validate it adds signal before keeping it.  
**Partial fix:**
```python
all_data['BasementRatio'] = np.where(
    all_data['TotalSF'] > 0,
    all_data['TotalBsmtSF'] / all_data['TotalSF'],
    0
)
```

---

### Bug 7 — MasVnrType Has 872 Missing in Train But Your Missing-Handling Table Is Incomplete

**What:** `MasVnrType` has 872 missing values in train — the largest non-facility-related gap — yet your FE plan only says fill with `"None"`. The interaction with `MasVnrArea` is underspecified. When `MasVnrType = None` and `MasVnrArea > 0`, you have a contradiction.  
**Confirmed in data:** 8 rows have `MasVnrArea > 0` and `MasVnrType` missing (not explicitly None). These need disambiguation.  
**Fix:**
```python
# Case 1: MasVnrType is genuinely None → area must be 0
mask_none = all_data['MasVnrType'].isin(['None', np.nan]) & all_data['MasVnrArea'].isna()
all_data.loc[mask_none, 'MasVnrArea'] = 0
all_data.loc[mask_none, 'MasVnrType'] = 'None'

# Case 2: MasVnrType missing but area > 0 → mode impute type
mask_type_missing = all_data['MasVnrType'].isna() & (all_data['MasVnrArea'] > 0)
mode_type = train.loc[train['MasVnrArea'] > 0, 'MasVnrType'].mode()[0]
all_data.loc[mask_type_missing, 'MasVnrType'] = mode_type
```

---

## Part 2 — Data Leakage Risks

### Leakage Risk 1 — SaleCondition Should Be Reviewed as a Quasi-Leakage Variable

**What:** `SaleCondition = "Partial"` (125 rows, 8.5% of train) has a median SalePrice of **$244,600** — vs. $160,000 for Normal sales. "Partial" means the house was not complete at time of sale (new construction). This is a known Ames Housing dataset peculiarity.  
**Why it is a leakage risk:** In real-world deployment, you would know the sale condition at prediction time (you're pricing a listing), so including it is technically valid. However, the "Partial" signal is almost entirely explained by `IsNewHouse` and `YearBuilt` — including both causes multicollinearity and makes your model appear to have found a unique signal when it hasn't. More importantly, if you ever use target encoding on `SaleCondition` without proper CV folds, it leaks the target heavily due to its strong and category-imbalanced correlation.  
**Fix:** Keep `SaleCondition` but never target-encode it without stratified K-fold CV. Consider whether it adds anything beyond `IsNewHouse + OverallQual`.

---

### Leakage Risk 2 — Neighborhood-Wise Median Imputation for LotFrontage Must Be Computed on Train Only

**What:** Your FE plan correctly prescribes neighborhood-median imputation for `LotFrontage`. But the implementation must use medians computed **only from the training fold**, not from the combined `all_data`.  
**Why it hurts:** If you compute the Neighborhood median on `all_data` (train + test combined), test-set values of `LotFrontage` influence the imputation used on training rows. This is a classic leakage that subtly deflates your CV score (makes it look better than it is).  
**Fix:**
```python
# Compute medians only from train
neighborhood_medians = train.groupby('Neighborhood')['LotFrontage'].median()

# Apply to all_data using those train-derived medians
def impute_lotfrontage(row):
    if pd.isna(row['LotFrontage']):
        return neighborhood_medians.get(row['Neighborhood'], train['LotFrontage'].median())
    return row['LotFrontage']

all_data['LotFrontage'] = all_data.apply(impute_lotfrontage, axis=1)
```
In a proper CV loop, this must be recomputed **inside each fold**.

---

### Leakage Risk 3 — Target Encoding Without Proper Fold Isolation

**What:** Your FE plan mentions target encoding as an "advanced option" for `Neighborhood` and high-cardinality features, with a note to "avoid leakage." But the plan does not specify the mechanism.  
**Why it hurts:** Naive target encoding (even on training data) causes leakage because each training row's encoding is computed using its own target value. This is a well-known failure mode.  
**Fix — use leave-one-out or CV-based encoding:**
```python
from category_encoders import TargetEncoder
# Must be applied inside each CV fold:
# In fold k: fit encoder on (train - fold_k), transform fold_k and test
# Never fit on the full training set and transform the same full training set
```
For a Kaggle competition, the safest approach is: **one-hot + rare grouping first**. Add target encoding only if it improves OOF score, and always within the CV loop.

---

## Part 3 — Ordinal Encoding Errors (Data-Verified)

### Error 1 — LotShape Direction Is Inverted Relative to Price

**What:** Your FE plan maps: `IR3 < IR2 < IR1 < Reg` (ascending = more regular = higher).  
**Data reality:**
```
LotShape  Median SalePrice
IR1       $189,000
IR2       $221,000
IR3       $203,570
Reg       $146,000   ← lowest
```
Irregular lots actually command *higher* prices in Ames, likely because irregular lots are more common in premium neighborhoods. A monotone ordinal encoding from "Reg=0 → IR3=3" would encode Reg as lowest quality, which matches price reality — but the direction label in your plan ("Reg" as the highest) implies the opposite.  
**Fix:** This is a case where ordinal encoding by natural ordering is wrong. Either:
- Use one-hot encoding for `LotShape` (safest), or
- Encode by empirical median: `Reg=0, IR1=1, IR3=2, IR2=3` (match price, not shape regularity), or
- Let the model treat it as nominal.

---

### Error 2 — Fence Ordinal Assumption Is Unsubstantiated

**What:** Your plan maps: `NoFence < MnWw < GdWo < MnPrv < GdPrv`.  
**Data reality:**
```
Fence   Median SalePrice
GdPrv   $167,500
GdWo    $138,750
MnPrv   $137,450
MnWw    $130,000
```
The ordering is not monotone: `GdPrv` is highest but `GdWo > MnPrv` is reversed from your encoding. More importantly, the ~80% who have no fence have a range of prices that overlap all fence categories. Fence appears to be **a neighborhood/area proxy**, not a quality signal.  
**Fix:** Treat `Fence` as nominal or drop it after CV validation. It is likely a low-value feature.

---

### Error 3 — Functional Ordinal Direction

**What:** Your FE plan maps: `Sal < Sev < Maj2 < Maj1 < Mod < Min2 < Min1 < Typ`.  
This is the **correct** direction per Ames data dictionary. However, the plan does **not** handle the test set which contains 2 missing `Functional` values. Missing `Functional` in test is not addressed.  
**Fix:**
```python
# For test missing Functional: impute with mode (almost always 'Typ')
all_data['Functional'] = all_data['Functional'].fillna('Typ')
```

---

## Part 4 — Test Set Missing Values Unaddressed in Plan (Critical Gap)

Your EDA is 100% train-focused. The test set has **15 features with missing values that have zero missingness in train**. Your FE plan does not handle these. This will crash your pipeline at prediction time.

| Feature | Missing in Test | Missing in Train | Required Fix |
|---|---|---|---|
| `MSZoning` | 4 | 0 | Mode impute or model with MSSubClass |
| `BsmtFullBath` | 2 | 0 | Fill 0 (no basement context) |
| `BsmtHalfBath` | 2 | 0 | Fill 0 |
| `Functional` | 2 | 0 | Fill 'Typ' |
| `Utilities` | 2 | 0 | Fill 'AllPub' (near-constant anyway) |
| `Exterior1st` | 1 | 0 | Mode impute |
| `Exterior2nd` | 1 | 0 | Mode impute |
| `KitchenQual` | 1 | 0 | Mode impute ('TA') |
| `SaleType` | 1 | 0 | Mode impute ('WD') |
| `GarageCars` | 1 | 0 | Fill 0 (likely no-garage row) |
| `GarageArea` | 1 | 0 | Fill 0 |
| `TotalBsmtSF` | 1 | 0 | Fill 0 |
| `BsmtFinSF1` | 1 | 0 | Fill 0 |
| `BsmtFinSF2` | 1 | 0 | Fill 0 |
| `BsmtUnfSF` | 1 | 0 | Fill 0 |

Add explicit test-set imputation as a separate final step in your pipeline, **after** all train-based computation.

---

## Part 5 — Feature Engineering Weaknesses

### Weakness 1 — TotalSF and TotalLivingSF Are Heavily Redundant

**What:** You propose both:
- `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
- `TotalLivingSF = GrLivArea + TotalBsmtSF`

Note that `GrLivArea = 1stFlrSF + 2ndFlrSF` — this is a fixed identity in the Ames dataset. So `TotalLivingSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF = TotalSF`. **These are the same feature with different names.**  
**Fix:** Keep `TotalSF`. Drop `TotalLivingSF`. You also have `QualTotalSF = OverallQual * TotalSF` and `QualAreaInteraction = OverallQual * GrLivArea` — both will survive in tree models, but for linear models they cause multicollinearity. Pick one.

---

### Weakness 2 — OverallQualCond Is a Weak Interaction

**What:** `OverallQualCond = OverallQual * OverallCond` is listed as "High priority."  
**Why it is questionable:** `OverallQual` has correlation 0.79 with SalePrice. `OverallCond` has correlation only 0.04 with SalePrice. Their product is dominated by `OverallQual`. `OverallCond` is essentially uncorrelated with price because high-quality new houses and well-maintained old houses score similarly on condition but very differently on price. The product adds noise, not signal.  
**Recommended instead:**
```python
# These are more meaningful:
QualityConditionGap = OverallQual - OverallCond  # detects overbuilt vs. neglected
IsHighQualLowCond   = (OverallQual >= 7) & (OverallCond <= 4)  # distressed luxury
```
Validate with CV before keeping the product.

---

### Weakness 3 — GarageAge = 0 for No-Garage Houses Creates a False Signal

**What:** Your plan sets `GarageYrBlt = 0` for no-garage houses, then computes `GarageAge = YrSold - 0 = ~2007`. This gives no-garage houses a `GarageAge` of approximately 2007, which will look like an extremely old garage to the model — the opposite of no garage.  
**Fix:**
```python
all_data['GarageAge'] = np.where(
    all_data['HasGarage'] == 1,
    all_data['YrSold'] - all_data['GarageYrBlt'],
    np.nan  # or a sentinel like -1 that the model can recognize
)
# For tree models, NaN is handled natively in LightGBM/XGBoost
# For linear models, fill with 0 after creating HasGarage flag
```

---

### Weakness 4 — Cyclic Month Encoding Is Likely Low Value

**What:** You propose `MoSold_sin` and `MoSold_cos` under "Experimental." This is theoretically correct for cyclic variables. However, `MoSold` in Ames Housing data has very weak correlation with `SalePrice` (most top Kaggle solutions drop it or find it near-zero importance). The cyclic encoding adds 2 features for minimal gain.  
**Recommendation:** Run a simple CV test with and without `MoSold`. If RMSE doesn't improve, drop it entirely. Don't add complexity without validation.

---

### Weakness 5 — Rare Category Threshold Is Imprecise

**What:** Your plan gives two conflicting thresholds: "count < 10" OR "percentage < 1%." On 1460 rows, 1% = 14.6 rows. "Count < 10" and "< 1%" are not the same rule, and "OR" means whichever is more aggressive applies — which on a 1460-row dataset gives count < 10, i.e., groups representing < 0.68%. This is very aggressive and may merge categories that have distinct price signals (e.g., certain `RoofMatl` categories that represent luxury materials).  
**Recommendation:** Use **percentage-based threshold only** (`< 1%`), and always inspect the merged categories' price distributions before collapsing them. Never automate rare grouping without a manual review step.

---

### Weakness 6 — MoSold and YrSold Converted to Category But No Clear Benefit

**What:** Your plan converts `MoSold` and `YrSold` to string/category for linear models. This generates dummy variables, adding 4 dummies for `YrSold` (5 years) and 11 for `MoSold`. That is 15 near-noise features for a dataset of 1460 rows. For tree models, numeric is fine. For linear models, this inflates the feature space.  
**Recommendation:** For linear models, use cyclic encoding for `MoSold` and treat `YrSold` as numeric. For tree models, keep raw numeric. Do not add 15 dummies unless CV confirms improvement.

---

### Weakness 7 — Utilities Feature Must Be Dropped Before Encoding

**What:** `Utilities` has 1459 "AllPub" and 1 "NoSeWa" in train — it is near-constant. In test, 2 values are missing. If you one-hot encode this, you get a dummy column that is 1 for exactly 1 training row, which will be 0 for all test rows. This feature is worthless and adds noise.  
**Fix:** Drop `Utilities` before encoding. Add an explicit check for near-constant features (variance < threshold) as a pre-encoding step.

---

## Part 6 — Missing from the Plan

### Missing 1 — No Cross-Validation Architecture Is Specified

Your FE plan says "validate with cross-validation" repeatedly but never defines:
- How many folds?
- Stratified or not?
- What is the fold seed?
- Is LotFrontage neighborhood-median computed inside the fold?

**Industry standard for Ames Housing:**
```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# All train-set-derived statistics (medians, encoders, scalers) must be
# fit inside the training fold and applied to the validation fold.
```

---

### Missing 2 — No Outlier Removal Decision for the Two Known Outliers

Two rows (IDs 524 and 1299) are the canonical Ames outliers: `GrLivArea > 4000` with anomalously low `SalePrice` (~$160K and ~$184K for `OverallQual = 10` homes in Edwards). These are not "large houses" — they are data anomalies (likely foreclosures or related party sales).  
**Data confirmation:** All other `GrLivArea > 4000` homes have `SalePrice > $700K`. Rows 524/1299 break this pattern by >50%.  
**Recommendation:** Remove these two rows from training. This is one of the most widely validated single decisions in Ames Housing Kaggle solutions and typically improves CV RMSE by 0.003–0.005.

```python
train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000))]
```

---

### Missing 3 — No Pipeline Abstraction

Your FE plan is a sequence of notebook steps, not a reusable pipeline. This creates two problems:
1. You cannot guarantee the exact same transformations are applied to train and test without rerunning the entire notebook.
2. CV is impossible to implement correctly without encapsulating fit/transform logic.

**Recommended structure:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Each transformation step should be a class or function that accepts
# (X_train, X_test) or implements fit(X_train) / transform(X)
# so it can be used inside CV folds correctly.
```

For a Kaggle solution, a minimal requirement is to separate:
- `fit_transformers(train_fold)` — computes all statistics from train
- `apply_transformers(data, fitted_stats)` — applies them to any split

---

### Missing 4 — No Model Stacking or Blending Strategy

Top Ames Housing solutions blend 3–5 models. Your plan mentions Lasso/Ridge/XGB/LGBM for feature importance but not for the final prediction. A common strong baseline:
```python
final_pred = 0.3 * lasso_pred + 0.3 * ridge_pred + 0.2 * xgb_pred + 0.2 * lgbm_pred
# Or use StackingRegressor with a linear meta-learner
```

---

## Part 7 — Priority Fix Order

| Priority | Issue | Impact |
|---|---|---|
| 🔴 Critical | GarageYrBlt = 2207 typo in test | Produces -197 age; corrupts GarageAge feature |
| 🔴 Critical | Test-set missing values (15 features) | Pipeline crash at prediction time |
| 🔴 Critical | LotFrontage imputation leakage | CV score inflation; unseen neighborhoods in test |
| 🔴 Critical | TotalSF = TotalLivingSF (identical features) | Model confusion, inflated feature importance |
| 🟠 High | GarageAge = 0 for no-garage | False ~2007 age signal for no-garage rows |
| 🟠 High | FinishedBsmtRatio div-by-zero | NaN/inf for 37 rows; corrupts predictions |
| 🟠 High | GarageAreaPerCar div-by-zero | NaN/inf for 81 rows |
| 🟠 High | Remove the 2 canonical outliers | ~0.003–0.005 CV RMSE improvement |
| 🟡 Medium | LotShape ordinal direction | Encodes Reg as best when price says opposite |
| 🟡 Medium | BathPerBedroom div-by-zero | Wrong value for 6 zero-bedroom rows |
| 🟡 Medium | SaleCondition target encoding risk | Leakage if target encoded naively |
| 🟡 Medium | YearRemodAdd fix ordering | IsRemodeled computed before source fix |
| 🟡 Medium | Utilities near-constant | Near-useless dummy; adds noise |
| 🟢 Low | Fence ordinal assumption | Treat as nominal; validate with CV |
| 🟢 Low | MoSold cyclic encoding | Likely near-zero importance; validate first |
| 🟢 Low | OverallQualCond product | Likely dominated by OverallQual alone |
| 🟢 Low | No CV architecture spec | Cannot do correct CV without this |

---

## Part 8 — Recommended Revised Pipeline Structure

```python
# ── 0. Load ─────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

# ── 1. Remove canonical outliers (train only) ────────────────────────────
train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000))]

# ── 2. Target transformation ─────────────────────────────────────────────
y = np.log1p(train['SalePrice'])

# ── 3. Combine for preprocessing ─────────────────────────────────────────
ntrain = len(train)
all_data = pd.concat([train.drop('SalePrice', axis=1), test], axis=0).reset_index(drop=True)

# ── 4. Fix known data errors ─────────────────────────────────────────────
all_data.loc[all_data['GarageYrBlt'] > 2010, 'GarageYrBlt'] = all_data['YearBuilt']
all_data['YearRemodAdd'] = all_data[['YearRemodAdd','YrSold']].min(axis=1)

# ── 5. Meaningful missing → absence labels ───────────────────────────────
cat_fill_map = {
    'GarageType':'NoGarage', 'GarageFinish':'NoGarage',
    'GarageQual':'NoGarage', 'GarageCond':'NoGarage',
    'BsmtQual':'NoBasement', 'BsmtCond':'NoBasement',
    'BsmtExposure':'NoBasement', 'BsmtFinType1':'NoBasement', 'BsmtFinType2':'NoBasement',
    'FireplaceQu':'NoFireplace', 'PoolQC':'NoPool',
    'Fence':'NoFence', 'Alley':'NoAlley', 'MiscFeature':'None',
}
for col, val in cat_fill_map.items():
    all_data[col] = all_data[col].fillna(val)

# Test-only numeric missings
zero_fill = ['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2',
             'BsmtUnfSF','TotalBsmtSF','GarageCars','GarageArea']
all_data[zero_fill] = all_data[zero_fill].fillna(0)
mode_fill = ['MSZoning','Exterior1st','Exterior2nd','KitchenQual',
             'SaleType','Functional','Electrical']
for col in mode_fill:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
all_data['Utilities'] = all_data['Utilities'].fillna('AllPub')

# ── 6. LotFrontage (train-derived medians only) ──────────────────────────
nhood_medians = train.groupby('Neighborhood')['LotFrontage'].median()
mask = all_data['LotFrontage'].isna()
all_data.loc[mask, 'LotFrontage'] = (
    all_data.loc[mask, 'Neighborhood'].map(nhood_medians)
    .fillna(train['LotFrontage'].median())
)

# MasVnrArea
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')

# ── 7. Drop near-constant features ───────────────────────────────────────
all_data.drop(columns=['Utilities'], inplace=True)

# ── 8. Feature engineering ───────────────────────────────────────────────
# Area
all_data['TotalSF']       = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalPorchSF']  = (all_data['OpenPorchSF'] + all_data['EnclosedPorch'] +
                              all_data['3SsnPorch'] + all_data['ScreenPorch'])

# Bathrooms
all_data['TotalBath'] = (all_data['FullBath'] + 0.5*all_data['HalfBath'] +
                          all_data['BsmtFullBath'] + 0.5*all_data['BsmtHalfBath'])

# Age (use fixed YearRemodAdd)
all_data['HouseAge']   = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge']   = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['IsRemodeled'] = (all_data['YearRemodAdd'] != all_data['YearBuilt']).astype(int)
all_data['IsNewHouse']  = (all_data['YrSold'] == all_data['YearBuilt']).astype(int)
all_data['GarageAge']  = np.where(
    all_data['GarageType'] != 'NoGarage',
    all_data['YrSold'] - all_data['GarageYrBlt'],
    np.nan
)

# Flags
all_data['HasGarage']    = (all_data['GarageCars'] > 0).astype(int)
all_data['HasBasement']  = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)
all_data['Has2ndFloor']  = (all_data['2ndFlrSF'] > 0).astype(int)
all_data['HasPorch']     = (all_data['TotalPorchSF'] > 0).astype(int)

# Interactions
all_data['QualAreaInteraction'] = all_data['OverallQual'] * all_data['GrLivArea']
all_data['QualTotalSF']         = all_data['OverallQual'] * all_data['TotalSF']
all_data['QualityConditionGap'] = all_data['OverallQual'] - all_data['OverallCond']

# Safe ratios
all_data['BathPerBedroom'] = (
    all_data['TotalBath'] / all_data['BedroomAbvGr'].replace(0, np.nan)
).fillna(all_data['TotalBath'].median())

all_data['GarageAreaPerCar'] = np.where(
    all_data['GarageCars'] > 0,
    all_data['GarageArea'] / all_data['GarageCars'], 0
)

all_data['FinishedBsmtRatio'] = np.where(
    all_data['TotalBsmtSF'] > 0,
    (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2']) / all_data['TotalBsmtSF'], 0
)

# ── 9. Ordinal encoding ───────────────────────────────────────────────────
qual_map = {'NoFacility':0, 'NoGarage':0, 'NoBasement':0,
            'NoFireplace':0, 'NoPool':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
qual_cols = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',
             'KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']
for col in qual_cols:
    all_data[col] = all_data[col].map(qual_map).fillna(0)

# ... (other ordinal maps as specified in FE plan)

# ── 10. One-hot encoding ──────────────────────────────────────────────────
# Drop LotShape from ordinal; treat as nominal
nominal_cols = [c for c in all_data.select_dtypes('object').columns]
all_data = pd.get_dummies(all_data, columns=nominal_cols, drop_first=False)

# ── 11. Skew transformation ───────────────────────────────────────────────
from scipy.stats import skew
num_cols = all_data.select_dtypes(include=[np.number]).columns
skewed   = all_data[num_cols].apply(skew).sort_values(ascending=False)
skewed   = skewed[abs(skewed) > 0.75]
# Exclude ordinal/count/binary/engineered score columns
exclude  = {'OverallQual','OverallCond','GarageCars','FullBath','HalfBath',
            'Fireplaces','TotalBath','HasGarage','HasBasement', ...}
to_transform = [c for c in skewed.index if c not in exclude]
all_data[to_transform] = np.log1p(all_data[to_transform].clip(lower=0))

# ── 12. Split back ────────────────────────────────────────────────────────
X_train = all_data[:ntrain]
X_test  = all_data[ntrain:]
```

---

## Summary of Key Takeaways

1. **Your conceptual framework is correct** — meaningful missingness, log target, ordinal encoding, train-test combine. The approach is sound.
2. **The implementation has seven confirmed runtime bugs** that will either crash your pipeline or silently corrupt features. Fix these before anything else.
3. **Test-set coverage is the biggest blind spot.** Always run your full pipeline on test before trusting your CV score.
4. **TotalSF and TotalLivingSF are identical.** This is the most wasteful mistake — it inflates feature importance and confuses model selection.
5. **The two canonical outliers should be removed.** This is one of the highest ROI decisions in Ames Housing modeling.
6. **LotFrontage imputation must be inside the CV loop.** Currently it is a leakage risk.
7. **CV architecture must be fully specified** before adding more features. A poorly specified CV loop invalidates all subsequent feature validation.
