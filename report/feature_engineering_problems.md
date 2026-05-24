# Feature Engineering Notebook — PROBLEMS FOUND ⚠️

Based on detailed comparison of `Feature_Engeneering.ipynb` against `corrected_fe_plan_no_code.md`, the following problems have been identified:

---

## 🔴 CRITICAL PROBLEMS (Will cause errors or major incorrect results)

### 1. **WRONG TARGET VARIABLE BEING USED (Cell 14)**
**Status:** ❌ BROKEN  
**Location:** Cell 14

```python
y = np.log1p(clean_train["SalePrice_log"]).copy()
```

**Problem:**
- You already created `SalePrice_log = np.log1p(clean_train["SalePrice"])` in Cell 11
- Then you're doing `np.log1p()` on `SalePrice_log` AGAIN
- This creates **double log transformation**: `log(log(price))`
- This is mathematically wrong and will destroy your target variable

**Correct approach:**
```python
y = clean_train["SalePrice_log"].copy()  # Use the log-transformed version directly
```

**Impact:** 🔴 CRITICAL — Your entire model is training on the wrong target

---

### 2. **MiscFeature filled twice with different values (Cells 37-38)**
**Status:** ❌ INCONSISTENT  
**Location:** Cells 37 and 38

```python
# Cell 37
full_data[misc_cols] = full_data[misc_cols].fillna("None")

# Cell 38 (overwrites Cell 37!)
full_data["MiscFeature"] = full_data["MiscFeature"].fillna("NoMiscFeature")
```

**Problem:**
- Cell 37 fills with `"None"`
- Cell 38 immediately overwrites it with `"NoMiscFeature"`
- Plan says fill with `"None"`, but Cell 38 changes it to `"NoMiscFeature"`
- This is inconsistent — one or the other should be removed

**Decision:** Follow your plan → keep **"None"** and DELETE Cell 38

---

### 3. **Missing Feature Engineering Sections (NOT IMPLEMENTED)**
**Status:** ❌ NOT DONE  
**Location:** Notebook ends abruptly

Your notebook **stops at Cell 90** but your plan requires:

- ✗ **Section 13-14** (Cells 86-90 incomplete) — Bathroom and Count Features not fully implemented
- ✗ **Section 15** — Presence/Absence Flags (partially done as `HasGarage`, `HasBasement`, `HasPool`)
- ✗ **Section 16** — Temporal and Age Features (HouseAge, RemodAge, GarageAge, IsNewHouse, etc.) — **NOT DONE**
- ✗ **Section 17** — Quality and Score Features (QualityConditionGap, ExteriorScore, GarageScore, etc.) — **NOT DONE**
- ✗ **Section 18** — Interaction and Ratio Features (QualAreaInteraction, QualTotalSF, BathPerBedroom, RoomPerBedroom, etc.) — **NOT DONE**
- ✗ **Section 19** — Rare Category Grouping — **NOT DONE**
- ✗ **Section 20** — One-Hot Encoding — **NOT DONE**
- ✗ **Section 21** — Skewed Feature Transformation — **NOT DONE**
- ✗ **Section 22** — Feature Selection and Redundancy Check — **NOT DONE**
- ✗ **Section 23** — Train-Test Split Back — **NOT DONE**
- ✗ **Section 24** — Cross-Validation Architecture — **NOT DONE**

**Impact:** 🔴 CRITICAL — Notebook is only ~40% complete; missing 60% of feature engineering steps

---

## 🟠 HIGH-PRIORITY PROBLEMS (Will cause incorrect results)

### 4. **BathPerBedroom and RoomPerBedroom division guards NOT implemented (Cells 86-90)**
**Status:** ❌ INCOMPLETE  
**Location:** Cell 90 (comment area ends without code)

**Problem:**
- Plan explicitly states (Section 14): "6 rows in train have `BedroomAbvGr = 0`"
- Division by zero WILL happen
- Plan says: "For zero-bedroom rows, fill with median of ratio computed on non-zero-bedroom rows"
- This is **NOT implemented**

**Missing code:**
```python
# Create ratios but guard against division by zero
safe_bedrooms = clean_train[clean_train["BedroomAbvGr"] > 0]
median_bath_per_bed = safe_bedrooms["TotalBath"].sum() / safe_bedrooms["BedroomAbvGr"].sum()
median_room_per_bed = safe_bedrooms["TotRmsAbvGrd"].sum() / safe_bedrooms["BedroomAbvGr"].sum()

# Create features with guard
full_data["BathPerBedroom"] = full_data.apply(
    lambda row: (row["TotalBath"] / row["BedroomAbvGr"]) 
    if row["BedroomAbvGr"] > 0 
    else median_bath_per_bed,
    axis=1
)

full_data["RoomPerBedroom"] = full_data.apply(
    lambda row: (row["TotRmsAbvGrd"] / row["BedroomAbvGr"]) 
    if row["BedroomAbvGr"] > 0 
    else median_room_per_bed,
    axis=1
)
```

**Impact:** 🟠 HIGH — Creates invalid inf/NaN values; will break downstream modeling

---

### 5. **YearBuilt, YearRemodAdd, GarageYrBlt NOT dropped (Section 22)**
**Status:** ❌ WILL NOT DO  
**Location:** Should be done after age feature creation

**Problem:**
- Plan says (Section 24.3): "After age features are created, the original year columns become redundant"
- Plan specifies to **drop**: `YearBuilt`, `YearRemodAdd`, `GarageYrBlt`
- Plan says: "Keep `YrSold` — it captures market cycle"
- These drops are **missing from notebook**
- This will cause **perfect multicollinearity** (e.g., `YearBuilt` vs `HouseAge` = -1.0 correlation)

**Missing code:**
```python
# After creating age features
full_data.drop(columns=["YearBuilt", "YearRemodAdd", "GarageYrBlt"], inplace=True, errors="ignore")
```

**Impact:** 🟠 HIGH — Linear models will fail; tree models will pick arbitrary year vs age

---

## 🟡 MEDIUM-PRIORITY PROBLEMS (Design issues)

### 6. **Unused cyclic encoding for MoSold (Cell 69)**
**Status:** ⚠️ PARTIALLY IMPLEMENTED  
**Location:** Cell 69-70

```python
full_data["MoSold_sin"] = np.sin(2 * np.pi * full_data["MoSold"] / 12)
full_data["MoSold_cos"] = np.cos(2 * np.pi * full_data["MoSold"] / 12)
```

**Problem:**
- You created cyclic features for `MoSold` but the comment in Cell 66 says "most common in your project" are tree models
- Plan explicitly says (Section 10.2, Option A): **"do nothing"** for tree models
- Cyclic encoding is useful **only for linear models** (Ridge/Lasso)
- If using XGBoost/LightGBM, these features are **unnecessary and wasted**

**Decision needed:**
- If tree models: Remove `MoSold_sin` and `MoSold_cos`, keep `MoSold` as raw integer
- If linear models: Keep cyclic encoding but document the choice

**Impact:** 🟡 MEDIUM — Adds unnecessary features; slight overfitting risk

---

### 7. **No documented decision on model type**
**Status:** ⚠️ UNCLEAR  
**Location:** Throughout (especially Cell 66, Cell 72)

**Problem:**
- Cell 66 says "most common in your project" (tree models)
- But no clear statement of which model you're building
- This affects decisions on:
  - MoSold encoding (cyclic vs raw)
  - Correlation pair handling (drop vs keep)
  - Feature importance (permutation vs split-based)
- Plan explicitly lists different paths for tree vs linear models

**What's needed:**
```python
# At top of notebook after setup
MODEL_TYPE = "tree"  # "tree" or "linear"
# Then use this flag in conditional feature engineering

if MODEL_TYPE == "tree":
    # Keep MoSold raw
    pass
else:  # linear
    # Use cyclic encoding
    pass
```

**Impact:** 🟡 MEDIUM — Causes inconsistent feature engineering decisions

---

## 🟢 MINOR ISSUES (Style/cleanup)

### 8. **Unnecessary cells without content (Cells 83-84)**
**Status:** ⚠️ CLEANUP  
**Location:** Cells 83-84

```python
# Cell 83
full_data.shape

# Cell 84
full_data['Street'].isna().sum()
```

**Problem:**
- These are exploratory checks, not part of production pipeline
- They should either be moved to a separate EDA notebook or removed

**Recommendation:** Delete or move to verification section

---

### 9. **HasGarage defined twice (Cells 31 and 85)**
**Status:** ⚠️ REDUNDANT  
**Location:** Cells 31 and 85

```python
# Cell 31
full_data["HasGarage"] = (full_data["GarageType"] != "NoGarage").astype(int)

# Cell 85 (re-defined with same logic)
full_data["HasGarage"] = (
    full_data["GarageType"] != "NoGarage"
).astype(int)
```

**Problem:**
- Same feature created twice
- Wastes computation and confuses readers
- Cell 31 is the original; Cell 85 is the duplicate

**Recommendation:** Delete one (keep the first in Cell 31)

---

### 10. **HasPool created as binary but encoded as ordinal (Cell 85)**
**Status:** ⚠️ INCONSISTENT LOGIC  
**Location:** Cell 85

```python
full_data["HasPool"] = (
    full_data["PoolQC"] != "NoPool"
).astype(int)
```

**Problem:**
- Earlier in Cell 73, `PoolQC` was ordinal-encoded to integers (0-5)
- So `PoolQC != "NoPool"` will FAIL because `PoolQC` is now a number, not a string
- This is a **type mismatch bug**

**Correct approach:**
```python
# Do NOT use string comparison on ordinal-encoded feature
full_data["HasPool"] = (full_data["PoolQC"] > 0).astype(int)  # Use ordinal value
```

**Impact:** 🔴 This will throw an error when Cell 85 runs

---

### 11. **HasBasement created but never actually used (Cell 85)**
**Status:** ⚠️ PARTIAL  
**Location:** Cell 85

```python
full_data["HasBasement"] = (
    full_data["BsmtQual"] != "NoBasement"
).astype(int)
```

**Problem:**
- Same type mismatch as HasPool above
- `BsmtQual` was ordinal-encoded to integers in Cell 78
- String comparison will fail

**Correct approach:**
```python
full_data["HasBasement"] = (full_data["BsmtQual"] > 0).astype(int)
```

---

## 📋 SUMMARY TABLE

| Problem # | Title | Severity | Type | Location | Status |
|-----------|-------|----------|------|----------|--------|
| 1 | Double log transformation of target | 🔴 CRITICAL | Logic error | Cell 14 | Will break training |
| 2 | MiscFeature filled twice inconsistently | 🔴 CRITICAL | Data error | Cells 37-38 | Duplicate/conflicting fills |
| 3 | Feature engineering 60% incomplete | 🔴 CRITICAL | Missing code | Cells 91+ | All remaining steps missing |
| 4 | BathPerBedroom/RoomPerBedroom division guards missing | 🟠 HIGH | Missing guard | Cell 90 | Will create inf/NaN |
| 5 | Year columns not dropped (redundant with age) | 🟠 HIGH | Missing code | Should be step 22 | Multicollinearity |
| 6 | Cyclic encoding inconsistent with model type | 🟡 MEDIUM | Design issue | Cells 69-70 | Unnecessary features |
| 7 | No documented model type decision | 🟡 MEDIUM | Design issue | Throughout | Inconsistent choices |
| 8 | Exploratory cells in production pipeline | 🟢 MINOR | Cleanup | Cells 83-84 | Not pipeline-ready |
| 9 | HasGarage defined twice (redundant) | 🟢 MINOR | Duplicate code | Cells 31 & 85 | Redundant computation |
| 10 | Type mismatch: HasPool on ordinal column | 🔴 CRITICAL | Bug | Cell 85 | Will error on execution |
| 11 | Type mismatch: HasBasement on ordinal column | 🔴 CRITICAL | Bug | Cell 85 | Will error on execution |

---

## ✅ WHAT'S WORKING WELL

1. **Proper outlier removal** (Cells 3-8) ✓
2. **Correct log transformation** of target (Cell 11, though misused in Cell 14) ✓
3. **Train-test combine structure** (Cells 13-16) ✓
4. **Data error fixes** (Cells 16-25) ✓
5. **Facility-based categorical imputation** (Cells 27-38) ✓ (except MiscFeature duplicate)
6. **Lot frontage neighborhood imputation** (Cells 49-52) ✓
7. **MasVnrType case-based disambiguation** (Cells 54-57) ✓
8. **Near-constant feature removal** (Cell 59) ✓
9. **Type conversions** (Cells 62-70) ✓
10. **Ordinal encoding with explicit maps** (Cells 73-81) ✓

---

## 🎯 IMMEDIATE ACTION ITEMS

### Before running notebook:
1. **Fix Cell 14** — Change from double log to single log
2. **Delete Cell 38** — Remove duplicate MiscFeature fill
3. **Fix Cell 85** — Change string comparisons to ordinal comparisons
4. **Add division guards** — Implement BathPerBedroom and RoomPerBedroom guards
5. **Continue notebook** — Add all remaining sections (15-25)

### After fixing:
1. Test that all cells run without errors
2. Verify target shape and values are correct
3. Run cross-validation to check feature quality
4. Document model type choice (tree vs linear)

---

## 📞 NEXT STEPS

Would you like me to:
1. Create a corrected version of the notebook?
2. Generate just the missing feature engineering code sections?
3. Provide detailed fixes for each problem above?

