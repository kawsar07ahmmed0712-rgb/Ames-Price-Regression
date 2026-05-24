# 🏡 Ames House Price Dataset Guide (`train(1).csv`)

This Markdown file explains the uploaded housing dataset in clear language: what each feature means, how to read it, examples from the data, and basic ML cleaning notes.

## 1. Quick Dataset Understanding 🚀

- **Rows:** `1460` houses
- **Columns:** `81` columns
- **Problem type:** Regression
- **Dataset identity:** This matches the common Ames/Kaggle House Prices training dataset based on the column names and shape.
- **Goal:** Predict the house sale price
- **Target column:** `SalePrice`
- **Main idea:** Each row = one sold house. Each column = one house property, quality score, size measure, location code, sale detail, or target price.
- **Feature count idea:** `79` explanatory house features + `Id` + target `SalePrice` = `81` columns in this training file.

### 🎯 Target variable

| Target | Meaning | Example |
|---|---|---|
| `SalePrice` | Final property sale price in US dollars | `208500` = $208,500 |

## 2. Important Symbols & Codes 🔑

- `SF` = square feet. Example: `GrLivArea = 1710` means 1,710 sq ft above-ground living area.
- `Ex/Gd/TA/Fa/Po` = Excellent / Good / Typical-Average / Fair / Poor.
- `NA` is tricky: in many columns it means **the house does not have that feature**, not missing data. Example: `PoolQC = NA` usually means **No Pool**.
- Some numeric-looking columns are actually categories. Example: `MSSubClass = 60` is a building class, not “60 units”.
- Quality scores like `OverallQual` and `OverallCond` use a 1–10 scale where higher usually means better.

## 3. Dataset Shape & Missing/NA Summary 🧹

In your CSV, missing/absence values are mostly written as the string `NA`. Below are the columns with the highest `NA` counts.

| Column | `NA` count | Percent of rows | Meaning of `NA` |
|---|---:|---:|---|
| `PoolQC` | 1453 | 99.5% | No pool |
| `MiscFeature` | 1406 | 96.3% | No miscellaneous feature |
| `Alley` | 1369 | 93.8% | No alley access |
| `Fence` | 1179 | 80.8% | No fence |
| `FireplaceQu` | 690 | 47.3% | No fireplace |
| `LotFrontage` | 259 | 17.7% | Likely missing street frontage |
| `GarageYrBlt` | 81 | 5.5% | No garage / no garage year |
| `GarageType` | 81 | 5.5% | No garage |
| `GarageQual` | 81 | 5.5% | No garage |
| `GarageFinish` | 81 | 5.5% | No garage |
| `GarageCond` | 81 | 5.5% | No garage |
| `BsmtFinType2` | 38 | 2.6% | No basement |
| `BsmtExposure` | 38 | 2.6% | No basement |
| `BsmtQual` | 37 | 2.5% | No basement |
| `BsmtFinType1` | 37 | 2.5% | No basement |
| `BsmtCond` | 37 | 2.5% | No basement |
| `MasVnrType` | 8 | 0.5% | Likely none or missing masonry veneer type |
| `MasVnrArea` | 8 | 0.5% | Likely missing masonry veneer area |
| `Electrical` | 1 | 0.1% | Missing electrical value |

## 4. First Row Example 🧠

Example row #1 in plain English:
- House ID `1` sold for **$208,500**.
- It is an `60` class house: a 2-story newer-style dwelling.
- Zoning is `RL`: low-density residential.
- Lot area is `8,450 sq ft`; above-grade living area is `1,710 sq ft`.
- Overall quality is `7/10`, condition is `5/10`.
- Built in `2003`, with `2`-car garage and `2` full bathrooms above ground.

## 5. Numeric Feature Snapshot 📊

| Feature | Quick stats from uploaded CSV |
|---|---|
| `SalePrice` | min 34900, max 755000, avg 180921.20 |
| `LotArea` | min 1300, max 215245, avg 10516.83 |
| `LotFrontage` | min 21, max 313, avg 70.05 |
| `OverallQual` | min 1, max 10, avg 6.10 |
| `OverallCond` | min 1, max 9, avg 5.58 |
| `YearBuilt` | min 1872, max 2010, avg 1971.27 |
| `YearRemodAdd` | min 1950, max 2010, avg 1984.87 |
| `GrLivArea` | min 334, max 5642, avg 1515.46 |
| `TotalBsmtSF` | min 0, max 6110, avg 1057.43 |
| `GarageArea` | min 0, max 1418, avg 472.98 |
| `GarageCars` | min 0, max 4, avg 1.77 |
| `FullBath` | min 0, max 3, avg 1.57 |

## 6. Strong Numeric Relationships With `SalePrice` 🔍

These are simple Pearson correlations calculated from the uploaded CSV for numeric/number-like columns. Correlation is not causation, but it is useful for first EDA.

| Rank | Feature | Correlation with `SalePrice` | Simple meaning |
|---:|---|---:|---|
| 1 | `OverallQual` | 0.791 | Overall material and finish quality of the house. |
| 2 | `GrLivArea` | 0.709 | Above-grade living area in square feet. |
| 3 | `GarageCars` | 0.640 | Garage size measured by car capacity. |
| 4 | `GarageArea` | 0.623 | Garage area in square feet. |
| 5 | `TotalBsmtSF` | 0.614 | Total basement square feet. |
| 6 | `1stFlrSF` | 0.606 | First-floor square feet. |
| 7 | `FullBath` | 0.561 | Number of full bathrooms above grade. |
| 8 | `TotRmsAbvGrd` | 0.534 | Total rooms above grade, excluding bathrooms. |
| 9 | `YearBuilt` | 0.523 | Original construction year. |
| 10 | `YearRemodAdd` | 0.507 | Remodel/addition year; same as construction year if never remodeled. |
| 11 | `GarageYrBlt` | 0.486 | Year garage was built. |
| 12 | `MasVnrArea` | 0.477 | Masonry veneer area in square feet. |

## 7. Full Feature Dictionary 📚

Use this section as your main reference. Every column from the uploaded CSV is included.

### 🪪 Identifier

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `Id` | Identifier | Unique row number for each house record. | Example: `1` means the first listed house. It is not a real house feature. | Unique: 1460; NA: 0; min 1, max 1460, avg 730.50 | Usually drop before modeling. |

### 🏠 Building type

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `MSSubClass` | Categorical code | Type/class of dwelling involved in the sale. | `60` = 2-story house built 1946 or newer; `20` = 1-story 1946 or newer. | Unique: 15; NA: 0; min 20, max 190, avg 56.90 | Treat as category, not a normal number. |
| `BldgType` | Nominal category | Type of dwelling. | `1Fam` = single-family detached; `Duplex`; `TwnhsE` = townhouse end unit. | Unique: 5; NA: 0; common: `1Fam` (1220), `TwnhsE` (114), `Duplex` (52) | Important structural category. |
| `HouseStyle` | Nominal category | Style/stories of the dwelling. | `1Story`, `2Story`, `1.5Fin`, `SLvl`. | Unique: 8; NA: 0; common: `1Story` (726), `2Story` (445), `1.5Fin` (154) | Related to floor area and age. |

### 📍 Zoning/location

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `MSZoning` | Nominal category | General zoning classification of the property. | `RL` = Residential Low Density; `RM` = Residential Medium Density; `FV` = Floating Village Residential. | Unique: 5; NA: 0; common: `RL` (1151), `RM` (218), `FV` (65) | One-hot encode; zoning can influence price. |

### 📐 Lot & land

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `LotFrontage` | Numeric continuous | Linear feet of street connected to the property. | If `65`, the property touches 65 feet of street. | Unique: 111; NA: 259; min 21, max 313, avg 70.05 | `NA` is likely missing/unknown; impute carefully. |
| `LotArea` | Numeric continuous | Lot size in square feet. | `8450` means the land area is 8,450 sq ft. | Unique: 1073; NA: 0; min 1300, max 215245, avg 10516.83 | Often skewed; log transform may help. |
| `LotShape` | Ordinal-ish category | General shape of the property. | `Reg` = regular; `IR1/IR2/IR3` = increasingly irregular. | Unique: 4; NA: 0; common: `Reg` (925), `IR1` (484), `IR2` (41) | Can be ordinal encoded or one-hot encoded. |
| `LotConfig` | Nominal category | Lot configuration. | `Inside`, `Corner`, `CulDSac`, `FR2`, `FR3`. | Unique: 5; NA: 0; common: `Inside` (1052), `Corner` (263), `CulDSac` (94) | Cul-de-sac/corner lots may price differently. |

### 🛣️ Access

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `Street` | Nominal category | Type of road access to the property. | `Pave` = paved road; `Grvl` = gravel road. | Unique: 2; NA: 0; common: `Pave` (1454), `Grvl` (6) | Very low variety; check if useful. |
| `Alley` | Nominal category | Type of alley access to the property. | `NA` = no alley access; `Grvl` = gravel alley; `Pave` = paved alley. | Unique: 3; NA: 1369; common: `NA` (1369), `Grvl` (50), `Pave` (41) | Here `NA` means “No alley”, not unknown. |
| `PavedDrive` | Ordinal category | Driveway pavement status. | `Y` = paved; `P` = partial pavement; `N` = dirt/gravel. | Unique: 3; NA: 0; common: `Y` (1340), `N` (90), `P` (30) | Ordinal encode. |

### 🌍 Land form

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `LandContour` | Nominal category | Flatness/contour of the property. | `Lvl` = near level; `Bnk` = banked; `HLS` = hillside; `Low` = depression. | Unique: 4; NA: 0; common: `Lvl` (1311), `Bnk` (63), `HLS` (50) | Land shape may affect desirability. |
| `LandSlope` | Ordinal category | Slope of the property. | `Gtl` = gentle, `Mod` = moderate, `Sev` = severe. | Unique: 3; NA: 0; common: `Gtl` (1382), `Mod` (65), `Sev` (13) | Ordinal encoding makes sense. |

### ⚡ Utilities

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `Utilities` | Nominal category | Type of utilities available. | `AllPub` = all public utilities; `NoSeWa` = electricity and gas only. | Unique: 2; NA: 0; common: `AllPub` (1459), `NoSeWa` (1) | Often almost constant; may have low predictive value. |
| `Electrical` | Nominal category | Electrical system type. | `SBrkr` = standard circuit breakers; `FuseA/FuseF/FuseP` = fuse systems. | Unique: 6; NA: 1; common: `SBrkr` (1334), `FuseA` (94), `FuseF` (27) | One `NA` in this file; impute mode or unknown. |

### 📍 Location

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `Neighborhood` | Nominal category | Physical neighborhood within Ames city limits. | `CollgCr`, `NoRidge`, `OldTown`, etc. | Unique: 25; NA: 0; common: `NAmes` (225), `CollgCr` (150), `OldTown` (113) | Usually very important; one-hot or target encoding. |

### 📍 Location condition

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `Condition1` | Nominal category | Proximity to main nearby condition. | `Norm` = normal; `Feedr` = near feeder street; `PosN` = near park/greenbelt. | Unique: 9; NA: 0; common: `Norm` (1260), `Feedr` (81), `Artery` (48) | Captures nearby positives/negatives. |
| `Condition2` | Nominal category | Second nearby condition if more than one exists. | Often `Norm`; used only when another condition exists. | Unique: 8; NA: 0; common: `Norm` (1445), `Feedr` (6), `Artery` (2) | May be sparse; combine carefully. |

### ⭐ Overall quality

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `OverallQual` | Ordinal numeric | Overall material and finish quality of the house. | Scale 1–10: `10` = very excellent, `5` = average, `1` = very poor. | Unique: 10; NA: 0; min 1, max 10, avg 6.10 | Usually one of the strongest predictors. |
| `OverallCond` | Ordinal numeric | Overall condition of the house. | Scale 1–10: high means better current condition. | Unique: 9; NA: 0; min 1, max 9, avg 5.58 | Useful but different from quality: old house can be well-maintained. |

### 📅 Age/time

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `YearBuilt` | Numeric year | Original construction year. | `2003` means the house was built in 2003. | Unique: 112; NA: 0; min 1872, max 2010, avg 1971.27 | Create `HouseAge = YrSold - YearBuilt`. |
| `YearRemodAdd` | Numeric year | Remodel/addition year; same as construction year if never remodeled. | `2003` means remodeled or built in 2003. | Unique: 61; NA: 0; min 1950, max 2010, avg 1984.87 | Create `YearsSinceRemodel = YrSold - YearRemodAdd`. |

### 🏗️ Exterior

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `RoofStyle` | Nominal category | Type/style of roof. | `Gable`, `Hip`, `Flat`, `Gambrel`, etc. | Unique: 6; NA: 0; common: `Gable` (1141), `Hip` (286), `Flat` (13) | One-hot encode. |
| `RoofMatl` | Nominal category | Roof material. | `CompShg` = composite shingle; `WdShngl` = wood shingles. | Unique: 8; NA: 0; common: `CompShg` (1434), `Tar&Grv` (11), `WdShngl` (6) | Rare categories may need grouping. |
| `Exterior1st` | Nominal category | Primary exterior covering/material. | `VinylSd`, `MetalSd`, `Wd Sdng`, etc. | Unique: 15; NA: 0; common: `VinylSd` (515), `HdBoard` (222), `MetalSd` (220) | Material quality/style can affect price. |
| `Exterior2nd` | Nominal category | Secondary exterior covering if more than one material. | Example: primary `VinylSd`, secondary `VinylSd` means mostly one exterior material. | Unique: 16; NA: 0; common: `VinylSd` (504), `MetalSd` (214), `HdBoard` (207) | Often paired with Exterior1st. |
| `MasVnrType` | Nominal category | Masonry veneer type. | `BrkFace` = brick face; `Stone`; `None`; `NA` appears in this file. | Unique: 5; NA: 8; common: `None` (864), `BrkFace` (445), `Stone` (128) | Treat `NA` as missing/none depending inspection. |
| `MasVnrArea` | Numeric continuous | Masonry veneer area in square feet. | `196` means 196 sq ft of masonry veneer. | Unique: 328; NA: 8; min 0, max 1600, avg 103.69 | `NA` is missing; many zeros/none expected. |

### ⭐ Quality ratings

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `ExterQual` | Ordinal category | Quality of exterior material. | `Ex`, `Gd`, `TA`, `Fa`, `Po` from excellent to poor. | Unique: 4; NA: 0; common: `TA` (906), `Gd` (488), `Ex` (52) | Ordinal encode: Ex > Gd > TA > Fa > Po. |
| `ExterCond` | Ordinal category | Present condition of exterior material. | `TA` = typical/average; `Gd` = good. | Unique: 5; NA: 0; common: `TA` (1282), `Gd` (146), `Fa` (28) | Ordinal encode. |

### 🏗️ Structure

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `Foundation` | Nominal category | Type of foundation. | `PConc` = poured concrete; `CBlock` = cinder block; `BrkTil` = brick/tile. | Unique: 6; NA: 0; common: `PConc` (647), `CBlock` (634), `BrkTil` (146) | May reflect house age/quality. |

### 🧱 Basement

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `BsmtQual` | Ordinal category | Basement height quality. | `Ex` = 100+ inches, `Gd` = 90–99, `TA` = 80–89, `NA` = no basement. | Unique: 5; NA: 37; common: `TA` (649), `Gd` (618), `Ex` (121) | `NA` means no basement. |
| `BsmtCond` | Ordinal category | General condition of basement. | `TA` = typical; `Fa` = damp/cracks; `NA` = no basement. | Unique: 5; NA: 37; common: `TA` (1311), `Gd` (65), `Fa` (45) | `NA` means no basement. |
| `BsmtExposure` | Ordinal category | Walkout/garden-level basement exposure. | `Gd`, `Av`, `Mn`, `No`, `NA`. | Unique: 5; NA: 38; common: `No` (953), `Av` (221), `Gd` (134) | `NA` means no basement. |
| `BsmtFinType1` | Ordinal category | Quality/type of main finished basement area. | `GLQ` = good living quarters; `Unf` = unfinished; `NA` = no basement. | Unique: 7; NA: 37; common: `Unf` (430), `GLQ` (418), `ALQ` (220) | Ordinal-ish; use sensible ordering. |
| `BsmtFinSF1` | Numeric continuous | Type 1 finished basement square feet. | `706` means 706 sq ft of the main finished basement type. | Unique: 637; NA: 0; min 0, max 5644, avg 443.64 | Can combine with other basement areas. |
| `BsmtFinType2` | Ordinal category | Quality/type of second finished basement area. | Mostly `Unf`; `NA` = no basement. | Unique: 7; NA: 38; common: `Unf` (1256), `Rec` (54), `LwQ` (46) | Often sparse; still useful with SF2. |
| `BsmtFinSF2` | Numeric continuous | Type 2 finished basement square feet. | `0` means no second finished basement area. | Unique: 144; NA: 0; min 0, max 1474, avg 46.55 | Usually many zeros. |
| `BsmtUnfSF` | Numeric continuous | Unfinished basement square feet. | `150` means 150 sq ft unfinished basement. | Unique: 780; NA: 0; min 0, max 2336, avg 567.24 | Adds to total basement size. |
| `TotalBsmtSF` | Numeric continuous | Total basement square feet. | Formula idea: finished SF1 + finished SF2 + unfinished SF. | Unique: 721; NA: 0; min 0, max 6110, avg 1057.43 | Strong size feature. |

### 🔥 Heating/air

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `Heating` | Nominal category | Type of heating system. | `GasA` = gas forced warm air; `GasW` = gas hot water/steam. | Unique: 6; NA: 0; common: `GasA` (1428), `GasW` (18), `Grav` (7) | Mostly one value; check usefulness. |
| `HeatingQC` | Ordinal category | Heating quality and condition. | `Ex`, `Gd`, `TA`, `Fa`, `Po`. | Unique: 5; NA: 0; common: `Ex` (741), `TA` (428), `Gd` (241) | Ordinal encode. |

### ❄️ Heating/air

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `CentralAir` | Binary category | Whether the house has central air conditioning. | `Y` = yes, `N` = no. | Unique: 2; NA: 0; common: `Y` (1365), `N` (95) | Binary encode 1/0. |

### 📏 Area

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `1stFlrSF` | Numeric continuous | First-floor square feet. | `856` means first floor is 856 sq ft. | Unique: 753; NA: 0; min 334, max 4692, avg 1162.63 | Strong size feature. |
| `2ndFlrSF` | Numeric continuous | Second-floor square feet. | `854` means second floor is 854 sq ft; `0` means no second floor. | Unique: 417; NA: 0; min 0, max 2065, avg 346.99 | Works with HouseStyle. |
| `LowQualFinSF` | Numeric continuous | Low-quality finished square feet across all floors. | `0` usually means none. | Unique: 24; NA: 0; min 0, max 572, avg 5.84 | Often many zeros; may be weak. |
| `GrLivArea` | Numeric continuous | Above-grade living area in square feet. | `1710` means 1,710 sq ft living area above ground. | Unique: 861; NA: 0; min 334, max 5642, avg 1515.46 | Usually very strong; watch outliers. |

### 🚿 Bathrooms

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `BsmtFullBath` | Discrete numeric | Number of full bathrooms in basement. | `1` means one full basement bathroom. | Unique: 4; NA: 0; min 0, max 3, avg 0.43 | Can combine total bathrooms. |
| `BsmtHalfBath` | Discrete numeric | Number of half bathrooms in basement. | `0`, `1`, etc. | Unique: 3; NA: 0; min 0, max 2, avg 0.06 | Can combine total bathrooms. |
| `FullBath` | Discrete numeric | Number of full bathrooms above grade. | `2` means two full bathrooms above ground. | Unique: 4; NA: 0; min 0, max 3, avg 1.57 | Useful comfort feature. |
| `HalfBath` | Discrete numeric | Number of half bathrooms above grade. | `1` means one half bathroom above ground. | Unique: 3; NA: 0; min 0, max 2, avg 0.38 | Often convert total bath score. |

### 🛏️ Rooms

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `BedroomAbvGr` | Discrete numeric | Bedrooms above grade; basement bedrooms are not included. | `3` means three above-ground bedrooms. | Unique: 8; NA: 0; min 0, max 8, avg 2.87 | More is not always better after controlling for area. |
| `TotRmsAbvGrd` | Discrete numeric | Total rooms above grade, excluding bathrooms. | `8` means eight rooms above ground excluding bathrooms. | Unique: 12; NA: 0; min 2, max 14, avg 6.52 | Correlated with living area. |

### 🍳 Rooms

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `KitchenAbvGr` | Discrete numeric | Kitchens above grade. | `1` is normal; `2` may indicate multi-family layout. | Unique: 4; NA: 0; min 0, max 3, avg 1.05 | Often small range. |
| `KitchenQual` | Ordinal category | Kitchen quality. | `Ex`, `Gd`, `TA`, `Fa`, `Po`. | Unique: 4; NA: 0; common: `TA` (735), `Gd` (586), `Ex` (100) | Usually important; ordinal encode. |

### 🛠️ Functionality

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `Functional` | Ordinal category | Home functionality; typical unless deductions are present. | `Typ` = typical; `Min1/Min2` minor deductions; `Maj1/Maj2` major deductions; `Sev`, `Sal`. | Unique: 7; NA: 0; common: `Typ` (1360), `Min2` (34), `Min1` (31) | Ordinal encode or group rare damaged categories. |

### 🔥 Fireplace

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `Fireplaces` | Discrete numeric | Number of fireplaces. | `0`, `1`, `2`, etc. | Unique: 4; NA: 0; min 0, max 3, avg 0.61 | Works with FireplaceQu. |
| `FireplaceQu` | Ordinal category | Fireplace quality. | `Gd`, `TA`, etc.; `NA` = no fireplace. | Unique: 6; NA: 690; common: `NA` (690), `Gd` (380), `TA` (313) | `NA` means no fireplace. |

### 🚗 Garage

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `GarageType` | Nominal category | Garage location/type. | `Attchd` = attached; `Detchd` = detached; `BuiltIn`; `NA` = no garage. | Unique: 7; NA: 81; common: `Attchd` (870), `Detchd` (387), `BuiltIn` (88) | `NA` means no garage. |
| `GarageYrBlt` | Numeric year | Year garage was built. | `2003` means garage built in 2003. | Unique: 98; NA: 81; min 1900, max 2010, avg 1978.51 | `NA` usually means no garage; create garage age or impute 0. |
| `GarageFinish` | Ordinal category | Interior finish of garage. | `Fin` = finished; `RFn` = rough finished; `Unf`; `NA` = no garage. | Unique: 4; NA: 81; common: `Unf` (605), `RFn` (422), `Fin` (352) | Ordinal encode. |
| `GarageCars` | Discrete numeric | Garage size measured by car capacity. | `2` means fits about two cars. | Unique: 5; NA: 0; min 0, max 4, avg 1.77 | Very important size/convenience feature. |
| `GarageArea` | Numeric continuous | Garage area in square feet. | `548` means 548 sq ft garage. | Unique: 441; NA: 0; min 0, max 1418, avg 472.98 | Highly related to GarageCars. |
| `GarageQual` | Ordinal category | Garage quality. | `TA`, `Gd`, etc.; `NA` = no garage. | Unique: 6; NA: 81; common: `TA` (1311), `NA` (81), `Fa` (48) | `NA` means no garage. |
| `GarageCond` | Ordinal category | Garage condition. | `TA`, `Fa`, etc.; `NA` = no garage. | Unique: 6; NA: 81; common: `TA` (1326), `NA` (81), `Fa` (35) | `NA` means no garage. |

### 🌳 Outdoor

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `WoodDeckSF` | Numeric continuous | Wood deck area in square feet. | `0` means no wood deck; `200` means 200 sq ft deck. | Unique: 274; NA: 0; min 0, max 857, avg 94.24 | Many zeros; useful outdoor amenity. |
| `OpenPorchSF` | Numeric continuous | Open porch area in square feet. | `61` means 61 sq ft open porch. | Unique: 202; NA: 0; min 0, max 547, avg 46.66 | Many zeros. |
| `EnclosedPorch` | Numeric continuous | Enclosed porch area in square feet. | `0` means none. | Unique: 120; NA: 0; min 0, max 552, avg 21.95 | Older houses may have more. |
| `3SsnPorch` | Numeric continuous | Three-season porch area in square feet. | `0` means none. | Unique: 20; NA: 0; min 0, max 508, avg 3.41 | Very sparse feature. |
| `ScreenPorch` | Numeric continuous | Screen porch area in square feet. | `0` means none. | Unique: 76; NA: 0; min 0, max 480, avg 15.06 | Sparse but can add value. |
| `Fence` | Ordinal category | Fence quality. | `GdPrv`, `MnPrv`, `GdWo`, `MnWw`; `NA` = no fence. | Unique: 5; NA: 1179; common: `NA` (1179), `MnPrv` (157), `GdPrv` (59) | `NA` means no fence. |

### 🏊 Outdoor luxury

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `PoolArea` | Numeric continuous | Pool area in square feet. | `0` means no pool. | Unique: 8; NA: 0; min 0, max 738, avg 2.76 | Very rare; handle carefully. |
| `PoolQC` | Ordinal category | Pool quality. | `Ex`, `Gd`, `TA`, `Fa`; `NA` = no pool. | Unique: 4; NA: 1453; common: `NA` (1453), `Gd` (3), `Ex` (2) | `NA` means no pool and dominates this column. |

### 🎁 Extra features

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `MiscFeature` | Nominal category | Miscellaneous feature not covered elsewhere. | `Shed`, `Gar2`, `Othr`, `TenC`; `NA` = none. | Unique: 5; NA: 1406; common: `NA` (1406), `Shed` (49), `Gar2` (2) | `NA` means no miscellaneous feature. |
| `MiscVal` | Numeric monetary | Dollar value of miscellaneous feature. | `0` means no extra misc value. | Unique: 21; NA: 0; min 0, max 15500, avg 43.49 | Many zeros; use with MiscFeature. |

### 🧾 Sale info

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `MoSold` | Discrete numeric/month | Month when the property was sold. | `2` = February; `7` = July. | Unique: 12; NA: 0; min 1, max 12, avg 6.32 | Can capture seasonality; treat as category or cyclic. |
| `YrSold` | Numeric year | Year when the property was sold. | `2008` means sold in 2008. | Unique: 5; NA: 0; min 2006, max 2010, avg 2007.82 | Use for age features; not future-known in some setups. |
| `SaleType` | Nominal category | Type of sale/deed/financing. | `WD` = warranty deed conventional; `New`; `COD`; contract types. | Unique: 9; NA: 0; common: `WD` (1267), `New` (122), `COD` (43) | One-hot encode; may capture unusual transactions. |
| `SaleCondition` | Nominal category | Condition of sale. | `Normal`, `Abnorml`, `Partial`, `Family`, etc. | Unique: 6; NA: 0; common: `Normal` (1198), `Partial` (125), `Abnorml` (101) | Important for unusual or new-home sales. |

### 🎯 Target

| Feature | Type | Meaning | Example / How to read | Uploaded data notes | ML / cleaning tip |
|---|---|---|---|---|---|
| `SalePrice` | Numeric continuous | Final sale price of the property in dollars. | `208500` means $208,500. | Unique: 663; NA: 0; min 34900, max 755000, avg 180921.20 | This is the value to predict. Do not use as input feature. |

## 8. How to Think About Feature Groups 🧩

### 📍 Location features
`Neighborhood`, `MSZoning`, `Condition1`, and `Condition2` describe where the house is and what is near it. In real estate, location can strongly influence price.

### 📏 Size features
`GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`, `GarageArea`, and `LotArea` describe how big the property or house is. Bigger often means more expensive, but outliers can exist.

### ⭐ Quality features
`OverallQual`, `ExterQual`, `KitchenQual`, `BsmtQual`, `GarageQual`, and similar columns describe quality. These features often matter a lot because buyers pay for condition and finish.

### 📅 Time/age features
`YearBuilt`, `YearRemodAdd`, `GarageYrBlt`, and `YrSold` can be converted into age features. Example: `HouseAge = YrSold - YearBuilt`.

### 🧱 Basement and garage features
Basement and garage columns work as families. Example: `GarageType`, `GarageCars`, `GarageArea`, `GarageFinish`, `GarageQual`, and `GarageCond` together describe the garage.

## 9. Beginner-Friendly Cleaning Plan 🧼

1. Drop `Id` for model training because it is only an identifier.
2. Keep `SalePrice` as target variable `y`.
3. Convert `MSSubClass` to a categorical feature even though it looks numeric.
4. For columns where `NA` means no feature, replace `NA` with `None`: `Alley`, `PoolQC`, `Fence`, `FireplaceQu`, garage features, basement features, `MiscFeature`.
5. For true missing numeric values like `LotFrontage` and `MasVnrArea`, impute using median or neighborhood-based median.
6. Encode categorical columns using one-hot encoding for simple models.
7. Ordinal quality columns can be mapped: `Ex=5`, `Gd=4`, `TA=3`, `Fa=2`, `Po=1`, `NA/None=0` when absence is meaningful.
8. Create helpful engineered features:
   - `HouseAge = YrSold - YearBuilt`
   - `RemodAge = YrSold - YearRemodAdd`
   - `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
   - `TotalBathrooms = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`
   - `HasGarage = GarageArea > 0`
   - `HasPool = PoolArea > 0`
9. Check outliers, especially very large `GrLivArea`, `LotArea`, and `SalePrice`.
10. Consider `log1p(SalePrice)` for regression because house prices are usually right-skewed.

## 10. Simple Mental Model 🧠

Think of the price like this:

> **SalePrice ≈ Location + Size + Quality + Age + Rooms + Basement + Garage + Outdoor features + Sale condition**

Example: a newer house in a strong neighborhood with high `OverallQual`, large `GrLivArea`, finished basement, and 2-car garage will usually be priced higher than an older, smaller house with lower quality and no garage.

## 11. Recommended First EDA Questions ✅

- Which neighborhoods have the highest median `SalePrice`?
- How does `OverallQual` change the average price?
- Is `GrLivArea` linearly related to price, or are there outliers?
- Do houses with garages sell for more?
- Does `YearBuilt` or `YearRemodAdd` matter more?
- Are `SaleCondition = Partial` houses mostly new homes?

## 12. References / Source Notes 🔗

- Feature meanings follow the standard Ames/Kaggle House Prices data dictionary for matching column names.
- Kaggle competition page: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
- Data dictionary mirror used for column meanings: https://github.com/rehassachdeva/House-Prices-Advanced-Regression-Techniques---Kaggle-Competition/blob/master/data_description.txt
- The uploaded CSV was directly profiled for row count, column count, examples, `NA` counts, unique counts, numeric stats, and correlations.
