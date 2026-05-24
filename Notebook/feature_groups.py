target_feature = "SalePrice"

id_feature = "Id"


nominal_categorical = [
    "MSSubClass",
    "MSZoning",
    "Alley",
    "LandContour",
    "LotConfig",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "Heating",
    "Electrical",
    "GarageType",
    "MiscFeature",
    "SaleType",
    "SaleCondition"
]


ordinal_categorical = [
    "LotShape",
    "Utilities",
    "LandSlope",
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "HeatingQC",
    "KitchenQual",
    "Functional",
    "FireplaceQu",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence"
]


binary_categorical = [
    "Street",
    "CentralAir"
]


categorical_features = (
    nominal_categorical
    + ordinal_categorical
    + binary_categorical
)


continuous_numeric = [
    "LotFrontage",
    "LotArea",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal"
]


count_numeric = [
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageCars"
]


ordinal_numeric = [
    "OverallQual",
    "OverallCond"
]


temporal_features = [
    "YearBuilt",
    "YearRemodAdd",
    "GarageYrBlt",
    "MoSold",
    "YrSold"
]


numeric_features = (
    continuous_numeric
    + count_numeric
    + ordinal_numeric
    + temporal_features
)


input_features = categorical_features + numeric_features


drop_features = [
    id_feature
]


all_feature_groups = {
    "nominal_categorical": nominal_categorical,
    "ordinal_categorical": ordinal_categorical,
    "binary_categorical": binary_categorical,
    "categorical_features": categorical_features,
    "continuous_numeric": continuous_numeric,
    "count_numeric": count_numeric,
    "ordinal_numeric": ordinal_numeric,
    "temporal_features": temporal_features,
    "numeric_features": numeric_features,
    "input_features": input_features,
    "drop_features": drop_features,
}