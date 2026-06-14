from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.feature_selection import VarianceThreshold

from House_Price.utils.exception import CustomException
from House_Price.utils.logger import logger


class AmesFeaturePipeline:
    """
    Feature engineering pipeline for Ames House Price Regression.

    This class preserves the feature engineering logic from Feature_Engineering.ipynb.
    Main purpose:
    - fit_transform raw train/test into final model-ready features
    - store transformation state for later prediction-time consistency
    """

    def __init__(self) -> None:
        self.id_column = "Id"
        self.target_column = "SalePrice"
        self.target_log_column = "SalePrice_log"

        self.n_train_: int | None = None
        self.feature_names_: List[str] = []
        self.removed_outlier_ids_: List[int] = []
        self.removed_outlier_indices_: List[int] = []

        self.mode_values_: Dict[str, Any] = {}
        self.fixed_default_values_: Dict[str, Any] = {"Functional": "Typ"}

        self.lot_median_by_neigh_: Dict[str, float] = {}
        self.lot_global_median_: float | None = None
        self.mode_veneer_: Any = None

        self.rare_mapping_: Dict[str, Dict[Any, str]] = {}
        self.high_skew_columns_: List[str] = []
        self.constant_columns_: List[str] = []
        self.dead_dummy_columns_: List[str] = []

        self.metadata_: Dict[str, Any] = {}

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Fit feature engineering state using train data and transform train/test.

        Returns:
            X_train, X_test, y_log, metadata
        """
        try:
            logger.info("Ames feature pipeline fit_transform started.")

            train = train_df.copy()
            test = test_df.copy()

            original_train_shape = train.shape
            original_test_shape = test.shape

            train_ids = train[self.id_column].copy() if self.id_column in train.columns else None
            test_ids = test[self.id_column].copy() if self.id_column in test.columns else None

            clean_train = self._remove_outliers(train)

            y_original = clean_train[self.target_column].copy()
            y_log = np.log1p(y_original)
            y_log.name = self.target_log_column

            train_features = clean_train.drop(
                columns=[self.target_column],
                errors="ignore",
            )

            self.n_train_ = train_features.shape[0]

            full_data = pd.concat(
                [train_features, test],
                axis=0,
                ignore_index=True,
            )

            full_data = self._fix_known_data_errors(full_data)
            full_data = self._handle_meaningful_missing_values(full_data)
            full_data = self._handle_test_missing_values(full_data, clean_train)
            full_data = self._impute_lot_frontage(full_data, clean_train)
            full_data = self._handle_masonry_veneer(full_data, clean_train)

            full_data = full_data.drop(columns=["Utilities", self.id_column], errors="ignore")

            full_data = self._type_conversions(full_data)
            full_data = self._apply_ordinal_encoding(full_data)
            full_data = self._apply_binary_encoding_and_basic_flags(full_data)

            full_data = self._create_core_area_features(full_data)
            full_data = self._create_bathroom_features(full_data)
            full_data = self._create_presence_flags(full_data)
            full_data = self._create_temporal_features(full_data)
            full_data = self._create_quality_score_features(full_data)
            full_data = self._create_interaction_features(full_data)
            full_data = self._create_ratio_features(full_data)

            full_data = self._group_rare_categories(full_data)
            full_data = self._one_hot_encode(full_data)
            full_data = self._apply_skew_transform(full_data)
            full_data = self._drop_redundant_and_low_variance_features(full_data)

            X_train, X_test = self._split_back(full_data)
            X_train, X_test = self._remove_dead_dummy_columns(X_train, X_test)

            self.feature_names_ = list(X_train.columns)

            self.metadata_ = self._build_metadata(
                original_train_shape=original_train_shape,
                original_test_shape=original_test_shape,
                X_train=X_train,
                X_test=X_test,
                y_log=y_log,
                train_ids=train_ids,
                test_ids=test_ids,
            )

            logger.info("Ames feature pipeline fit_transform completed successfully.")
            return X_train, X_test, y_log, self.metadata_

        except Exception as e:
            logger.error("Ames feature pipeline fit_transform failed.")
            raise CustomException(e, sys)

    def _remove_outliers(self, train: pd.DataFrame) -> pd.DataFrame:
        clean_train = train.copy()

        canonical_outlier_mask = (
            (clean_train["GrLivArea"] > 4000)
            & (clean_train["SalePrice"] < 200000)
        )

        if self.id_column in clean_train.columns:
            self.removed_outlier_ids_ = clean_train.loc[
                canonical_outlier_mask, self.id_column
            ].tolist()

        self.removed_outlier_indices_ = clean_train.index[canonical_outlier_mask].tolist()

        clean_train = clean_train.loc[~canonical_outlier_mask].reset_index(drop=True)

        logger.info(f"Removed outlier IDs: {self.removed_outlier_ids_}")
        logger.info(f"Train shape after outlier removal: {clean_train.shape}")

        return clean_train

    def _fix_known_data_errors(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        garage_typo_mask = df["GarageYrBlt"] == 2207
        if garage_typo_mask.any():
            df.loc[garage_typo_mask, "GarageYrBlt"] = df.loc[
                garage_typo_mask, "YearBuilt"
            ]

        remodel_after_sold_mask = df["YearRemodAdd"] > df["YrSold"]
        if remodel_after_sold_mask.any():
            df.loc[remodel_after_sold_mask, "YearRemodAdd"] = df.loc[
                remodel_after_sold_mask, "YrSold"
            ]

        return df

    def _handle_meaningful_missing_values(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        garage_cols = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
        bsmt_cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]

        fireplace_cols = ["FireplaceQu"]
        pool_cols = ["PoolQC"]
        fence_cols = ["Fence"]
        alley_cols = ["Alley"]
        misc_cols = ["MiscFeature"]

        df[garage_cols] = df[garage_cols].fillna("NoGarage")
        df["HasGarage"] = (df["GarageType"] != "NoGarage").astype(int)

        df[bsmt_cols] = df[bsmt_cols].fillna("NoBasement")

        df[fireplace_cols] = df[fireplace_cols].fillna("NoFireplace")
        df[pool_cols] = df[pool_cols].fillna("NoPool")
        df[fence_cols] = df[fence_cols].fillna("NoFence")
        df[alley_cols] = df[alley_cols].fillna("NoAlley")
        df[misc_cols] = df[misc_cols].fillna("None")

        return df

    def _handle_test_missing_values(
        self,
        full_data: pd.DataFrame,
        clean_train: pd.DataFrame,
    ) -> pd.DataFrame:
        df = full_data.copy()

        bsmt_num_cols = [
            "BsmtFullBath",
            "BsmtHalfBath",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
        ]

        garage_num_cols = ["GarageCars", "GarageArea"]

        df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)
        df[garage_num_cols] = df[garage_num_cols].fillna(0)

        mode_cols = [
            "MSZoning",
            "Exterior1st",
            "Exterior2nd",
            "KitchenQual",
            "SaleType",
            "Electrical",
        ]

        for col in mode_cols:
            if col in df.columns:
                mode_value = clean_train[col].mode(dropna=True)[0]
                self.mode_values_[col] = mode_value
                df[col] = df[col].fillna(mode_value)

        for col, val in self.fixed_default_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        return df

    def _impute_lot_frontage(
        self,
        full_data: pd.DataFrame,
        clean_train: pd.DataFrame,
    ) -> pd.DataFrame:
        df = full_data.copy()

        lot_median_by_neigh = clean_train.groupby("Neighborhood")["LotFrontage"].median()
        lot_global_median = clean_train["LotFrontage"].median()

        self.lot_median_by_neigh_ = lot_median_by_neigh.to_dict()
        self.lot_global_median_ = float(lot_global_median)

        def fill_lot_frontage(row: pd.Series) -> float:
            if pd.isnull(row["LotFrontage"]):
                return self.lot_median_by_neigh_.get(
                    row["Neighborhood"],
                    self.lot_global_median_,
                )
            return row["LotFrontage"]

        df["LotFrontage"] = df.apply(fill_lot_frontage, axis=1)
        df["LotFrontage"] = df["LotFrontage"].fillna(self.lot_global_median_)

        return df

    def _handle_masonry_veneer(
        self,
        full_data: pd.DataFrame,
        clean_train: pd.DataFrame,
    ) -> pd.DataFrame:
        df = full_data.copy()

        mask_a = df["MasVnrType"].isna() & (df["MasVnrArea"].fillna(0) == 0)
        df.loc[mask_a, "MasVnrType"] = "None"
        df.loc[mask_a, "MasVnrArea"] = 0

        mask_b = df["MasVnrType"].isna() & (df["MasVnrArea"] > 0)
        self.mode_veneer_ = clean_train["MasVnrType"].mode(dropna=True)[0]
        df.loc[mask_b, "MasVnrType"] = self.mode_veneer_

        df["MasVnrType"] = df["MasVnrType"].fillna("None")
        df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

        return df

    def _type_conversions(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        df["MSSubClass"] = df["MSSubClass"].astype(str)

        df["MoSold_sin"] = np.sin(2 * np.pi * df["MoSold"] / 12)
        df["MoSold_cos"] = np.cos(2 * np.pi * df["MoSold"] / 12)

        return df

    def _apply_ordinal_encoding(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        quality_map = {
            "NoPool": 0,
            "NoGarage": 0,
            "NoBasement": 0,
            "NoFireplace": 0,
            "Po": 1,
            "Fa": 2,
            "TA": 3,
            "Gd": 4,
            "Ex": 5,
        }

        quality_cols = [
            "ExterQual",
            "ExterCond",
            "BsmtQual",
            "BsmtCond",
            "HeatingQC",
            "KitchenQual",
            "FireplaceQu",
            "GarageQual",
            "GarageCond",
            "PoolQC",
        ]

        for col in quality_cols:
            df[col] = df[col].map(quality_map)

        quality_missing = df[quality_cols].isna().sum()
        quality_missing = quality_missing[quality_missing > 0]

        if len(quality_missing) > 0:
            absent_ok_cols = [
                "BsmtQual",
                "BsmtCond",
                "FireplaceQu",
                "GarageQual",
                "GarageCond",
                "PoolQC",
            ]

            for col in quality_missing.index:
                if col in absent_ok_cols:
                    df[col] = df[col].fillna(0)
                else:
                    train_median = df.iloc[: self.n_train_][col].median()
                    df[col] = df[col].fillna(train_median)

        bsmt_exposure_map = {
            "NoBasement": 0,
            "No": 1,
            "Mn": 2,
            "Av": 3,
            "Gd": 4,
        }

        bsmt_fin_map = {
            "NoBasement": 0,
            "Unf": 1,
            "LwQ": 2,
            "Rec": 3,
            "BLQ": 4,
            "ALQ": 5,
            "GLQ": 6,
        }

        garage_finish_map = {
            "NoGarage": 0,
            "Unf": 1,
            "RFn": 2,
            "Fin": 3,
        }

        paved_drive_map = {
            "N": 0,
            "P": 1,
            "Y": 2,
        }

        land_slope_map = {
            "Sev": 0,
            "Mod": 1,
            "Gtl": 2,
        }

        functional_map = {
            "Sal": 0,
            "Sev": 1,
            "Maj2": 2,
            "Maj1": 3,
            "Mod": 4,
            "Min2": 5,
            "Min1": 6,
            "Typ": 7,
        }

        df["BsmtExposure"] = df["BsmtExposure"].map(bsmt_exposure_map)
        df["BsmtFinType1"] = df["BsmtFinType1"].map(bsmt_fin_map)
        df["BsmtFinType2"] = df["BsmtFinType2"].map(bsmt_fin_map)
        df["GarageFinish"] = df["GarageFinish"].map(garage_finish_map)
        df["PavedDrive"] = df["PavedDrive"].map(paved_drive_map)
        df["LandSlope"] = df["LandSlope"].map(land_slope_map)
        df["Functional"] = df["Functional"].map(functional_map)

        ordinal_cols = [
            "BsmtExposure",
            "BsmtFinType1",
            "BsmtFinType2",
            "GarageFinish",
            "PavedDrive",
            "LandSlope",
            "Functional",
        ]

        ordinal_missing = df[ordinal_cols].isna().sum()
        ordinal_missing = ordinal_missing[ordinal_missing > 0]

        if len(ordinal_missing) > 0:
            zero_absent_cols = [
                "BsmtExposure",
                "BsmtFinType1",
                "BsmtFinType2",
                "GarageFinish",
            ]

            for col in ordinal_missing.index:
                if col in zero_absent_cols:
                    df[col] = df[col].fillna(0)
                else:
                    train_median = df.iloc[: self.n_train_][col].median()
                    df[col] = df[col].fillna(train_median)

        return df

    def _apply_binary_encoding_and_basic_flags(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        df["CentralAir"] = df["CentralAir"].astype(str).str.strip()
        df["Street"] = df["Street"].astype(str).str.strip()

        df["CentralAir"] = df["CentralAir"].map({"Y": 1, "N": 0})
        df["Street"] = df["Street"].map({"Pave": 1, "Grvl": 0})

        df["HasGarage"] = (df["GarageType"] != "NoGarage").astype(int)

        # These two are overwritten later after ordinal encoding logic.
        df["HasBasement"] = (df["BsmtQual"] != "NoBasement").astype(int)
        df["HasPool"] = (df["PoolQC"] != "NoPool").astype(int)

        return df

    def _create_core_area_features(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

        df["TotalFinishedSF"] = (
            df["1stFlrSF"]
            + df["2ndFlrSF"]
            + df["BsmtFinSF1"]
            + df["BsmtFinSF2"]
        )

        df["TotalPorchSF"] = (
            df["OpenPorchSF"]
            + df["EnclosedPorch"]
            + df["3SsnPorch"]
            + df["ScreenPorch"]
        )

        df["TotalOutdoorSF"] = (
            df["WoodDeckSF"]
            + df["OpenPorchSF"]
            + df["EnclosedPorch"]
            + df["3SsnPorch"]
            + df["ScreenPorch"]
            + df["PoolArea"]
        )

        if "TotalLivingSF" in df.columns:
            df.drop(columns=["TotalLivingSF"], inplace=True)

        return df

    def _create_bathroom_features(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        df["TotalBath"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"]
            + 0.5 * df["BsmtHalfBath"]
        )

        df["TotalFullBath"] = df["FullBath"] + df["BsmtFullBath"]
        df["TotalHalfBath"] = df["HalfBath"] + df["BsmtHalfBath"]

        train_ratio_source = df.iloc[: self.n_train_].copy()
        train_with_beds = train_ratio_source[train_ratio_source["BedroomAbvGr"] > 0].copy()

        train_with_beds["TotalBath_temp"] = (
            train_with_beds["FullBath"]
            + 0.5 * train_with_beds["HalfBath"]
            + train_with_beds["BsmtFullBath"]
            + 0.5 * train_with_beds["BsmtHalfBath"]
        )

        median_bath_per_bed = (
            train_with_beds["TotalBath_temp"] / train_with_beds["BedroomAbvGr"]
        ).median()

        median_room_per_bed = (
            train_with_beds["TotRmsAbvGrd"] / train_with_beds["BedroomAbvGr"]
        ).median()

        total_bath_series = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"]
            + 0.5 * df["BsmtHalfBath"]
        )

        df["BathPerBedroom"] = np.where(
            df["BedroomAbvGr"] > 0,
            total_bath_series / df["BedroomAbvGr"],
            median_bath_per_bed,
        )

        df["RoomPerBedroom"] = np.where(
            df["BedroomAbvGr"] > 0,
            df["TotRmsAbvGrd"] / df["BedroomAbvGr"],
            median_room_per_bed,
        )

        return df

    def _create_presence_flags(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        df["HasBasement"] = (df["BsmtQual"] > 0).astype(int)
        df["HasFireplace"] = (df["FireplaceQu"] > 0).astype(int)
        df["HasPool"] = (df["PoolQC"] > 0).astype(int)

        df["HasPorch"] = (df["OpenPorchSF"] > 0).astype(int)
        df["HasWoodDeck"] = (df["WoodDeckSF"] > 0).astype(int)
        df["HasMasVnr"] = (df["MasVnrArea"] > 0).astype(int)

        return df

    def _create_temporal_features(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

        df["GarageAge"] = df.apply(
            lambda row: row["YrSold"] - row["GarageYrBlt"]
            if row["GarageYrBlt"] > 0
            else -1,
            axis=1,
        )

        df["IsNewHouse"] = (df["HouseAge"] == 0).astype(int)
        df["IsRemodeled"] = (df["RemodAge"] < df["HouseAge"]).astype(int)

        return df

    def _create_quality_score_features(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        df["QualityConditionGap"] = df["OverallQual"] - df["OverallCond"]
        df["ExteriorScore"] = (df["ExterQual"] + df["ExterCond"]) / 2.0
        df["GarageScore"] = (df["GarageQual"] + df["GarageCond"]) / 2.0
        df["BasementScore"] = (df["BsmtQual"] + df["BsmtCond"]) / 2.0
        df["FireplaceScore"] = df["FireplaceQu"]
        df["KitchenScore"] = df["KitchenQual"]
        df["HeatingScore"] = df["HeatingQC"]

        return df

    def _create_interaction_features(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        df["QualAreaInteraction"] = df["OverallQual"] * df["TotalSF"]
        df["QualTotalSF"] = (df["OverallQual"] * df["TotalSF"]) / 1000.0
        df["CondAreaInteraction"] = df["OverallCond"] * df["TotalSF"]

        df["QualGarageInteraction"] = (
            df["OverallQual"] * df["HasGarage"] * df["GarageArea"]
        )

        df["QualBasementInteraction"] = df["OverallQual"] * df["TotalBsmtSF"]

        df["QualGrLivAreaInteraction"] = df["OverallQual"] * df["GrLivArea"]

        return df

    def _create_ratio_features(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        df["GarageAreaPerCar"] = np.where(
            df["GarageCars"] > 0,
            df["GarageArea"] / df["GarageCars"],
            0,
        )

        df["FinishedBsmtRatio"] = np.where(
            df["TotalBsmtSF"] > 0,
            (df["BsmtFinSF1"] + df["BsmtFinSF2"]) / df["TotalBsmtSF"],
            0,
        )

        df["BasementRatio"] = np.where(
            df["TotalSF"] > 0,
            df["TotalBsmtSF"] / df["TotalSF"],
            0,
        )

        df["LivingAreaRatio"] = np.where(
            df["TotalSF"] > 0,
            df["GrLivArea"] / df["TotalSF"],
            0,
        )

        df["PorchRatio"] = np.where(
            df["TotalSF"] > 0,
            df["TotalPorchSF"] / df["TotalSF"],
            0,
        )

        return df

    def _group_rare_categories(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        rare_threshold = 0.01
        rare_cutoff = int(rare_threshold * self.n_train_)

        nominal_features = [
            "Neighborhood",
            "Condition1",
            "Condition2",
            "HouseStyle",
            "Exterior1st",
            "Exterior2nd",
            "RoofMatl",
            "RoofStyle",
            "Foundation",
            "SaleType",
            "SaleCondition",
        ]

        self.rare_mapping_ = {}

        for col in nominal_features:
            if col not in df.columns:
                continue

            train_subset = df.iloc[: self.n_train_]
            value_counts = train_subset[col].value_counts()

            rare_cats = value_counts[value_counts <= rare_cutoff].index.tolist()

            if rare_cats:
                mapping = {cat: "Rare" for cat in rare_cats}
                self.rare_mapping_[col] = mapping
                df[col] = df[col].replace(mapping)

        return df

    def _one_hot_encode(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        if categorical_cols:
            df = pd.get_dummies(
                df,
                columns=categorical_cols,
                drop_first=True,
                dtype=int,
            )

        return df

    def _apply_skew_transform(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        no_transform = {
            "OverallQual", "OverallCond", "GarageQual", "GarageCond",
            "BsmtQual", "BsmtCond", "ExterQual", "ExterCond",
            "FireplaceQu", "HeatingQC", "KitchenQual", "PoolQC",
            "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "GarageFinish",
            "PavedDrive", "LandSlope", "Functional",
            "GarageCars", "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath",
            "Fireplaces", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
            "TotalBath", "TotalFullBath", "TotalHalfBath",
            "HouseAge", "RemodAge", "GarageAge",
            "YrSold", "MoSold",
            "HasGarage", "HasBasement", "HasFireplace", "HasPool",
            "HasPorch", "HasWoodDeck", "HasMasVnr", "CentralAir", "Street",
            "IsRemodeled", "IsNewHouse",
            "QualityConditionGap", "ExteriorScore", "GarageScore",
            "BasementScore", "FireplaceScore", "KitchenScore", "HeatingScore",
            "QualAreaInteraction", "QualTotalSF", "CondAreaInteraction",
            "QualGarageInteraction", "QualBasementInteraction",
            "QualGrLivAreaInteraction",
            "BathPerBedroom", "RoomPerBedroom", "GarageAreaPerCar",
            "FinishedBsmtRatio", "BasementRatio", "LivingAreaRatio", "PorchRatio",
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        binary_cols = [
            col for col in numeric_cols
            if set(df[col].dropna().unique()).issubset({0, 1})
        ]

        no_transform_final = set(no_transform).union(binary_cols)

        train_numeric = df.iloc[: self.n_train_][numeric_cols]

        skewness_scores = {}

        for col in numeric_cols:
            if col not in no_transform_final:
                values = train_numeric[col].replace([np.inf, -np.inf], np.nan).fillna(0)
                skewness_scores[col] = abs(skew(values))

        high_skew = {
            col: score
            for col, score in skewness_scores.items()
            if score > 0.75
        }

        self.high_skew_columns_ = list(high_skew.keys())

        for col in self.high_skew_columns_:
            df[col] = np.log1p(df[col].clip(lower=0))

        return df

    def _drop_redundant_and_low_variance_features(self, full_data: pd.DataFrame) -> pd.DataFrame:
        df = full_data.copy()

        year_cols_to_drop = ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]

        for col in year_cols_to_drop:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        selector = VarianceThreshold(threshold=0.01)
        selector.fit(df.iloc[: self.n_train_][numeric_cols])

        self.constant_columns_ = [
            col for col, keep in zip(numeric_cols, selector.get_support())
            if not keep
        ]

        if self.constant_columns_:
            df.drop(columns=self.constant_columns_, inplace=True)

        return df

    def _split_back(self, full_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_train = full_data[: self.n_train_].copy()
        X_test = full_data[self.n_train_:].copy()

        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)

        return X_train, X_test

    def _remove_dead_dummy_columns(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dead_cols = [
            col for col in X_train.columns
            if (X_train[col] == 0).all()
        ]

        self.dead_dummy_columns_ = dead_cols

        if dead_cols:
            X_train = X_train.drop(columns=dead_cols)
            X_test = X_test.drop(columns=dead_cols)

        return X_train, X_test

    def _build_metadata(
        self,
        original_train_shape: Tuple[int, int],
        original_test_shape: Tuple[int, int],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_log: pd.Series,
        train_ids: pd.Series | None,
        test_ids: pd.Series | None,
    ) -> Dict[str, Any]:
        train_numeric = X_train.select_dtypes(include=[np.number])
        test_numeric = X_test.select_dtypes(include=[np.number])

        metadata = {
            "original_train_shape": list(original_train_shape),
            "original_test_shape": list(original_test_shape),
            "clean_train_rows_after_outlier_removal": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "feature_count": int(X_train.shape[1]),
            "target_raw": self.target_column,
            "target_log": self.target_log_column,
            "removed_outlier_ids": self.removed_outlier_ids_,
            "removed_outlier_indices": self.removed_outlier_indices_,
            "same_columns": list(X_train.columns) == list(X_test.columns),
            "train_missing_values": int(X_train.isna().sum().sum()),
            "test_missing_values": int(X_test.isna().sum().sum()),
            "train_infinite_values": int(np.isinf(train_numeric).sum().sum()),
            "test_infinite_values": int(np.isinf(test_numeric).sum().sum()),
            "high_skew_columns_count": len(self.high_skew_columns_),
            "constant_columns_removed": self.constant_columns_,
            "dead_dummy_columns_removed": self.dead_dummy_columns_,
            "rare_mapping_columns": list(self.rare_mapping_.keys()),
            "train_id_count": int(len(train_ids)) if train_ids is not None else None,
            "test_id_count": int(len(test_ids)) if test_ids is not None else None,
            "note": "Production conversion of Feature_Engineering.ipynb logic. No model training included.",
        }

        return metadata