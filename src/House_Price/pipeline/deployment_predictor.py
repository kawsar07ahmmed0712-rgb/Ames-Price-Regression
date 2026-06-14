import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


class DeploymentPredictionPipeline:
    def __init__(self, bundle_dir: str = "artifacts/deployment"):
        self.bundle_dir = Path(bundle_dir)

        self.preprocessor_path = self.bundle_dir / "preprocessor.pkl"
        self.model_path = self.bundle_dir / "deploy_model.pkl"
        self.metadata_path = self.bundle_dir / "deploy_metadata.json"

        self._check_required_files()

        self.preprocessor = self._load_pickle(self.preprocessor_path)
        self.model = self._load_pickle(self.model_path)
        self.metadata = self._load_json(self.metadata_path)

        self.feature_names = self.metadata["feature_names"]
        self.default_row = self.metadata["default_row"]
        self.form_options = self.metadata["form_options"]
        self.default_form_values = self.metadata["default_form_values"]

    def _check_required_files(self):
        required_files = [
            self.preprocessor_path,
            self.model_path,
            self.metadata_path,
        ]

        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(
                    f"Deployment artifact not found: {path}. "
                    "Create it locally using scripts/create_deployment_bundle.py"
                )

    @staticmethod
    def _load_pickle(path: Path):
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def _load_json(path: Path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def get_form_options(self) -> Dict[str, Any]:
        return self.form_options

    def get_default_form_values(self) -> Dict[str, Any]:
        return self.default_form_values

    def build_raw_dataframe(self, form_values: Dict[str, Any]) -> pd.DataFrame:
        row = dict(self.default_row)

        rename_map = {
            "FirstFlrSF": "1stFlrSF",
        }

        for key, value in form_values.items():
            raw_key = rename_map.get(key, key)
            row[raw_key] = value

        garage_cars = int(form_values.get("GarageCars", row.get("GarageCars", 0)))
        total_bsmt_sf = float(form_values.get("TotalBsmtSF", row.get("TotalBsmtSF", 0)))

        if garage_cars == 0:
            row["GarageType"] = "NoGarage"
            row["GarageFinish"] = "NoGarage"
            row["GarageQual"] = "NoGarage"
            row["GarageCond"] = "NoGarage"
            row["GarageYrBlt"] = 0
            row["GarageArea"] = 0

        if total_bsmt_sf == 0:
            row["BsmtQual"] = "NoBasement"
            row["BsmtCond"] = "NoBasement"
            row["BsmtExposure"] = "NoBasement"
            row["BsmtFinType1"] = "NoBasement"
            row["BsmtFinType2"] = "NoBasement"
            row["BsmtFinSF1"] = 0
            row["BsmtFinSF2"] = 0
            row["BsmtUnfSF"] = 0
            row["TotalBsmtSF"] = 0
            row["BsmtFullBath"] = 0
            row["BsmtHalfBath"] = 0

        raw_df = pd.DataFrame([row])

        return raw_df

    def transform_raw_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        try:
            X = self.preprocessor.transform(
                raw_df,
                strict_unknown_categories=False,
            )
        except TypeError:
            X = self.preprocessor.transform(raw_df)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)

        missing_features = [col for col in self.feature_names if col not in X.columns]

        if missing_features:
            for col in missing_features:
                X[col] = 0

        X = X[self.feature_names]

        return X

    def predict(self, form_values: Dict[str, Any]) -> Dict[str, Any]:
        raw_df = self.build_raw_dataframe(form_values)
        X = self.transform_raw_dataframe(raw_df)

        log_prediction = float(self.model.predict(X)[0])
        price = float(np.expm1(log_prediction))

        if price < 0:
            price = 0.0

        return {
            "PredictedSalePrice": price,
            "PredictedSalePriceFormatted": f"${price:,.0f}",
            "model_type": self.metadata.get("model_type", "Deployment model"),
            "feature_count": self.metadata.get("feature_count", len(self.feature_names)),
        }
