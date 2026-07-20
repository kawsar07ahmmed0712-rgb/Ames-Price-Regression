from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)


class DeploymentPredictionPipeline:
    """
    Inference-only deployment pipeline.

    It loads:
    1. Saved feature preprocessor
    2. Exact final AmesEnsembleModel
    3. Deployment metadata

    It does not fit or train anything.
    """

    def __init__(
        self,
        bundle_dir: str | Path = (
            "artifacts/deployment"
        ),
    ) -> None:
        bundle_path = Path(
            bundle_dir
        )

        self.bundle_dir = (
            bundle_path
            if bundle_path.is_absolute()
            else PROJECT_ROOT
            / bundle_path
        )

        self.preprocessor_path = (
            self.bundle_dir
            / "preprocessor.pkl"
        )

        self.model_path = (
            self.bundle_dir
            / "final_model.pkl"
        )

        self.metadata_path = (
            self.bundle_dir
            / "deploy_metadata.json"
        )

        self._check_required_files()

        self.preprocessor = (
            self._load_artifact(
                self.preprocessor_path
            )
        )

        self.model = (
            self._load_artifact(
                self.model_path
            )
        )

        self.metadata = (
            self._load_json(
                self.metadata_path
            )
        )

        self.feature_names = list(
            self.metadata[
                "feature_names"
            ]
        )

        self.default_row = dict(
            self.metadata[
                "default_row"
            ]
        )

        self.form_options = dict(
            self.metadata[
                "form_options"
            ]
        )

        self.default_form_values = dict(
            self.metadata[
                "default_form_values"
            ]
        )

        self._validate_loaded_objects()

    def _check_required_files(
        self,
    ) -> None:
        required_files = [
            self.preprocessor_path,
            self.model_path,
            self.metadata_path,
        ]

        missing_files = [
            str(path)
            for path in required_files
            if not path.is_file()
        ]

        if missing_files:
            formatted = "\n".join(
                f"  - {path}"
                for path
                in missing_files
            )

            raise FileNotFoundError(
                "Deployment artifacts "
                "are missing:\n"
                f"{formatted}\n\n"
                "Create them with:\n"
                "python scripts/"
                "create_deployment_bundle.py"
            )

    @staticmethod
    def _load_artifact(
        path: Path,
    ) -> Any:
        """
        Load an artifact saved
        by this project with joblib.
        """

        if not path.is_file():
            raise FileNotFoundError(
                f"Artifact not found: {path}"
            )

        return joblib.load(path)

    @staticmethod
    def _load_json(
        path: Path,
    ) -> Dict[str, Any]:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def _validate_loaded_objects(
        self,
    ) -> None:
        if not hasattr(
            self.preprocessor,
            "transform",
        ):
            raise TypeError(
                "Deployment preprocessor "
                "does not implement transform()."
            )

        if not hasattr(
            self.model,
            "predict_price",
        ):
            raise TypeError(
                "Deployment model is incorrect. "
                "AmesEnsembleModel.predict_price() "
                "was not found."
            )

        expected_feature_count = getattr(
            self.model,
            "metadata",
            {},
        ).get(
            "feature_count"
        )

        if (
            expected_feature_count
            is not None
            and int(
                expected_feature_count
            )
            != len(
                self.feature_names
            )
        ):
            raise ValueError(
                "Deployment feature "
                "contract mismatch: "
                f"model="
                f"{expected_feature_count}, "
                f"metadata="
                f"{len(self.feature_names)}."
            )

    def get_form_options(
        self,
    ) -> Dict[str, Any]:
        return dict(
            self.form_options
        )

    def get_default_form_values(
        self,
    ) -> Dict[str, Any]:
        return dict(
            self.default_form_values
        )

    def get_model_info(
        self,
    ) -> Dict[str, Any]:
        return {
            "model_type": (
                self.metadata.get(
                    "model_type",
                    type(
                        self.model
                    ).__name__,
                )
            ),

            "selected_strategy": (
                self.metadata.get(
                    "selected_strategy",
                    "unknown",
                )
            ),

            "feature_count": int(
                self.metadata.get(
                    "feature_count",
                    len(
                        self.feature_names
                    ),
                )
            ),

            "bundle_version": (
                self.metadata.get(
                    "bundle_version",
                    "unknown",
                )
            ),
        }

    @staticmethod
    def _validate_form_values(
        form_values: Dict[str, Any],
    ) -> None:
        required_fields = [
            "Neighborhood",
            "OverallQual",
            "GrLivArea",
            "GarageCars",
            "GarageArea",
            "YearBuilt",
            "YearRemodAdd",
            "TotalBsmtSF",
            "FirstFlrSF",
            "FullBath",
            "TotRmsAbvGrd",
            "ExterQual",
            "KitchenQual",
            "BsmtQual",
            "GarageFinish",
        ]

        missing_fields = [
            field
            for field in required_fields
            if field
            not in form_values
        ]

        if missing_fields:
            raise ValueError(
                "Required form fields "
                "are missing: "
                f"{missing_fields}"
            )

        non_negative_fields = [
            "GrLivArea",
            "GarageCars",
            "GarageArea",
            "TotalBsmtSF",
            "FirstFlrSF",
            "FullBath",
            "TotRmsAbvGrd",
        ]

        for field in (
            non_negative_fields
        ):
            if (
                float(
                    form_values[field]
                )
                < 0
            ):
                raise ValueError(
                    f"{field} "
                    "cannot be negative."
                )

        overall_quality = int(
            form_values[
                "OverallQual"
            ]
        )

        if not (
            1
            <= overall_quality
            <= 10
        ):
            raise ValueError(
                "OverallQual must be "
                "between 1 and 10."
            )

        year_built = int(
            form_values[
                "YearBuilt"
            ]
        )

        year_remodeled = int(
            form_values[
                "YearRemodAdd"
            ]
        )

        if not (
            1800
            <= year_built
            <= 2100
        ):
            raise ValueError(
                "YearBuilt must be "
                "between 1800 and 2100."
            )

        if not (
            1800
            <= year_remodeled
            <= 2100
        ):
            raise ValueError(
                "YearRemodAdd must be "
                "between 1800 and 2100."
            )

        if (
            year_remodeled
            < year_built
        ):
            raise ValueError(
                "YearRemodAdd cannot "
                "be earlier than YearBuilt."
            )

        garage_cars = int(
            form_values[
                "GarageCars"
            ]
        )

        garage_finish = str(
            form_values[
                "GarageFinish"
            ]
        )

        if (
            garage_cars > 0
            and garage_finish
            == "NoGarage"
        ):
            raise ValueError(
                "GarageFinish cannot be "
                "NoGarage when GarageCars "
                "is above zero."
            )

        basement_area = float(
            form_values[
                "TotalBsmtSF"
            ]
        )

        basement_quality = str(
            form_values[
                "BsmtQual"
            ]
        )

        if (
            basement_area > 0
            and basement_quality
            == "NoBasement"
        ):
            raise ValueError(
                "BsmtQual cannot be "
                "NoBasement when "
                "TotalBsmtSF is above zero."
            )

    def build_raw_dataframe(
        self,
        form_values: Dict[str, Any],
    ) -> pd.DataFrame:
        self._validate_form_values(
            form_values
        )

        row = dict(
            self.default_row
        )

        rename_map = {
            "FirstFlrSF": "1stFlrSF",
        }

        for (
            key,
            value,
        ) in form_values.items():

            raw_key = rename_map.get(
                key,
                key,
            )

            row[raw_key] = value

        garage_cars = int(
            form_values[
                "GarageCars"
            ]
        )

        total_bsmt_sf = float(
            form_values[
                "TotalBsmtSF"
            ]
        )

        if garage_cars == 0:
            row.update(
                {
                    "GarageType": (
                        "NoGarage"
                    ),

                    "GarageFinish": (
                        "NoGarage"
                    ),

                    "GarageQual": (
                        "NoGarage"
                    ),

                    "GarageCond": (
                        "NoGarage"
                    ),

                    "GarageYrBlt": 0,

                    "GarageCars": 0,

                    "GarageArea": 0,
                }
            )

        else:
            row["GarageYrBlt"] = int(
                form_values[
                    "YearBuilt"
                ]
            )

        if total_bsmt_sf == 0:
            row.update(
                {
                    "BsmtQual": (
                        "NoBasement"
                    ),

                    "BsmtCond": (
                        "NoBasement"
                    ),

                    "BsmtExposure": (
                        "NoBasement"
                    ),

                    "BsmtFinType1": (
                        "NoBasement"
                    ),

                    "BsmtFinType2": (
                        "NoBasement"
                    ),

                    "BsmtFinSF1": 0,

                    "BsmtFinSF2": 0,

                    "BsmtUnfSF": 0,

                    "TotalBsmtSF": 0,

                    "BsmtFullBath": 0,

                    "BsmtHalfBath": 0,
                }
            )

        return pd.DataFrame(
            [row]
        )

    def transform_raw_dataframe(
        self,
        raw_df: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            transformed = (
                self.preprocessor.transform(
                    raw_df,
                    strict_unknown_categories=False,
                )
            )

        except TypeError:
            transformed = (
                self.preprocessor.transform(
                    raw_df
                )
            )

        if isinstance(
            transformed,
            pd.DataFrame,
        ):
            transformed_df = (
                transformed.copy()
            )

        else:
            transformed_array = np.asarray(
                transformed
            )

            if (
                transformed_array.ndim
                != 2
            ):
                raise ValueError(
                    "Preprocessor returned "
                    "an invalid array."
                )

            if (
                transformed_array.shape[1]
                != len(
                    self.feature_names
                )
            ):
                raise ValueError(
                    "Feature count mismatch: "
                    f"expected "
                    f"{len(self.feature_names)}, "
                    f"received "
                    f"{transformed_array.shape[1]}."
                )

            transformed_df = (
                pd.DataFrame(
                    transformed_array,
                    columns=(
                        self.feature_names
                    ),
                )
            )

        missing_features = [
            feature
            for feature
            in self.feature_names
            if feature
            not in transformed_df.columns
        ]

        if missing_features:
            raise ValueError(
                "Preprocessor output is "
                "missing required features: "
                f"{missing_features[:20]}"
            )

        return transformed_df.reindex(
            columns=(
                self.feature_names
            )
        )

    def predict(
        self,
        form_values: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_df = (
            self.build_raw_dataframe(
                form_values
            )
        )

        transformed_df = (
            self.transform_raw_dataframe(
                raw_df
            )
        )

        predictions = np.asarray(
            self.model.predict_price(
                X=transformed_df,
                apply_tail_lift=True,
                apply_clipping=True,
            ),
            dtype=float,
        )

        if predictions.shape != (1,):
            raise ValueError(
                "Expected one prediction, "
                f"received shape "
                f"{predictions.shape}."
            )

        price = float(
            predictions[0]
        )

        if (
            not np.isfinite(price)
            or price <= 0
        ):
            raise ValueError(
                "Model returned an "
                f"invalid price: {price}"
            )

        model_info = (
            self.get_model_info()
        )

        return {
            "PredictedSalePrice": (
                price
            ),

            "PredictedSalePriceFormatted": (
                f"${price:,.0f}"
            ),

            "model_type": (
                model_info[
                    "model_type"
                ]
            ),

            "selected_strategy": (
                model_info[
                    "selected_strategy"
                ]
            ),

            "feature_count": (
                model_info[
                    "feature_count"
                ]
            ),
        }