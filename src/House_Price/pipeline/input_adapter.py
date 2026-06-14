import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from House_Price.utils.exception import CustomException
from House_Price.utils.logger import logger


class AmesRawInputAdapter:
    """
    Converts frontend/basic user input into a full raw Ames-style row.

    The model/preprocessor expects the original Ames raw schema.
    The user will only provide important fields.
    This adapter fills the remaining raw fields using training-data defaults.
    """

    def __init__(
        self,
        raw_train_path: str,
        raw_test_path: str,
        schema: Dict[str, Any],
    ):
        self.raw_train_path = raw_train_path
        self.raw_test_path = raw_test_path
        self.schema = schema

        self.id_column = self.schema["columns"]["id_column"]
        self.target_column = self.schema["columns"]["target_column"]

        self.raw_columns: List[str] = []
        self.default_row: Dict[str, Any] = {}
        self.form_options: Dict[str, List[Any]] = {}

        self._fit_defaults()

    def _check_file_exists(self, file_path: str, file_label: str) -> None:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"{file_label} file not found: {path}")

        if not path.is_file():
            raise ValueError(f"{file_label} path exists but is not a file: {path}")

    def _fit_defaults(self) -> None:
        try:
            self._check_file_exists(self.raw_train_path, "Raw train")
            self._check_file_exists(self.raw_test_path, "Raw test")

            train_df = pd.read_csv(self.raw_train_path)
            test_df = pd.read_csv(self.raw_test_path)

            self.raw_columns = list(test_df.columns)

            train_features = train_df.drop(columns=[self.target_column], errors="ignore")

            default_row = {}

            for col in self.raw_columns:
                if col == self.id_column:
                    default_row[col] = 999999
                    continue

                if col not in train_features.columns:
                    default_row[col] = np.nan
                    continue

                if pd.api.types.is_numeric_dtype(train_features[col]):
                    default_row[col] = float(train_features[col].median())
                else:
                    mode_values = train_features[col].mode(dropna=True)
                    default_row[col] = mode_values.iloc[0] if len(mode_values) > 0 else "None"

            self.default_row = default_row

            self.form_options = {
                "Neighborhood": sorted(train_features["Neighborhood"].dropna().unique().tolist()),
                "OverallQual": list(range(1, 11)),
                "GarageCars": [0, 1, 2, 3, 4],
                "FullBath": [0, 1, 2, 3, 4],
                "TotRmsAbvGrd": list(range(2, 15)),
                "quality_values": ["Po", "Fa", "TA", "Gd", "Ex"],
                "bsmt_quality_values": ["NoBasement", "Po", "Fa", "TA", "Gd", "Ex"],
                "garage_finish_values": ["NoGarage", "Unf", "RFn", "Fin"],
            }

            logger.info("AmesRawInputAdapter defaults fitted successfully.")

        except Exception as e:
            logger.error("AmesRawInputAdapter default fitting failed.")
            raise CustomException(e, sys)

    def get_form_options(self) -> Dict[str, List[Any]]:
        return self.form_options

    def build_raw_dataframe(self, user_input: Dict[str, Any]) -> pd.DataFrame:
        """
        Build one full raw Ames-style row from partial frontend input.
        """
        try:
            row = self.default_row.copy()

            mapped_input = user_input.copy()

            if "FirstFlrSF" in mapped_input:
                mapped_input["1stFlrSF"] = mapped_input.pop("FirstFlrSF")

            for col, value in mapped_input.items():
                if col in row:
                    row[col] = value

            if not row.get("YearRemodAdd"):
                row["YearRemodAdd"] = row.get("YearBuilt", self.default_row.get("YearRemodAdd"))

            if row.get("GarageCars", 0) == 0:
                row["GarageType"] = "NoGarage"
                row["GarageFinish"] = "NoGarage"
                row["GarageQual"] = "NoGarage"
                row["GarageCond"] = "NoGarage"
                row["GarageArea"] = 0
                row["GarageYrBlt"] = 0

            if row.get("TotalBsmtSF", 0) == 0:
                row["BsmtQual"] = "NoBasement"
                row["BsmtCond"] = "NoBasement"
                row["BsmtExposure"] = "NoBasement"
                row["BsmtFinType1"] = "NoBasement"
                row["BsmtFinType2"] = "NoBasement"
                row["BsmtFinSF1"] = 0
                row["BsmtFinSF2"] = 0
                row["BsmtUnfSF"] = 0
                row["BsmtFullBath"] = 0
                row["BsmtHalfBath"] = 0

            raw_df = pd.DataFrame([row], columns=self.raw_columns)

            return raw_df

        except Exception as e:
            logger.error("Failed to build raw dataframe from user input.")
            raise CustomException(e, sys)