from __future__ import annotations

import json
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "Data" / "train.csv"

PREPROCESSOR_PATH = (
    ROOT
    / "artifacts"
    / "data_transformation"
    / "preprocessor.pkl"
)

FEATURE_NAMES_PATH = (
    ROOT
    / "artifacts"
    / "data_transformation"
    / "feature_names.json"
)

FINAL_MODEL_PATH = (
    ROOT
    / "artifacts"
    / "model_trainer"
    / "final_model.pkl"
)

BLEND_MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "model_trainer"
    / "blend_manifest.json"
)

MODEL_METADATA_PATH = (
    ROOT
    / "artifacts"
    / "model_trainer"
    / "model_metadata.json"
)

DEPLOY_DIR = (
    ROOT
    / "artifacts"
    / "deployment"
)

DEPLOY_PREPROCESSOR_PATH = (
    DEPLOY_DIR
    / "preprocessor.pkl"
)

DEPLOY_MODEL_PATH = (
    DEPLOY_DIR
    / "final_model.pkl"
)

DEPLOY_METADATA_PATH = (
    DEPLOY_DIR
    / "deploy_metadata.json"
)

REQUIREMENTS_DEPLOY_PATH = (
    ROOT
    / "requirements-deploy.txt"
)


CORE_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "PyYAML",
    "fastapi",
    "uvicorn",
    "jinja2",
    "python-multipart",
]


OPTIONAL_MODEL_PACKAGES = {
    "catboost": "catboost",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
}


def load_artifact(
    path: Path,
) -> Any:
    """
    Load a trusted model or preprocessor
    artifact saved with joblib.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Artifact not found: {path}"
        )

    return joblib.load(path)


def load_json(
    path: Path,
) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def json_safe(
    value: Any,
) -> Any:
    """
    Convert NumPy and Pandas values
    into JSON-safe Python values.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None

    except (
        TypeError,
        ValueError,
    ):
        pass

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        np.ndarray,
    ):
        return [
            json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    return value


def read_feature_names() -> list[str]:
    data = load_json(
        FEATURE_NAMES_PATH
    )

    if isinstance(
        data,
        list,
    ):
        feature_names = data

    elif isinstance(
        data,
        dict,
    ):
        feature_names = (
            data.get("feature_names")
            or data.get("features")
            or data.get("columns")
        )

    else:
        feature_names = None

    if not feature_names:
        raise ValueError(
            "No feature list was found "
            "inside feature_names.json."
        )

    return [
        str(name)
        for name in feature_names
    ]


def build_default_row(
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Create representative values for
    hidden raw features.

    Numeric columns:
        cleaned training median

    Categorical columns:
        cleaned training mode
    """

    clean_df = train_df.copy()

    if {
        "GrLivArea",
        "SalePrice",
    }.issubset(
        clean_df.columns
    ):
        outlier_mask = (
            (
                clean_df["GrLivArea"]
                > 4000
            )
            & (
                clean_df["SalePrice"]
                < 300000
            )
        )

        clean_df = clean_df.loc[
            ~outlier_mask
        ].copy()

    feature_df = clean_df.drop(
        columns=["SalePrice"],
        errors="ignore",
    )

    if feature_df.empty:
        raise ValueError(
            "Training data is empty."
        )

    default_row: dict[
        str,
        Any,
    ] = {}

    for column in feature_df.columns:
        series = feature_df[column]

        if column == "Id":
            ids = pd.to_numeric(
                series,
                errors="coerce",
            ).dropna()

            default_row[column] = (
                int(ids.max()) + 1
                if not ids.empty
                else 999999
            )

            continue

        if is_numeric_dtype(
            series
        ):
            median_value = pd.to_numeric(
                series,
                errors="coerce",
            ).median()

            if pd.isna(
                median_value
            ):
                median_value = 0

            if is_integer_dtype(
                series.dtype
            ):
                default_row[column] = int(
                    round(
                        float(
                            median_value
                        )
                    )
                )

            else:
                default_row[column] = float(
                    median_value
                )

        else:
            mode_values = (
                series
                .dropna()
                .mode()
            )

            default_row[column] = (
                str(
                    mode_values.iloc[0]
                )
                if not mode_values.empty
                else "None"
            )

    return json_safe(
        default_row
    )


def build_form_options(
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    neighborhoods = sorted(
        train_df[
            "Neighborhood"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return {
        "Neighborhood": neighborhoods,

        "OverallQual": list(
            range(1, 11)
        ),

        "GarageCars": [
            0,
            1,
            2,
            3,
            4,
        ],

        "FullBath": [
            0,
            1,
            2,
            3,
            4,
        ],

        "TotRmsAbvGrd": list(
            range(2, 15)
        ),

        "quality_values": [
            "Po",
            "Fa",
            "TA",
            "Gd",
            "Ex",
        ],

        "bsmt_quality_values": [
            "NoBasement",
            "Po",
            "Fa",
            "TA",
            "Gd",
            "Ex",
        ],

        "garage_finish_values": [
            "NoGarage",
            "Unf",
            "RFn",
            "Fin",
        ],
    }


def build_default_form_values(
    default_row: dict[str, Any],
) -> dict[str, Any]:
    form_to_raw = {
        "Neighborhood": "Neighborhood",
        "OverallQual": "OverallQual",
        "GrLivArea": "GrLivArea",
        "GarageCars": "GarageCars",
        "GarageArea": "GarageArea",
        "YearBuilt": "YearBuilt",
        "YearRemodAdd": "YearRemodAdd",
        "TotalBsmtSF": "TotalBsmtSF",
        "FirstFlrSF": "1stFlrSF",
        "FullBath": "FullBath",
        "TotRmsAbvGrd": "TotRmsAbvGrd",
        "ExterQual": "ExterQual",
        "KitchenQual": "KitchenQual",
        "BsmtQual": "BsmtQual",
        "GarageFinish": "GarageFinish",
    }

    return {
        form_name: default_row.get(
            raw_name
        )
        for form_name, raw_name
        in form_to_raw.items()
    }


def align_features(
    transformed: Any,
    feature_names: list[str],
) -> pd.DataFrame:
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

        if transformed_array.ndim != 2:
            raise ValueError(
                "Preprocessor returned "
                "an invalid array."
            )

        if (
            transformed_array.shape[1]
            != len(feature_names)
        ):
            raise ValueError(
                "Feature count mismatch: "
                f"expected "
                f"{len(feature_names)}, "
                f"received "
                f"{transformed_array.shape[1]}."
            )

        transformed_df = pd.DataFrame(
            transformed_array,
            columns=feature_names,
        )

    missing_features = [
        feature
        for feature in feature_names
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
        columns=feature_names
    )


def smoke_test(
    preprocessor: Any,
    final_model: Any,
    feature_names: list[str],
    default_row: dict[str, Any],
) -> float:
    """
    Test the exact deployment flow
    before writing the bundle.
    """

    raw_df = pd.DataFrame(
        [default_row]
    )

    try:
        transformed = (
            preprocessor.transform(
                raw_df,
                strict_unknown_categories=False,
            )
        )

    except TypeError:
        transformed = (
            preprocessor.transform(
                raw_df
            )
        )

    transformed_df = align_features(
        transformed=transformed,
        feature_names=feature_names,
    )

    prediction = np.asarray(
        final_model.predict_price(
            X=transformed_df,
            apply_tail_lift=True,
            apply_clipping=True,
        ),
        dtype=float,
    )

    if prediction.shape != (1,):
        raise ValueError(
            "Smoke test expected "
            "one prediction, "
            f"received "
            f"{prediction.shape}."
        )

    price = float(
        prediction[0]
    )

    if (
        not np.isfinite(price)
        or price <= 0
    ):
        raise ValueError(
            "Smoke test returned "
            f"an invalid price: {price}"
        )

    return price


def get_runtime_packages(
    final_model: Any,
) -> list[str]:
    packages = list(
        CORE_PACKAGES
    )

    model_modules = {
        type(model).__module__.lower()
        for model
        in getattr(
            final_model,
            "models",
            {},
        ).values()
    }

    for (
        module_keyword,
        package_name,
    ) in OPTIONAL_MODEL_PACKAGES.items():

        if any(
            module_keyword in module_name
            for module_name
            in model_modules
        ):
            packages.append(
                package_name
            )

    return list(
        dict.fromkeys(
            packages
        )
    )


def write_deployment_requirements(
    final_model: Any,
) -> dict[str, str]:
    package_versions: dict[
        str,
        str,
    ] = {}

    missing_packages: list[
        str
    ] = []

    for package_name in (
        get_runtime_packages(
            final_model
        )
    ):
        try:
            package_versions[
                package_name
            ] = version(
                package_name
            )

        except PackageNotFoundError:
            missing_packages.append(
                package_name
            )

    if missing_packages:
        raise RuntimeError(
            "Required deployment "
            "packages are missing: "
            f"{missing_packages}"
        )

    requirements_text = "\n".join(
        f"{package_name}"
        f"=={package_version}"
        for (
            package_name,
            package_version,
        ) in package_versions.items()
    )

    REQUIREMENTS_DEPLOY_PATH.write_text(
        requirements_text + "\n",
        encoding="utf-8",
    )

    return package_versions


def main() -> None:
    required_files = [
        TRAIN_PATH,
        PREPROCESSOR_PATH,
        FEATURE_NAMES_PATH,
        FINAL_MODEL_PATH,
        BLEND_MANIFEST_PATH,
        MODEL_METADATA_PATH,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "Required training artifacts "
            "are missing:\n"
            f"{formatted}\n\n"
            "Run the training pipeline first:\n"
            "python main.py"
        )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    preprocessor = load_artifact(
        PREPROCESSOR_PATH
    )

    final_model = load_artifact(
        FINAL_MODEL_PATH
    )

    feature_names = (
        read_feature_names()
    )

    blend_manifest = load_json(
        BLEND_MANIFEST_PATH
    )

    model_metadata = load_json(
        MODEL_METADATA_PATH
    )

    if not hasattr(
        preprocessor,
        "transform",
    ):
        raise TypeError(
            "Saved preprocessor does "
            "not implement transform()."
        )

    if not hasattr(
        final_model,
        "predict_price",
    ):
        raise TypeError(
            "final_model.pkl is not "
            "AmesEnsembleModel. "
            "predict_price() was not found."
        )

    expected_feature_count = getattr(
        final_model,
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
        != len(feature_names)
    ):
        raise ValueError(
            "Model and preprocessor "
            "feature counts do not match: "
            f"model="
            f"{expected_feature_count}, "
            f"metadata="
            f"{len(feature_names)}."
        )

    default_row = (
        build_default_row(
            train_df
        )
    )

    default_form_values = (
        build_default_form_values(
            default_row
        )
    )

    form_options = (
        build_form_options(
            train_df
        )
    )

    smoke_price = smoke_test(
        preprocessor=preprocessor,
        final_model=final_model,
        feature_names=feature_names,
        default_row=default_row,
    )

    runtime_versions = (
        write_deployment_requirements(
            final_model
        )
    )

    if DEPLOY_DIR.exists():
        shutil.rmtree(
            DEPLOY_DIR
        )

    DEPLOY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Copy the exact evaluated artifacts.
    # Do not train another deployment model.
    shutil.copy2(
        PREPROCESSOR_PATH,
        DEPLOY_PREPROCESSOR_PATH,
    )

    shutil.copy2(
        FINAL_MODEL_PATH,
        DEPLOY_MODEL_PATH,
    )

    model_manifest = (
        final_model.get_manifest()
        if hasattr(
            final_model,
            "get_manifest",
        )
        else blend_manifest
    )

    metadata = {
        "bundle_version": "1.0.0",

        "model_type": (
            type(
                final_model
            ).__name__
        ),

        "selected_strategy": (
            model_metadata.get(
                "selected_strategy",
                getattr(
                    final_model,
                    "metadata",
                    {},
                ).get(
                    "selected_strategy",
                    "unknown",
                ),
            )
        ),

        "feature_count": len(
            feature_names
        ),

        "feature_names": (
            feature_names
        ),

        "default_row": (
            default_row
        ),

        "form_options": (
            form_options
        ),

        "default_form_values": (
            default_form_values
        ),

        "model_manifest": (
            json_safe(
                model_manifest
            )
        ),

        "model_metadata": (
            json_safe(
                model_metadata
            )
        ),

        "blend_manifest": (
            json_safe(
                blend_manifest
            )
        ),

        "runtime_versions": (
            runtime_versions
        ),

        "smoke_test": {
            "passed": True,

            "default_prediction": (
                smoke_price
            ),
        },
    }

    DEPLOY_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Deployment bundle "
        "created successfully."
    )

    print(
        f"Preprocessor : "
        f"{DEPLOY_PREPROCESSOR_PATH}"
    )

    print(
        f"Final model  : "
        f"{DEPLOY_MODEL_PATH}"
    )

    print(
        f"Metadata     : "
        f"{DEPLOY_METADATA_PATH}"
    )

    print(
        f"Requirements : "
        f"{REQUIREMENTS_DEPLOY_PATH}"
    )

    print(
        f"Model type   : "
        f"{type(final_model).__name__}"
    )

    print(
        f"Strategy     : "
        f"{metadata['selected_strategy']}"
    )

    print(
        f"Feature count: "
        f"{len(feature_names)}"
    )

    print(
        f"Smoke price  : "
        f"${smoke_price:,.0f}"
    )


if __name__ == "__main__":
    main()