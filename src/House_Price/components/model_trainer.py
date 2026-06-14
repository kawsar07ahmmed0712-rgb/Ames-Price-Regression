import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

from House_Price.entity.config_entity import ModelTrainerConfig
from House_Price.model.model_package import AmesEnsembleModel
from House_Price.utils.common import save_json, save_object
from House_Price.utils.exception import CustomException
from House_Price.utils.logger import logger


class ModelTrainer:
    """
    Model trainer component.

    Responsibility:
    - Load transformed training data
    - Train Step 10 notebook-style final ensemble
    - Compute OOF scores
    - Build final AmesEnsembleModel package
    - Save final_model.pkl, blend_manifest.json, and model_metadata.json

    Important:
    - This component does NOT redo hyperparameter tuning.
    - This component does NOT create Kaggle submission files.
    """

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.target_column = "SalePrice_log"

        self.n_splits = 10
        self.seeds = [42, 2024]
        self.inverse_power = 8
        self.selection_margin = 0.015

        self.final_blend_weights = {
            "inverse": 0.45,
            "public_safe": 0.55,
        }

        self.tail_lift_strength = 0.010
        self.tail_lift_start_quantile = 0.90

    def _check_file_exists(self, file_path: str, file_label: str) -> None:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"{file_label} file not found: {path}")

        if not path.is_file():
            raise ValueError(f"{file_label} path exists but is not a file: {path}")

    def _load_training_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        self._check_file_exists(
            self.config.transformed_train_path,
            "Transformed train",
        )

        train_df = pd.read_csv(self.config.transformed_train_path)

        if self.target_column not in train_df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in transformed train data."
            )

        X = train_df.drop(columns=[self.target_column])
        y_log = train_df[self.target_column]
        y_original = np.expm1(y_log)

        logger.info(f"Transformed training data loaded with shape: {train_df.shape}")
        logger.info(f"X shape: {X.shape}")
        logger.info(f"y_log shape: {y_log.shape}")

        return X, y_log, y_original

    def _define_base_models(self) -> Dict[str, Any]:
        """
        Define stable base models from Model_training.ipynb Step 10.
        No hyperparameter tuning is performed here.
        """
        models: Dict[str, Any] = {}

        models["ElasticNet"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", ElasticNet(
                alpha=0.0003,
                l1_ratio=0.9,
                max_iter=200000,
                random_state=42,
            )),
        ])

        models["Ridge"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", Ridge(alpha=10)),
        ])

        models["Lasso"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", Lasso(
                alpha=0.0005,
                max_iter=200000,
                random_state=42,
            )),
        ])

        models["SVR"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", SVR(
                C=50,
                epsilon=0.01,
                gamma="scale",
            )),
        ])

        models["KernelRidge"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", KernelRidge(
                alpha=0.6,
                kernel="polynomial",
                degree=2,
                coef0=2.5,
            )),
        ])

        models["ExtraTrees"] = ExtraTreesRegressor(
            n_estimators=800,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

        try:
            from catboost import CatBoostRegressor

            models["CatBoost"] = CatBoostRegressor(
                iterations=2500,
                learning_rate=0.02,
                depth=4,
                l2_leaf_reg=4,
                loss_function="RMSE",
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            )

            logger.info("CatBoost model added.")

        except Exception as e:
            logger.warning(f"CatBoost skipped: {e}")

        try:
            from xgboost import XGBRegressor

            models["XGBoost"] = XGBRegressor(
                n_estimators=2500,
                learning_rate=0.02,
                max_depth=3,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.001,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )

            logger.info("XGBoost model added.")

        except Exception as e:
            logger.warning(f"XGBoost skipped: {e}")

        try:
            from lightgbm import LGBMRegressor

            models["LightGBM"] = LGBMRegressor(
                n_estimators=2500,
                learning_rate=0.015,
                num_leaves=20,
                max_depth=4,
                min_child_samples=20,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.001,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )

            logger.info("LightGBM model added.")

        except Exception as e:
            logger.warning(f"LightGBM skipped: {e}")

        logger.info(f"Total base models defined: {len(models)}")
        logger.info(f"Base model names: {list(models.keys())}")

        return models

    def _rmse_log(self, y_true_log: pd.Series, y_pred_log: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))

    def _repeated_kfold_oof_prediction(
        self,
        model: Any,
        X: pd.DataFrame,
        y_log: pd.Series,
    ) -> np.ndarray:
        """
        Repeated KFold OOF prediction from notebook Step 10.

        This is used only for estimating model/blend OOF performance
        and calculating inverse RMSE weights.
        """
        oof_sum = np.zeros(X.shape[0])
        oof_count = np.zeros(X.shape[0])

        for seed in self.seeds:
            kf = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=seed,
            )

            for fold, (train_idx, valid_idx) in enumerate(kf.split(X), 1):
                X_train_fold = X.iloc[train_idx]
                X_valid_fold = X.iloc[valid_idx]
                y_train_fold = y_log.iloc[train_idx]

                model_clone = clone(model)
                model_clone.fit(X_train_fold, y_train_fold)

                valid_pred = model_clone.predict(X_valid_fold)

                oof_sum[valid_idx] += valid_pred
                oof_count[valid_idx] += 1

        oof_pred = oof_sum / oof_count

        return oof_pred

    def _generate_oof_predictions(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y_log: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate OOF predictions and model scores.
        """
        oof_predictions = pd.DataFrame(index=X.index)
        model_scores: List[Dict[str, Any]] = []

        for model_name, model in models.items():
            logger.info(f"Training OOF model: {model_name}")
            print(f"Training OOF model: {model_name}")

            oof_pred_log = self._repeated_kfold_oof_prediction(
                model=model,
                X=X,
                y_log=y_log,
            )

            score = self._rmse_log(y_log, oof_pred_log)

            oof_predictions[model_name] = oof_pred_log

            model_scores.append({
                "Model": model_name,
                "OOF_RMSE_log1p": score,
            })

            logger.info(f"{model_name} OOF RMSE log1p: {score}")
            print(f"{model_name} OOF RMSE log1p: {score:.6f}")

        model_scores_df = pd.DataFrame(model_scores).sort_values(
            by="OOF_RMSE_log1p",
            ascending=True,
        )

        return oof_predictions, model_scores_df

    def _select_strong_models(self, model_scores_df: pd.DataFrame) -> List[str]:
        best_score = model_scores_df["OOF_RMSE_log1p"].min()

        selected_models = model_scores_df[
            model_scores_df["OOF_RMSE_log1p"] <= best_score + self.selection_margin
        ]["Model"].tolist()

        logger.info(f"Best single model OOF RMSE: {best_score}")
        logger.info(f"Selected models for inverse blend: {selected_models}")

        return selected_models

    def _calculate_inverse_weights(
        self,
        selected_models: List[str],
        model_scores_df: pd.DataFrame,
    ) -> Dict[str, float]:
        selected_scores = (
            model_scores_df
            .set_index("Model")
            .loc[selected_models]["OOF_RMSE_log1p"]
        )

        raw_weights = 1 / (selected_scores ** self.inverse_power)
        inverse_weights = raw_weights / raw_weights.sum()

        return {
            model_name: float(weight)
            for model_name, weight in inverse_weights.items()
        }

    def _calculate_public_safe_weights(
        self,
        available_model_names: List[str],
    ) -> Dict[str, float]:
        public_safe_order = [
            "ElasticNet",
            "Ridge",
            "Lasso",
            "SVR",
            "KernelRidge",
            "CatBoost",
            "XGBoost",
            "LightGBM",
        ]

        manual_weights = {}

        if "ElasticNet" in available_model_names:
            manual_weights["ElasticNet"] = 0.30
        if "Ridge" in available_model_names:
            manual_weights["Ridge"] = 0.15
        if "Lasso" in available_model_names:
            manual_weights["Lasso"] = 0.10
        if "SVR" in available_model_names:
            manual_weights["SVR"] = 0.10
        if "KernelRidge" in available_model_names:
            manual_weights["KernelRidge"] = 0.10
        if "CatBoost" in available_model_names:
            manual_weights["CatBoost"] = 0.15
        if "XGBoost" in available_model_names:
            manual_weights["XGBoost"] = 0.05
        if "LightGBM" in available_model_names:
            manual_weights["LightGBM"] = 0.05

        ordered_weights = {
            model_name: manual_weights[model_name]
            for model_name in public_safe_order
            if model_name in manual_weights
        }

        weight_sum = sum(ordered_weights.values())

        if weight_sum <= 0:
            raise ValueError("Public-safe weights are empty.")

        return {
            model_name: float(weight / weight_sum)
            for model_name, weight in ordered_weights.items()
        }

    def _weighted_blend(
        self,
        oof_predictions: pd.DataFrame,
        weights: Dict[str, float],
    ) -> np.ndarray:
        blend = np.zeros(oof_predictions.shape[0])

        for model_name, weight in weights.items():
            blend += weight * oof_predictions[model_name].values

        return blend

    def _apply_high_price_lift(
        self,
        log_predictions: np.ndarray,
        q_start: float = None,
        q_max: float = None,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Notebook Step 10 tail-lift logic.
        """
        adjusted = np.array(log_predictions, dtype=float).copy()

        if q_start is None:
            q_start = float(np.quantile(adjusted, self.tail_lift_start_quantile))

        if q_max is None:
            q_max = float(np.max(adjusted))

        if q_max <= q_start:
            return adjusted, q_start, q_max

        lift_ratio = (adjusted - q_start) / (q_max - q_start)
        lift_ratio = np.clip(lift_ratio, 0, 1)

        adjusted = adjusted + self.tail_lift_strength * lift_ratio

        return adjusted, q_start, q_max

    def _fit_models_on_full_data(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y_log: pd.Series,
    ) -> Dict[str, Any]:
        """
        Fit each base model on full transformed training data for deployment.
        """
        fitted_models = {}

        for model_name, model in models.items():
            logger.info(f"Fitting final full-data model: {model_name}")
            print(f"Fitting final full-data model: {model_name}")

            model_clone = clone(model)
            model_clone.fit(X, y_log)

            fitted_models[model_name] = model_clone

        return fitted_models

    def _to_json_safe(self, data: Any) -> Any:
        """
        Convert numpy/pandas values into JSON-safe Python values.
        """
        if isinstance(data, dict):
            return {str(k): self._to_json_safe(v) for k, v in data.items()}

        if isinstance(data, list):
            return [self._to_json_safe(v) for v in data]

        if isinstance(data, tuple):
            return [self._to_json_safe(v) for v in data]

        if isinstance(data, (np.integer,)):
            return int(data)

        if isinstance(data, (np.floating,)):
            return float(data)

        if isinstance(data, np.ndarray):
            return data.tolist()

        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient="records")

        if isinstance(data, pd.Series):
            return data.to_dict()

        return data

    def initiate_model_training(self) -> Dict[str, Any]:
        """
        Run final notebook-style model training.
        """
        try:
            logger.info("Model training started.")

            Path(self.config.root_dir).mkdir(parents=True, exist_ok=True)

            X, y_log, y_original = self._load_training_data()
            models = self._define_base_models()

            oof_predictions, model_scores_df = self._generate_oof_predictions(
                models=models,
                X=X,
                y_log=y_log,
            )

            selected_models = self._select_strong_models(model_scores_df)

            inverse_weights = self._calculate_inverse_weights(
                selected_models=selected_models,
                model_scores_df=model_scores_df,
            )

            public_safe_weights = self._calculate_public_safe_weights(
                available_model_names=list(models.keys()),
            )

            inverse_blend_oof_log = self._weighted_blend(
                oof_predictions=oof_predictions,
                weights=inverse_weights,
            )

            public_safe_oof_log = self._weighted_blend(
                oof_predictions=oof_predictions,
                weights=public_safe_weights,
            )

            final_blend_oof_log = (
                self.final_blend_weights["inverse"] * inverse_blend_oof_log
                + self.final_blend_weights["public_safe"] * public_safe_oof_log
            )

            final_blend_oof_log_tail, q_start, q_max = self._apply_high_price_lift(
                final_blend_oof_log
            )

            best_single_score = float(model_scores_df["OOF_RMSE_log1p"].min())
            inverse_blend_rmse = self._rmse_log(y_log, inverse_blend_oof_log)
            public_safe_rmse = self._rmse_log(y_log, public_safe_oof_log)
            final_blend_rmse = self._rmse_log(y_log, final_blend_oof_log)
            tail_lift_rmse = self._rmse_log(y_log, final_blend_oof_log_tail)

            fitted_models = self._fit_models_on_full_data(
                models=models,
                X=X,
                y_log=y_log,
            )

            lower_clip = float(y_original.quantile(0.001))
            upper_clip = float(y_original.quantile(0.999) * 1.05)

            tail_lift_params = {
                "enabled": True,
                "strength": self.tail_lift_strength,
                "start_quantile": self.tail_lift_start_quantile,
                "q_start": q_start,
                "q_max": q_max,
            }

            price_clip_params = {
                "enabled": True,
                "lower": lower_clip,
                "upper": upper_clip,
            }

            metadata = {
                "selected_strategy": "step10_tail_lift_blend",
                "target_column": self.target_column,
                "training_rows": int(X.shape[0]),
                "feature_count": int(X.shape[1]),
                "n_splits": self.n_splits,
                "seeds": self.seeds,
                "inverse_power": self.inverse_power,
                "selection_margin": self.selection_margin,
                "selected_inverse_models": selected_models,
                "all_base_models": list(models.keys()),
                "best_single_score": best_single_score,
                "inverse_blend_rmse": inverse_blend_rmse,
                "public_safe_rmse": public_safe_rmse,
                "final_conservative_blend_rmse": final_blend_rmse,
                "tail_lift_rmse": tail_lift_rmse,
                "model_scores": model_scores_df.to_dict(orient="records"),
                "note": (
                    "Model trainer productizes Model_training.ipynb Step 10. "
                    "No hyperparameter tuning and no submission generation are performed here."
                ),
            }

            ensemble_model = AmesEnsembleModel(
                models=fitted_models,
                inverse_weights=inverse_weights,
                public_safe_weights=public_safe_weights,
                final_blend_weights=self.final_blend_weights,
                tail_lift_params=tail_lift_params,
                price_clip_params=price_clip_params,
                metadata=metadata,
            )

            save_object(
                file_path=self.config.final_model_path,
                obj=ensemble_model,
            )

            blend_manifest = ensemble_model.get_manifest()

            save_json(
                file_path=self.config.blend_manifest_path,
                data=self._to_json_safe(blend_manifest),
            )

            save_json(
                file_path=self.config.model_metadata_path,
                data=self._to_json_safe(metadata),
            )

            output = {
                "model_training_completed": True,
                "final_model_path": self.config.final_model_path,
                "blend_manifest_path": self.config.blend_manifest_path,
                "model_metadata_path": self.config.model_metadata_path,
                "selected_strategy": "step10_tail_lift_blend",
                "feature_count": int(X.shape[1]),
                "training_rows": int(X.shape[0]),
                "best_single_score": best_single_score,
                "inverse_blend_rmse": inverse_blend_rmse,
                "public_safe_rmse": public_safe_rmse,
                "final_conservative_blend_rmse": final_blend_rmse,
                "tail_lift_rmse": tail_lift_rmse,
            }

            logger.info("Model training completed successfully.")
            return output

        except Exception as e:
            logger.error("Model training failed.")
            raise CustomException(e, sys)