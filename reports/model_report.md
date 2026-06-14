# Ames House Price Regression — Model Evaluation Report

## Final Selected Strategy

`step10_tail_lift_blend`

## Primary Validation Metric

The primary score is based on out-of-fold validation from the model trainer.

| Metric | Value |
|---|---:|
| Best Single Model OOF RMSE log1p | 0.108330 |
| Inverse Weighted Blend OOF RMSE log1p | 0.107347 |
| Public-Safe Blend OOF RMSE log1p | 0.107553 |
| Final Conservative Blend OOF RMSE log1p | 0.107373 |
| Tail-Lift Final Blend OOF RMSE log1p | 0.107328 |

## Final Model Summary

| Item | Value |
|---|---:|
| Training rows | 1458 |
| Feature count | 199 |
| Primary metric | OOF_RMSE_log1p |

## Selected Inverse-Blend Models

- KernelRidge
- ElasticNet
- Ridge
- Lasso
- CatBoost
- XGBoost
- LightGBM

## All Base Models Used

- ElasticNet
- Ridge
- Lasso
- SVR
- KernelRidge
- ExtraTrees
- CatBoost
- XGBoost
- LightGBM

## Diagnostic Training-Fit Metrics

These are diagnostic only because the final model is fitted on the full training data.

| Metric | Value |
|---|---:|
| Train-fit RMSE log1p | 0.073928 |
| Train-fit RMSE price | 13298.65 |
| Train-fit MAE price | 9061.18 |

## Notes

- No hyperparameter tuning was performed in the production trainer.
- Kaggle submission generation is intentionally kept outside model evaluation.
- The final production strategy is the notebook Step 10 tail-lift blend.
