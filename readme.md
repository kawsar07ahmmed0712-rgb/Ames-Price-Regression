# House Price Prediction Regression

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Project Overview

This project is an end-to-end machine learning regression project focused on predicting house sale prices using structured housing data.

The main goal of this project is to build a reliable regression model that can learn from property-related features such as location, area, quality, condition, basement information, garage information, year-built details, and other housing attributes to estimate the final sale price.

This project includes:

- Exploratory Data Analysis
- Data cleaning
- Missing value handling
- Feature engineering
- Outlier analysis
- Model training
- Cross-validation
- Model comparison
- Weighted blending
- Stacking / meta-modeling
- Final prediction generation

---

## Problem Statement

House prices are affected by many factors, including property size, neighborhood, building quality, age, garage capacity, basement condition, and overall living area.

The objective of this project is to predict the `SalePrice` of houses using the available numerical and categorical features.

This is a supervised machine learning regression problem.

---

## Dataset

The dataset contains house-level information with multiple numerical and categorical features.

Main files used in this project:

```text
Data/
├── train.csv
├── test.csv
├── sample_submission.csv
└── data_description.txt
```

The target variable is:

```text
SalePrice
```

---

## Project Workflow

```text
Raw Data
   ↓
Exploratory Data Analysis
   ↓
Missing Value Analysis
   ↓
Outlier Detection
   ↓
Feature Engineering
   ↓
Feature Transformation
   ↓
Model Training
   ↓
Cross Validation
   ↓
Model Comparison
   ↓
Weighted Blending / Stacking
   ↓
Final Prediction
```

---

## Exploratory Data Analysis

The EDA stage was used to understand the structure of the dataset and identify important patterns before modeling.

Main EDA tasks:

- Checked dataset shape and feature types
- Analyzed numerical and categorical variables
- Studied target variable distribution
- Checked missing values
- Analyzed relationships between features and `SalePrice`
- Reviewed outliers
- Investigated important property-related features

Important observations:

- `SalePrice` is right-skewed, so log transformation is useful.
- Overall quality, living area, garage capacity, basement size, and neighborhood-related features have strong relationships with house price.
- Some missing values represent the absence of a feature, such as no basement, no garage, no pool, or no alley access.
- Outliers need careful handling because some extreme values can strongly affect regression models.

---

## Feature Engineering

Feature engineering was one of the most important parts of this project.

Main feature engineering steps:

- Handled missing values based on feature meaning
- Converted some missing categorical values into `"None"`
- Applied target transformation using `log1p(SalePrice)`
- Created meaningful property-level features
- Created total area and quality-related features
- Created age-related features
- Created binary indicator features
- Handled skewed numerical features
- Prepared train and test data with matching feature structure

Examples of engineered feature ideas:

```text
TotalSF
TotalBathrooms
HouseAge
RemodAge
GarageAge
TotalPorchSF
HasGarage
HasBasement
HasPool
HasFireplace
OverallQualityScore
```

---

## Models Used

Multiple regression models were trained and compared.

Models include:

- Ridge Regression
- ElasticNet
- Bayesian Ridge
- Support Vector Regression
- CatBoost Regressor
- Weighted Blending
- Stacking / Meta Model

---

## Model Evaluation

The main evaluation metric used in this project is:

```text
OOF Log RMSE
```

OOF means out-of-fold validation. It gives a more reliable estimate of model performance compared to a single train-test split.

---

## Model Comparison

| Model | Type | OOF Log RMSE |
|---|---:|---:|
| Master Weighted Blend | Weighted Blend | 0.10716 |
| ElasticNet Robust Tuned | Single Model | 0.10992 |
| ElasticNet RobustScaler Baseline | Single Model | 0.11104 |
| SVR RBF RobustScaler Master | Single Model | 0.11109 |
| Ridge RobustScaler Alpha10 | Single Model | 0.11214 |
| Bayesian Ridge RobustScaler | Single Model | 0.11217 |
| Ridge RobustScaler Baseline | Single Model | 0.11232 |
| Ridge RobustScaler Alpha30 | Single Model | 0.11270 |
| CatBoost Master V2 | Single Model | 0.11386 |

Best weighted blend result:

```text
OOF Log RMSE: 0.10716
```

Best meta model result:

```text
OOF Log RMSE: 0.10637
```

---

## Final Ensemble

The final ensemble used a weighted blend of selected models.

Selected models:

```text
Ridge_RobustScaler_baseline
ElasticNet_robust_tuned
ElasticNet_RobustScaler_baseline
SVR_RBF_RobustScaler_master
Ridge_RobustScaler_alpha10_master
BayesianRidge_RobustScaler_master
Ridge_RobustScaler_alpha30_master
CatBoost_master_v2
```

The ensemble improved performance compared to individual models by combining the strengths of different regression algorithms.

---

## Repository Structure

```text
House-Price-Prediction-Regression/
│
├── Data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── data_description.txt
│
├── Research/
│   ├── EDA_House_Price_clean_structured.ipynb
│   ├── Feature_Engeneering.ipynb
│   ├── Model_Training.ipynb
│   ├── bivariate/
│   ├── multivariate/
│   ├── univariate/
│   └── report/
│
├── fe_report/
├── fe_report_v2/
├── fe_result/
├── fe_result_v2/
├── model_report/
├── model_result/
│
├── Exploratory_data_analysis.md
├── Feature_Engeneering.md
├── FE_v2_problem_solution_feature_map.md
├── advanced_feature_engineering_route_house_price (1).md
├── metadata.yaml
├── LICENSE
└── README.md
```

---

## Key Results

This project achieved strong regression performance through advanced feature engineering and ensemble modeling.

Main results:

- Best single model: ElasticNet Robust Tuned
- Best single model OOF Log RMSE: `0.10992`
- Best weighted blend OOF Log RMSE: `0.10716`
- Best meta model OOF Log RMSE: `0.10637`

The weighted blend and meta-modeling approach improved performance over individual models.

---

## How to Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/kawsar07ahmmed0712-rgb/House-Price-Prediction-Regression.git
cd House-Price-Prediction-Regression
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment.

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install required libraries

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

If not, install the main libraries manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn catboost scipy jupyter
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook
```

Then run the notebooks from the `Research/` folder in this order:

```text
1. EDA_House_Price_clean_structured.ipynb
2. Feature_Engeneering.ipynb
3. Model_Training.ipynb
```

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- CatBoost
- Jupyter Notebook

---

## Skills Demonstrated

This project demonstrates the following machine learning skills:

- Regression modeling
- Exploratory data analysis
- Missing value treatment
- Feature engineering
- Outlier handling
- Categorical encoding
- Numerical transformation
- Cross-validation
- Model comparison
- Ensemble learning
- Weighted blending
- Stacking / meta-modeling
- Kaggle-style submission generation

---

## Future Improvements

Possible improvements for this project:

- Add a clean `requirements.txt` file
- Convert notebook code into reusable Python scripts
- Add a `src/` folder for modular pipeline code
- Add model saving using `joblib` or `pickle`
- Build a simple Streamlit web app for house price prediction
- Add SHAP-based model explainability
- Add automated training and prediction scripts
- Clean old notebooks and keep only final versions
- Add final visual summary images to the README

---

## Suggested Clean Project Structure

For a more production-ready version, this project can be reorganized as:

```text
House-Price-Prediction-Regression/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── data_description.txt
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── reports/
│   ├── figures/
│   └── model_results.csv
│
├── models/
│   └── final_model.pkl
│
└── outputs/
    └── final_submission.csv
```

---

## Conclusion

This project shows a complete machine learning workflow for house price prediction. Starting from raw housing data, the project performs EDA, feature engineering, model training, cross-validation, and ensemble modeling.

The best performance was achieved using ensemble methods, especially weighted blending and meta-modeling. The results show that combining multiple models can improve prediction accuracy compared to relying on a single model.

---

## Author

**Kawsar Ahmmed**

GitHub: [kawsar07ahmmed0712-rgb](https://github.com/kawsar07ahmmed0712-rgb)

---

## License

This project is licensed under the MIT License.