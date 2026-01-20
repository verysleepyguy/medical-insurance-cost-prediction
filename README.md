# Predicting Medical Insurance Costs

## Project Overview
This project predicts annual medical insurance charges using patient demographic and health-related data. The objective is to build an interpretable regression model while demonstrating strong data science fundamentals, including exploratory data analysis (EDA), preprocessing, and model evaluation.

The project emphasizes reproducibility, clean modeling pipelines, and clear communication of results, making it suitable for internship-level data science and health analytics roles.

---


## Dataset
- **Source:** Kaggle – Medical Cost Personal Dataset  
- **Observations:** 1,338 individuals  
- **Target Variable:** `charges`  
- **Features:**
  - `age` – Age of the primary beneficiary  
  - `sex` – Gender  
  - `bmi` – Body Mass Index  
  - `children` – Number of dependents  
  - `smoker` – Smoking status  
  - `region` – U.S. residential region  

---

## Objective
To accurately predict medical insurance costs and identify the most influential factors affecting charges using regression-based machine learning models.

---

## Data Cleaning & Preprocessing
The dataset required minimal structural cleaning but substantial preprocessing for modeling:

- Verified absence of missing and duplicate values  
- One-hot encoded categorical variables (`sex`, `smoker`, `region`)  
- Scaled numerical features using standardization  
- Log-transformed the target variable (`charges`) to address right-skew and heteroscedasticity  
- Implemented scikit-learn `Pipeline` and `ColumnTransformer` to prevent data leakage  

---

## Exploratory Data Analysis
Key insights from EDA include:
- Medical charges are heavily right-skewed  
- Smoking status is the strongest predictor of medical costs  
- BMI and age show positive relationships with charges  
- Regional effects exist but are relatively minor  

All visualizations and observations are documented in the notebook.

---

## Modeling Approach

### Models Implemented
- Linear Regression (baseline)  
- Decision Tree Regressor  
- Random Forest Regressor  

### Evaluation Metrics
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

---

## Model Performance
| Model | R² Score |
|------|----------|
| Linear Regression | ~0.75 |
| Decision Tree | ~0.82 |
| Random Forest | ~0.86 |

Random Forest achieved the best performance by capturing non-linear interactions between demographic and health features.

---

## Key Insights
- Smoking dramatically increases predicted medical costs  
- Tree-based models outperform linear models for this problem  
- Log-transforming the target variable improves model stability and accuracy  

---

## Future Improvements
- Hyperparameter tuning using GridSearchCV  
- Cross-validation for more robust evaluation  
- Model explainability using SHAP  
- Exploration of gradient boosting models (XGBoost, LightGBM)  

---

## Tools & Libraries
- Python  
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn  

---

## How to Run the Project
```bash
pip install -r requirements.txt
