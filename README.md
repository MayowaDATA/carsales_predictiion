# 🚗 End-to-End Used Car Selling Price Prediction

An end-to-end Machine Learning web application that predicts used car market valuations based on vehicle specifications, historical usage, and mechanical features. The project includes data cleaning, exploratory data analysis (EDA), feature engineering pipelines, multiple regression models, hyperparameter optimization, and a Flask deployment interface.

---

## 📌 Project Overview

- **Problem:** Accurate estimation of used car market prices based on multiple categorical and continuous vehicle attributes[cite: 1].
- **Dataset:** Scraped vehicle listings from CarDekho containing 15,411 records and 13 initial attributes[cite: 1].
- **Best Model:** Random Forest Regressor (tuned with `RandomizedSearchCV`) achieving an $R^2$ score of **~94.2%** on test data[cite: 1].
- **Web App:** Flask backend serving a responsive glassmorphism dark-theme web user interface[cite: 2, 3].

---

## Web Application Demo

### Home Page
![Car Price Prediction Home](images/Screenshot%20(924).png)

### Prediction Interface
![Car Price Prediction Interface](images/Screenshot%20(925).png)

---

## 📊 Workflow & Methodology
Data Ingestion (SQL/CSV)
│
▼
EDA & Preprocessing (Handling missing values, Outliers, Feature Dropping)
│
▼
Transformation Pipeline (OneHotEncoder + StandardScaler + OrdinalEncoder)
│
▼
Model Exploration (Linear, Ridge, Lasso, KNN, Decision Tree, Random Forest, AdaBoost, GradientBoost, XGBoost)
│
▼
Hyperparameter Tuning (RandomizedSearchCV on Random Forest & XGBoost)
│
▼
Model & Preprocessor Serialization (Pickle / Joblib)
│
▼
Flask Web Application Deployment


---

## 📈 Model Performance Benchmark

| Model | Test MAE | Test RMSE | Test $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **Random Forest (Tuned)**[cite: 1] | **97,520.42**[cite: 1] | **208,883.45**[cite: 1] | **0.9420 (94.2%)**[cite: 1] |
| **Random Forest (Baseline)**[cite: 1] | 101,477.97[cite: 1] | 222,135.62[cite: 1] | 0.9345[cite: 1] |
| **Gradient Boosting**[cite: 1] | 126,393.96[cite: 1] | 255,516.71[cite: 1] | 0.9133[cite: 1] |
| **K-Neighbors Regressor**[cite: 1] | 110,757.94[cite: 1] | 285,107.61[cite: 1] | 0.8920[cite: 1] |
| **Decision Tree**[cite: 1] | 124,793.46[cite: 1] | 302,471.48[cite: 1] | 0.8785[cite: 1] |
| **XGBoost (Tuned)**[cite: 1] | 103,961.90[cite: 1] | 300,625.37[cite: 1] | 0.8799[cite: 1] |
| **Linear / Ridge / Lasso**[cite: 1] | 279,557.36[cite: 1] | 502,533.89[cite: 1] | 0.6645[cite: 1] |

---

## 🛠️ Project Structure

```text
CAR_SALESPREDICTION/
├── EDA/
│   ├── cardekho_imputated.csv
│   └── Xgboost Regression.ipynb
├── models/
│   ├── preprocessor.pkl
│   └── random_forest_model.pkl
├── templates/
│   ├── index.html
│   └── home.html
├── application.py
├── requirements.txt
├── .gitignore
└── README.md