# 🚗 End-to-End Used Car Selling Price Prediction & MLOps Pipeline

An end-to-end Machine Learning web application that predicts used car market valuations based on vehicle specifications, historical usage, and mechanical features. The project encompasses exploratory data analysis (EDA), automated preprocessing pipelines, hyperparameter-optimized regression models, Flask backend integration, and a production CI/CD cloud deployment on AWS.

🔗 **Live Deployment:** [http://carsalespricedemo-env-1.eba-uswsmwsk.eu-north-1.elasticbeanstalk.com/](http://carsalespricedemo-env-1.eba-uswsmwsk.eu-north-1.elasticbeanstalk.com/)

---

## 📌 Project Overview

* **Problem Statement:** Accurate estimation of used car resale prices based on continuous and categorical vehicle attributes.
* **Dataset:** 15,411 vehicle listing records from CarDekho across 13 core attributes.
* **Best Model:** Random Forest Regressor optimized via `RandomizedSearchCV`, achieving an **$R^2$ score of ~94.2%** on test data.
* **Cloud Infrastructure:** Hosted on AWS Elastic Beanstalk with automated CI/CD pipelines via AWS CodePipeline and model artifact storage in AWS S3.
* **Web App:** Lightweight Flask interface running on Gunicorn with on-demand S3 artifact caching.

---

## 🏗️ System Architecture & CI/CD Pipeline

```text
  +--------------------+         Git Push          +--------------------+
  |  GitHub Repository | ------------------------> |  AWS CodePipeline  |
  |  (Source Control)  |                           |  (Automated CI/CD) |
  +--------------------+                           +---------+----------+
                                                             |
                                                             v Deploy
+-------------------------+    Lazy-Load Artifacts   +----------------------+
|     AWS S3 Bucket       | -----------------------> | AWS Elastic Beanstalk|
| (carsalespredict .pkl)  |                          | (Python / Gunicorn)  |
+-------------------------+                          +----------------------+

Version Control: GitHub

CI/CD Automation: AWS CodePipeline

Compute / Hosting: AWS Elastic Beanstalk (Amazon Linux 2023 / Python 3.11 WSGI)

Model Storage: AWS S3 (eu-north-1 Stockholm region)

Web Server: Flask & Gunicorn

📸 Web Application & Cloud Pipeline Demo1. Web Application User ExperienceHome / Landing PagePrediction FormModel Valuation Output2. AWS Cloud Infrastructure & CI/CD PipelineAWS CodePipeline Automated CI/CDLive AWS Elastic Beanstalk Deployment📊 Workflow & MethodologyPlaintextData Ingestion (SQL / CSV)
         │
         ▼
EDA & Preprocessing (Missing value imputation, outlier handling, feature encoding)
         │
         ▼
Transformation Pipeline (OneHotEncoder + StandardScaler + ColumnTransformer)
         │
         ▼
Model Exploration (Linear, Ridge, Lasso, KNN, Decision Tree, Random Forest, AdaBoost, GradientBoost, XGBoost)
         │
         ▼
Hyperparameter Tuning (RandomizedSearchCV on Random Forest & XGBoost)
         │
         ▼
Model Artifact Storage (AWS S3) & In-Memory Lazy Loading
         │
         ▼
Flask Web App Deployment (AWS Elastic Beanstalk via AWS CodePipeline)
📈 Model Performance BenchmarkModelTest MAETest RMSETest R2 ScoreRandom Forest (Tuned)97,520.42208,883.450.9420 (94.2%)Random Forest (Baseline)101,477.97222,135.620.9345Gradient Boosting126,393.96255,516.710.9133K-Neighbors Regressor110,757.94285,107.610.8920Decision Tree124,793.46302,471.480.8785XGBoost (Tuned)103,961.90300,625.370.8799Linear / Ridge / Lasso279,557.36502,533.890.6645🛠️ Tech StackMachine Learning & Data Science: Scikit-Learn, Pandas, NumPy, XGBoostWeb Framework: Flask, Jinja2, HTML5, CSS3Cloud & MLOps: AWS Elastic Beanstalk, AWS S3, AWS CodePipeline, Boto3, Gunicorn, Git📂 Project StructurePlaintextCAR_SALESPREDICTION/
├── .ebextensions/
│   └── python.config          # Elastic Beanstalk WSGI configuration
├── EDA/
│   ├── cardekho_imputated.csv # Cleaned training dataset
│   └── Xgboost Regression.ipynb
├── images/
│   ├── aws_deployed_app.png   # Screenshot of live deployed application
│   ├── codepipeline.png       # Screenshot of successful CodePipeline build
│   ├── home.png               # Home landing page UI
│   ├── predict.png            # Prediction form UI
│   └── result.png             # Prediction output UI
├── templates/
│   ├── index.html             # Landing page
│   └── home.html              # Prediction form UI & output display
├── application.py             # Flask application with AWS S3 lazy loading
├── download_model.py          # Standalone S3 artifact retrieval script
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md                  # Project documentation