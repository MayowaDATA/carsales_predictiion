import os
import boto3
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from botocore.exceptions import ClientError

application = Flask(__name__)
app = application

S3_BUCKET = "carsalespredict"
S3_REGION = "eu-north-1"

s3 = boto3.client("s3", region_name=S3_REGION)

# Cache dictionary to store models in memory after the first load
model_cache = {
    "rf_model": None,
    "preprocessor": None
}

def load_model_and_preprocessor():
    """Lazily downloads and caches the model artifacts on the first prediction call."""
    if model_cache["rf_model"] is None or model_cache["preprocessor"] is None:
        os.makedirs("artifacts", exist_ok=True)
        
        rf_path = os.path.join("artifacts", "random_forest_model.pkl")
        prep_path = os.path.join("artifacts", "preprocessor.pkl")
        
        # Download random_forest_model.pkl if missing
        if not os.path.exists(rf_path) or os.path.getsize(rf_path) == 0:
            print(f"Downloading random_forest_model.pkl from S3 bucket {S3_BUCKET}...")
            try:
                s3.download_file(S3_BUCKET, "random_forest_model.pkl", rf_path)
                print("Downloaded random_forest_model.pkl successfully.")
            except ClientError as e:
                print(f"Error downloading random_forest_model.pkl: {e}")
                raise e
                
        # Download preprocessor.pkl if missing
        if not os.path.exists(prep_path) or os.path.getsize(prep_path) == 0:
            print(f"Downloading preprocessor.pkl from S3 bucket {S3_BUCKET}...")
            try:
                s3.download_file(S3_BUCKET, "preprocessor.pkl", prep_path)
                print("Downloaded preprocessor.pkl successfully.")
            except ClientError as e:
                print(f"Error downloading preprocessor.pkl: {e}")
                raise e
                
        # Load the artifacts into memory
        print("Loading models into memory...")
        with open(rf_path, "rb") as f:
            model_cache["rf_model"] = pickle.load(f)
        with open(prep_path, "rb") as f:
            model_cache["preprocessor"] = pickle.load(f)
        print("Models loaded into memory successfully.")
            
    return model_cache["rf_model"], model_cache["preprocessor"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        # Load artifacts on demand
        rf_model, preprocessor = load_model_and_preprocessor()

        # Collect form inputs
        model = request.form.get("model")
        vehicle_age = float(request.form.get("vehicle_age"))
        km_driven = float(request.form.get("km_driven"))
        seller_type = request.form.get("seller_type")
        fuel_type = request.form.get("fuel_type")
        transmission_type = request.form.get("transmission_type")
        mileage = float(request.form.get("mileage"))
        engine = float(request.form.get("engine"))
        max_power = float(request.form.get("max_power"))
        seats = float(request.form.get("seats"))

        # Construct dataframe matching model features
        new_data = pd.DataFrame({
            "model": [model],
            "vehicle_age": [vehicle_age],
            "km_driven": [km_driven],
            "seller_type": [seller_type],
            "fuel_type": [fuel_type],
            "transmission_type": [transmission_type],
            "mileage": [mileage],
            "engine": [engine],
            "max_power": [max_power],
            "seats": [seats]
        })

        # Preprocess and generate prediction
        new_data_processed = preprocessor.transform(new_data)
        prediction = rf_model.predict(new_data_processed)

        return render_template(
            "home.html",
            result=round(prediction[0], 2)
        )

    return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)