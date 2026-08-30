import os
import boto3
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# S3 Configuration
S3_BUCKET = "carsalesprojectmodels"  # verify this matches your S3 bucket name exactly
S3_REGION = "eu-north-1"

s3 = boto3.client("s3", region_name=S3_REGION)

def download_if_missing(filename):
    local_path = os.path.join("models", filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from S3...")
        s3.download_file(S3_BUCKET, filename, local_path)
    return local_path

# Auto-download from S3 if files are not present locally
rf_model_path = download_if_missing("random_forest_model.pkl")
preprocessor_path = download_if_missing("preprocessor.pkl")

# Load models
rf_model = pickle.load(open(rf_model_path, "rb"))
preprocessor = pickle.load(open(preprocessor_path, "rb"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        # Get data from form
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

        # Create dataframe
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

        # Preprocess the data
        new_data_processed = preprocessor.transform(new_data)

        # Predict
        prediction = rf_model.predict(new_data_processed)

        return render_template(
            "home.html",
            result=round(prediction[0], 2)
        )

    return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)