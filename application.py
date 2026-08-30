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

def download_if_missing(filename):
    os.makedirs("artifacts", exist_ok=True)
    local_path = os.path.join("artifacts", filename)
    
    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        print(f"Downloading {filename} from S3 bucket {S3_BUCKET} in {S3_REGION}...")
        try:
            s3.download_file(S3_BUCKET, filename, local_path)
            print(f"Successfully downloaded {filename}")
        except ClientError as e:
            print(f"Failed to download {filename} from S3: {e}")
            raise e
    return local_path

# 1. Download paths
rf_model_path = download_if_missing("random_forest_model.pkl")
preprocessor_path = download_if_missing("preprocessor.pkl")

# 2. Load model and preprocessor into memory
with open(rf_model_path, "rb") as f:
    rf_model = pickle.load(f)

with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)


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
    app.run(host="0.0.0.0", port=5000)