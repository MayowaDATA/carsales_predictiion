import boto3
import os

BUCKET_NAME = "car-price-models"

FILES = [
    "random_forest_model.pkl",
    "preprocessor.pkl"
]

LOCAL_FOLDER = "models"

os.makedirs(LOCAL_FOLDER, exist_ok=True)

s3 = boto3.client("s3")

for file in FILES:
    local_path = os.path.join(LOCAL_FOLDER, file)

    if not os.path.exists(local_path):
        s3.download_file(
            BUCKET_NAME,
            file,
            local_path
        )
        print(f"{file} downloaded successfully")
    else:
        print(f"{file} already exists")

print("All models are ready.")

