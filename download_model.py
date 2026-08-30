import os
import boto3

BUCKET_NAME = "carsalespredict"
REGION_NAME = "eu-north-1"

FILES = [
    "random_forest_model.pkl",
    "preprocessor.pkl"
]

LOCAL_FOLDER = "artifacts"

os.makedirs(LOCAL_FOLDER, exist_ok=True)

s3 = boto3.client("s3", region_name=REGION_NAME)

for file in FILES:
    local_path = os.path.join(LOCAL_FOLDER, file)

    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        print(f"Downloading {file} from S3 bucket {BUCKET_NAME}...")
        s3.download_file(
            BUCKET_NAME,
            file,
            local_path
        )
        print(f"{file} downloaded successfully")
    else:
        print(f"{file} already exists")

print("All models are ready.")