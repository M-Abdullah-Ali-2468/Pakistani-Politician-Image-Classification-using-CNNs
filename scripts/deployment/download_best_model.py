import os
import mlflow
import dagshub

REPO_OWNER = "f230046"
REPO_NAME = "Pakistani-Politician-Image-Classification-using-CNNs"
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
REGISTERED_MODEL_NAME = "ResNet-50" 

print(f"Connecting to DagsHub to fetch '{REGISTERED_MODEL_NAME}'...")

try:
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{REGISTERED_MODEL_NAME}/latest",
        dst_path="./model"
    )
    
    print(f"✅ Success! Your model package is downloaded and ready.")
    print(f"📁 Location: {local_path}")
    
except Exception as e:
    print(f"❌ Error downloading model: {e}")