import os
import pickle
import mlflow
from dotenv import load_dotenv

# 1. Force the script to read the .env file
load_dotenv()

# 2. Connect to DagsHub MLflow Server
mlflow.set_tracking_uri("https://dagshub.com/Thanderer/stp-customer-segmentation.mlflow")

# 3. Test the connection by logging a dummy run
with mlflow.start_run():
    mlflow.log_param("test_param", "hello_world")
    mlflow.log_metric("test_connection", 1.0)
    print("Successfully logged dummy metric to MLflow!")

# 4. Create the output directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# 5. Save a dummy model to satisfy DVC's output requirement
dummy_model = {"model_type": "KMeans", "k": 4}
with open("models/model.pkl", "wb") as f:
    pickle.dump(dummy_model, f)