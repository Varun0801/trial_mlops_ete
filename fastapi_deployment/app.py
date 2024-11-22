# app.py
import logging
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
import joblib
import pandas as pd
import os

app = FastAPI()

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code}")
    return response

# Set the static directory path
root_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(root_dir, '../static')

# Mount the directory where the index.html is located
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Load model and preprocessors
model_dir = os.path.join(root_dir, '../models')
model = joblib.load(f'{model_dir}/best_model.joblib')
scaler = joblib.load(f'{model_dir}/scaler.joblib')
label_encoder = joblib.load(f'{model_dir}/label_encoder.joblib')

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, 'index.html'))


@app.post('/predict')
async def predict(request: Request):
    try:
        data = await request.json()
        print(f"input_data: {data}")
        # Process the key, value pairs coming from API request
        if 'data' not in data.keys():
            data = {'data': data.values()}
        data_df = pd.DataFrame([data['data']], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
        print(f"data_df: {data_df}")
        # Apply the same scaling and encoding as in training
        scaled_data = scaler.transform(data_df)
        print(scaled_data)
        # Make prediction
        prediction = model.predict(scaled_data)
        # Convert label back to original form if necessary
        original_label = label_encoder.inverse_transform(prediction)
        logging.info(f"Predicted class: {prediction}")
        return {"prediction": int(original_label[0])}
    
    except Exception as e:
        print(f"An error occurred during prediction from endpoint: {e}")


# Run this with: uvicorn fastapi_deployment.app:app --host 0.0.0.0 --port 8000 --reload

