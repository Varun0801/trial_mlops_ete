# app.py
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import pandas as pd
import os

app = FastAPI()

# Set the static directory path
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../static')

# Mount the directory where the index.html is located
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Load model and preprocessors
model = joblib.load(r'../models/best_model.joblib')
scaler = joblib.load(r'../scaler.joblib')
label_encoder = joblib.load(r'../label_encoder.joblib')

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, 'index.html'))


@app.post('/predict')
async def predict(request: Request):
    try:
        data = await request.json()
        data_df = pd.DataFrame([data['data']], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
        
        # Apply the same scaling and encoding as in training
        scaled_data = scaler.transform(data_df)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        
        # Convert label back to original form if necessary
        original_label = label_encoder.inverse_transform(prediction)
        
        return {"prediction": int(original_label[0])}
    
    except Exception as e:
        print(f"An error occurred during prediction from endpoint: {e}")


# Run this with: uvicorn app:app --reload
