# predict_model.py
import pandas as pd
import joblib
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the static directory path
root_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and preprocessors
model_dir = os.path.join(root_dir, '../models')
model = joblib.load(f'{model_dir}/best_model.joblib')
scaler = joblib.load(f'{model_dir}/scaler.joblib')

def predict():
    try:
        X = pd.read_csv('data/test.csv')
        
        # Apply the same scaling and encoding as in training
        scaled_data = scaler.transform(X)
        
        # Make prediction
        predictions = model.predict(scaled_data)
        
        # Save predictions
        pd.DataFrame(predictions, columns=['predictions']).to_csv('data/submission.csv', index=False)
        
        logger.info("Predictions saved to submission.csv")
    except Exception as e:
        logger.info(f"Error at prediction step: {e}")

if __name__ == '__main__':
    predict()
