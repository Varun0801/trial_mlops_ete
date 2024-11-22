import joblib
import pytest
import numpy as np

def test_model_loading():
    model = joblib.load('models/best_model.joblib')
    assert model is not None

def test_model_prediction():
    model = joblib.load('models/best_model.joblib')
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(sample_input)
    print(f"Prediction: {prediction}")
    assert len(prediction) == 1
