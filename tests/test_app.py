import pytest
from fastapi.testclient import TestClient
from fastapi_deployment.app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_predict():
    payload = {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
    }
    response = client.post("/predict", json=payload)
    print(response)
    assert response.status_code == 200
    assert "prediction" in response.json()
