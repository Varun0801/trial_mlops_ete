# predict_model.py
import pandas as pd
import joblib

def predict():
    try:
        X = pd.read_csv('iris_preprocessed.csv')
        
        # Load the best model
        model = joblib.load('models/best_model.joblib')
        
        predictions = model.predict(X)
        
        # Save predictions
        pd.DataFrame(predictions, columns=['predictions']).to_csv('submission.csv', index=False)
        
        print("Predictions saved to submission.csv")
    except Exception as e:
        print(f"Error at prediction step: {e}")

if __name__ == '__main__':
    predict()
