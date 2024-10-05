# preprocessing.py
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import joblib

def preprocess_data():
    try:
        df = pd.read_csv('iris.csv')
        
        # Drop target column for feature scaling
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Scaling the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Label encoding target column
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Save preprocessors
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(le, 'label_encoder.joblib')
        
        pd.DataFrame(X_scaled, columns=df.columns[:-1]).to_csv('iris_preprocessed.csv', index=False)
        pd.DataFrame(y_encoded, columns=['target']).to_csv('iris_target.csv', index=False)
        
        print("Data preprocessing completed!")
    except Exception as e:
        print(f"Error at preprocessing step: {e}")


if __name__ == '__main__':
    preprocess_data()