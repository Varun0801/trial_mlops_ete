# preprocessing.py
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import joblib
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data():
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(root_dir, '../models')

        data_train = pd.read_csv('data/train.csv')
        data_val = pd.read_csv('data/validation.csv')
        
        # Drop target column for feature scaling
        X_train = data_train.drop('target', axis=1)
        y_train = data_train['target']

        X_val = data_val.drop('target', axis=1)
        y_val = data_val['target']
        
        # Scaling the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Label encoding target column
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_val_encoded = le.transform(y_val)

        # Save preprocessors
        joblib.dump(scaler, f'{model_dir}/scaler.joblib')
        joblib.dump(le, f'{model_dir}/label_encoder.joblib')
        
        # Save to new csv files
        pd.DataFrame(X_train_scaled, columns=data_train.columns[:-1]).to_csv('data/train_preprocessed.csv', index=False)
        pd.DataFrame(y_train_encoded, columns=['target']).to_csv('data/train_target.csv', index=False)

        pd.DataFrame(X_val_scaled, columns=data_val.columns[:-1]).to_csv('data/validation_preprocessed.csv', index=False)
        pd.DataFrame(y_val_encoded, columns=['target']).to_csv('data/validation_target.csv', index=False)

        logger.info("Data preprocessing completed!")
    except Exception as e:
        logger.info(f"Error at preprocessing step: {e}")


if __name__ == '__main__':
    preprocess_data()
