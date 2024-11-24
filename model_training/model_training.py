# train_models.py
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_models():
    try:
        X = pd.read_csv('data/train_preprocessed.csv')
        y = pd.read_csv('data/train_target.csv').values.ravel()
        
        # SVM Model
        svc = SVC(probability=True)
        svc.fit(X, y)
        joblib.dump(svc, 'models/svc.joblib')
        
        # Random Forest Model
        rf = RandomForestClassifier()
        rf.fit(X, y)
        joblib.dump(rf, 'models/rf.joblib')
        
        logger.info("Models trained and saved!")
    except Exception as e:
        logger.info(f"Error at training step: {e}")

if __name__ == '__main__':
    train_models()
