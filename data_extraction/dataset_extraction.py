# extract_data.py
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df.to_csv('data/iris.csv', index=False)
    logger.info("Data extracted to iris.csv")

def make_dataset(complete_data_path):
    data = pd.read_csv(complete_data_path)

    # Split into train (70%), validation (15%), and test (15%)
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    test_data_target_labels = test_data['target']
    test_data = test_data.drop('target', axis=1)
    
    # Save splits
    train_data.to_csv("data/train.csv", index=False)
    val_data.to_csv("data/validation.csv", index=False)
    test_data.to_csv("data/test.csv", index=False)
    test_data_target_labels.to_csv("data/test_target.csv", index=False)

    logger.info(f"Train: {train_data.shape}, Validation: {val_data.shape}, Test: {test_data.shape}")

if __name__ == '__main__':
    extract_data()
    make_dataset('data/iris.csv')
