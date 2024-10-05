# extract_data.py
from sklearn.datasets import load_iris
import pandas as pd

def extract_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df.to_csv('iris.csv', index=False)
    print("Data extracted to iris.csv")

if __name__ == '__main__':
    extract_data()
