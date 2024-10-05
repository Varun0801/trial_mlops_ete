# hyperparameter_tuning.py
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

def tune_hyperparameters():
    try:
        X = pd.read_csv('iris_preprocessed.csv')
        y = pd.read_csv('iris_target.csv').values.ravel()
        
        # RandomForest Model Hyperparameter Tuning
        rf = RandomForestClassifier()
        rf_param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
        }
        
        rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=3, n_jobs=-1)
        rf_grid_search.fit(X, y)
        
        # SVM Model Hyperparameter Tuning
        svm = SVC(probability=True)
        svm_param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
        
        svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=3, n_jobs=-1)
        svm_grid_search.fit(X, y)
        
        # Compare both models
        if rf_grid_search.best_score_ > svm_grid_search.best_score_:
            best_model = rf_grid_search.best_estimator_
            print(f"Best model is RandomForest with params: {rf_grid_search.best_params_} and score: {rf_grid_search.best_score_}")
        else:
            best_model = svm_grid_search.best_estimator_
            print(f"Best model is SVM with params: {svm_grid_search.best_params_} and score: {svm_grid_search.best_score_}")
        
        # Save the best model
        joblib.dump(best_model, 'models/best_model.joblib')
        print("Best model saved!")
    except Exception as e:
        print(f"Error at hyperparameter tuning step: {e}")

if __name__ == '__main__':
    tune_hyperparameters()
