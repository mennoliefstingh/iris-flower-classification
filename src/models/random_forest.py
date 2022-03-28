from sklearn.ensemble import RandomForestClassifier
from src.models.helper_functions import find_optimal_parameters
import os.path
import json
import numpy as np

def _optimal_parameters_rf(filepath:str, X:np.ndarray, y:np.ndarray) -> dict:
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            return json.load(file)
    else:
        param_grid = {'n_estimators':np.arange(25, 200, 25).tolist(),
                      'max_depth': np.arange(5, 50, 5).tolist(),
                      'min_samples_split':[2, 4, 6, 8, 10],
                      'min_samples_leaf': [1, 2, 3, 4, 5]}
        best_params = find_optimal_parameters(RandomForestClassifier(), param_grid, X, y)
        with open('src/models/rf_params.json', 'w') as file:
            json.dump(best_params, file)
        return best_params

def get_untrained_rf_classifier(X, y):
    optimal_params = _optimal_parameters_rf('src/models/rf_params.json', X, y)
    return RandomForestClassifier(**optimal_params)

def get_trained_rf_classifier(X, y):
    optimal_params = _optimal_parameters_rf('src/models/rf_params.json', X, y)
    clf = RandomForestClassifier(**optimal_params)
    return clf.fit(X, y)