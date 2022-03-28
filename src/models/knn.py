from sklearn.neighbors import KNeighborsClassifier
from src.models.helper_functions import find_optimal_parameters
import os.path
import json
import numpy as np

def _optimal_parameters_knn(filepath:str, X:np.ndarray, y:np.ndarray) -> dict:
    print("looking for optimal params")
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            return json.load(file)
    else:
        print("no param file found")
        param_grid = {'n_neighbors': np.arange(1, 10, 1).tolist()}
        best_params = find_optimal_parameters(KNeighborsClassifier(), param_grid, X, y)
        with open('src/models/knn_params.json', 'w') as file:
            json.dump(best_params, file)
        return best_params

def get_untrained_knn_classifier(X, y):
    optimal_params = _optimal_parameters_knn('src/models/knn_params.json', X, y)
    return KNeighborsClassifier(**optimal_params)

def get_trained_knn_classifier(X, y):
    optimal_params = _optimal_parameters_knn('src/models/knn_params.json', X, y)
    clf = KNeighborsClassifier(**optimal_params)
    return clf.fit(X, y)