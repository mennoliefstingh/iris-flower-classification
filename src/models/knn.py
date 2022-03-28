import json
import os.path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.models.helper_functions import find_optimal_parameters


def _optimal_parameters_knn(
    filepath: str, X: np.ndarray, y: np.ndarray, use_cached: bool = True
) -> dict:
    """
    Finds optimal parameters for the KNN classifier using a (pre-defined) grid search.

    Keyword arguments:
        filepath: The file location for cached parameters
        X: Training data
        y: target labels
        use_cached: True by default, set to fals if param search has to be redone

    Returns:
        A dictionary containing optimal parameters for this training set. Also saves to file.

    """
    if os.path.exists(filepath) and use_cached:
        with open(filepath, "r") as file:
            return json.load(file)
    else:
        param_grid = {"n_neighbors": np.arange(1, 10, 1).tolist()}
        best_params = find_optimal_parameters(KNeighborsClassifier(), param_grid, X, y)
        with open("src/models/knn_params.json", "w") as file:
            json.dump(best_params, file)
        return best_params


def get_untrained_knn_classifier(X, y):
    """
    Returns an untrained KNN classifier with optimal parameters found. If there are no
    cached optimal parameters, these will be found using the supplied X and y.
    IMPORTANT: To prevent leakage, do *not* pass the entire dataset as that will influence parameter selection

    Keyword arguments:
        X: Training data
        y: target labels
        use_cached: True by default, set to false if param search has to be redone

    Returns:
        An untrained KNN classifier with optimal parameters based on X/y sets.
    """
    optimal_params = _optimal_parameters_knn("src/models/knn_params.json", X, y)
    return KNeighborsClassifier(**optimal_params)


def get_trained_knn_classifier(X, y):
    """ "
    Returns a trained KNN classifier with optimal parameters found. If there are no
    cached optimal parameters, these will be found using the supplied X and y.

    Keyword arguments:
    X: data
    y: target labels
    use_cached: True by default, set to false if param search has to be redone

    Returns:
    A KNN classifier trained on
    """
    optimal_params = _optimal_parameters_knn("src/models/knn_params.json", X, y)
    clf = KNeighborsClassifier(**optimal_params)
    return clf.fit(X, y)
