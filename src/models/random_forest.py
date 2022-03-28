import json
import os.path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.helper_functions import find_optimal_parameters


def _optimal_parameters_rf(
    filepath: str, X: np.ndarray, y: np.ndarray, use_cached: bool = True
) -> dict:
    """
    Finds optimal parameters for the random forest classifier using a (pre-defined) grid search.

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
        param_grid = {
            "n_estimators": np.arange(25, 200, 25).tolist(),
            "max_depth": np.arange(5, 50, 5).tolist(),
            "min_samples_split": [2, 4, 6, 8, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
        }
        best_params = find_optimal_parameters(
            RandomForestClassifier(), param_grid, X, y
        )
        with open("src/models/rf_params.json", "w") as file:
            json.dump(best_params, file)
        return best_params


def get_untrained_rf_classifier(
    X: np.ndarray, y: np.ndarray, use_cached: bool = True
) -> RandomForestClassifier:
    """
    Returns an untrained random forest classifier with optimal parameters found. If there are no
    cached optimal parameters, these will be found using the supplied X and y.
    IMPORTANT: To prevent leakage, do *not* pass the entire dataset as that will influence parameter selection

    Keyword arguments:
        X: Training data
        y: target labels
        use_cached: True by default, set to false if param search has to be redone

    Returns:
        An untrained random forest classifier with optimal parameters based on X/y sets.
    """
    optimal_params = _optimal_parameters_rf(
        "src/models/rf_params.json", X, y, use_cached
    )
    return RandomForestClassifier(**optimal_params)


def get_trained_rf_classifier(
    X: np.ndarray, y: np.ndarray, use_cached: bool = True
) -> RandomForestClassifier:
    """ "
    Returns a trained random forest classifier with optimal parameters found. If there are no
    cached optimal parameters, these will be found using the supplied X and y.

    Keyword arguments:
    X: data
    y: target labels
    use_cached: True by default, set to false if param search has to be redone

    Returns:
    A random forest classifier trained on
    """
    optimal_params = _optimal_parameters_rf(
        "src/models/rf_params.json", X, y, use_cached
    )
    clf = RandomForestClassifier(**optimal_params)
    return clf.fit(X, y)
