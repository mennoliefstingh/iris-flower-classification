from typing import Tuple
from sklearn.model_selection import cross_val_score
from src.model_evaluation import pprint_xval_results
import numpy as np

from src.data_loading import load_iris_data

X, y = load_iris_data()
print("iris data loaded")

from src.models.knn import get_untrained_knn_classifier

knn_clf = get_untrained_knn_classifier(X, y)

pprint_xval_results(knn_clf, X, y)

from src.models.random_forest import get_trained_rf_classifier, get_untrained_rf_classifier

rf_clf = get_untrained_rf_classifier(X, y)
pprint_xval_results(rf_clf, X, y)

