from typing import Tuple
from sklearn.model_selection import train_test_split
from src.model_evaluation import pprint_results
from src.data_loading import load_iris_data
from src.models.knn import get_trained_knn_classifier
from src.models.random_forest import get_trained_rf_classifier

X, y = load_iris_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

knn_clf = get_trained_knn_classifier(X_train, y_train)
rf_clf = get_trained_rf_classifier(X_train, y_train)

pprint_results(knn_clf, X_test, y_test)
pprint_results(rf_clf, X_test, y_test)

