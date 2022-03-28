import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def pprint_results(clf, X_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred = clf.predict(X_test)

    print("Results for ", type(clf).__name__)
    print("Classification accuracy: ", round(accuracy_score(y_test, y_pred), 3))
    print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))
