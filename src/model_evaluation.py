from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
import numpy as np

def pprint_results(clf, X_test:np.ndarray, y_test:np.ndarray) -> None:
    y_pred = clf.predict(X_test)

    print("Results for ", type(clf).__name__)
    print("Classification accuracy: ", round(accuracy_score(y_test, y_pred), 3))
    print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))