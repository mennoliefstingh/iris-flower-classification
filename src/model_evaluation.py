from sklearn.model_selection import cross_val_score
import numpy as np

def pprint_xval_results(clf, X:np.ndarray, y:np.ndarray) -> None:
    results = cross_val_score(clf, X, y)
    print("Cross validation results for ", type(clf))
    print("Total folds: ", len(results))
    print("Average score: ", round(np.mean(results), 3))
    print("Standard deviation: ", round(np.std(results), 3))