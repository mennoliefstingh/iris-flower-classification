from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np

X, y = datasets.load_iris(return_X_y=True)

clf = RandomForestClassifier()

param_grid = {'n_estimators':np.arange(25, 200, 25).tolist(),
             'max_depth': np.arange(5, 50, 5).tolist(),
             'min_samples_split':[2, 4, 6, 8, 10],
             'min_samples_leaf': [1, 2, 3, 4, 5]}

def find_optimal_parameters(clf, param_grid:dict, X, y) -> dict:
    gridsearch = GridSearchCV(clf, param_grid, n_jobs=6)
    gridsearch.fit(X,y)
    return gridsearch.best_params_

best_params = find_optimal_parameters(clf, param_grid, X, y)

print('training and testing classifier')

clf = RandomForestClassifier()
clf.set_params(**best_params)
print(np.mean(cross_val_score(clf, X, y)))


