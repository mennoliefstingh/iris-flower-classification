from sklearn.model_selection import GridSearchCV, cross_val_score

def find_optimal_parameters(clf, param_grid:dict, X, y) -> dict:
    gridsearch = GridSearchCV(clf, param_grid, n_jobs=4)
    gridsearch.fit(X,y)
    return gridsearch.best_params_


