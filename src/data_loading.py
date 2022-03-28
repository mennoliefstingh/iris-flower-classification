import numpy as np
from sklearn import datasets
from typing import Tuple

def load_iris_data() -> Tuple[np.ndarray, np.ndarray]:
    return datasets.load_iris(return_X_y=True)