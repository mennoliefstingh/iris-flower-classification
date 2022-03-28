from typing import Tuple

import numpy as np
from sklearn import datasets


def load_iris_data() -> Tuple[np.ndarray, np.ndarray]:
    return datasets.load_iris(return_X_y=True)
