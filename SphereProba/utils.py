"Utils functions"
import numpy as np


def normalize_rows(X):
    "Normalize each row of the 2D input array X"
    return X / np.linalg.norm(X, axis=1)[:, np.newaxis]
