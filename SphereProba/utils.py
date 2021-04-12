"Utils functions"
import numpy as np
from functools import wraps


def normalize_rows(X):
    "Normalize each row of the 2D input array X"
    return X / np.linalg.norm(X, axis=1)[:, np.newaxis]


def store_value_on_first_computation(f):
    "Compute and store the result of a method its first call"

    @wraps(f)
    def g(self, *args, **kwargs):
        "Store the value of f on the first call and return it"
        method_name = f.__name__
        stored_result_name = "_" + method_name

        if getattr(self, stored_result_name, None) is None:
            setattr(self, stored_result_name, f(self, *args, **kwargs))

        return getattr(self, stored_result_name)

    return g


def cart2polar(x, y, z):
    "Convert cartesian coordinates to polar coordinates"
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(x / r)
    phi = np.arctan2(z, y)
    return r, theta, phi


def sorted_eig(X):
    "Return the sorted eigenvalues (based on real part) and eigenvectors of X"
    eigvalues, eigvectors = np.linalg.eig(X)
    order = np.argsort(np.real(eigvalues))[::-1]
    return eigvalues[order], eigvectors[:, order]
