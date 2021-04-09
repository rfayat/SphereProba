"""
A few Probability density functions on the unit sphere.

Author: Romain Fayat, April 2021
"""
import numpy as np
from scipy.special import gamma as gamma_func
from .utils import normalize_rows


def fit_vMF(X):
    "Fit a von Mises Fisher on 3D input data and return the MLE parameters"
    n_dim = 3
    n_points = len(X)
    X = normalize_rows(X)
    X_av = np.mean(X, axis=0)
    # Simple approximation (Sra, 2011)
    R = np.linalg.norm(X.sum(axis=0)) / n_points
    kappa_est = R * (n_dim - R**2) / (1 - R**2)

    return X_av / R, kappa_est


class VonMisesFisher():
    "3-dimensional von Mises-Misher distribution"

    def __init__(self, mu, kappa):
        self.mu = mu.reshape((-1, 1))
        self.kappa = kappa

    @property
    def C(self):
        "Normalization constant in for the S2 sphere"
        return self.kappa / (4 * np.pi * np.sinh(self.kappa))

    @classmethod
    def fit(cls, X):
        "Fit a vMF on input data and return an instance with fitted parameters"
        mu, kappa = fit_vMF(X)
        return cls(mu, kappa)

    def __call__(self, X):
        "Return the density for a range of 3-dimensional unit vectors"
        X = normalize_rows(X)
        return self.C * np.exp(self.kappa * X @ self.mu).flatten()

    def __repr__(self):
        return(f"""vMF distribution with parameters:
        μ = {self.mu.flatten()}
        κ = {self.kappa}""")


if __name__ == "__main__":
    dummy_data = np.array([[0, 0, 1.], [0, 0.01, 1.01]])
    print(VonMisesFisher.fit(dummy_data))
    dummy_data = np.random.random((100000, 3)) - np.array([[.5, .5, .5]])
    print(VonMisesFisher.fit(dummy_data))
