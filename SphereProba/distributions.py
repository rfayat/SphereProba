"""
A few Probability density functions on the unit sphere.

Author: Romain Fayat, April 2021
"""
import numpy as np
from scipy.special import gamma as gamma_func
from . import utils
from .utils import store_value_on_first_computation


def _fit_vMF(X, weights):
    "Fit a von Mises Fisher on 3D input data and return the MLE parameters"
    n_dim = 3
    n_points = len(X)
    X = utils.normalize_rows(X)
    X_av = np.average(X, axis=0, weights=weights)
    # Simple approximation (Sra, 2011)
    R = np.linalg.norm(X_av)
    kappa_est = R * (n_dim - R**2) / (1 - R**2)

    return X_av / R, kappa_est


def _fit_kent(X):
    """Fit a Kent distribution on 3D input data and return the parameters

    We use here the formulas introduced in Kent 1982, which hold for kappa
    large versus beta.
    """
    # Precomputations
    n_dim = 3
    n_points = len(X)
    X = utils.normalize_rows(X)
    X_av = np.mean(X, axis=0)
    X_cov = np.cov(X.T)

    # Estimate Gamma
    _, theta, phi = utils.cart2polar(*X_av)
    theta_cos, theta_sin = np.cos(theta), np.sin(theta)
    phi_cos, phi_sin = np.cos(phi), np.sin(phi)
    H = np.array([[theta_cos,          -theta_sin,                  0],
                  [theta_sin * phi_cos, theta_cos * phi_cos, -phi_sin],
                  [theta_sin * phi_sin, theta_cos * phi_sin,  phi_cos]])
    B = H.T @ X_cov @ H
    psi = np.arctan(2 * B[1, 2] / (B[1, 1] - B[2, 2])) / 2
    K = np.array([[1,           0,            0],
                  [0, np.cos(psi), -np.sin(psi)],
                  [0, np.sin(psi),  np.cos(psi)]])

    gamma = (H @ K).T  # gamma1, gamma2, gamma3 as rows of gamma

    # Estimate kappa and beta
    T = gamma @ X_cov @ gamma.T
    r1 = np.linalg.norm(X_av)
    r2 = T[1, 1] - T[2, 2]
    kappa = 1 / (2 - 2 * r1 - r2) + 1 / (2 - 2 * r1 + r2)
    beta = (1 / (2 - 2 * r1 - r2) - 1 / (2 - 2 * r1 + r2)) / 2

    return gamma, kappa, beta

class VonMisesFisher():
    "3-dimensional von Mises-Misher distribution"

    def __init__(self, mu, kappa):
        self.mu = mu.reshape((-1, 1))
        self.kappa = kappa
        try:
            self._check_params()
        except AssertionError:
            raise ValueError("Invalid parameters\n" + self.__repr__())

    @property
    @store_value_on_first_computation
    def C(self):
        "Normalization constant for the S2 sphere"
        return self.kappa / (4 * np.pi * np.sinh(self.kappa))

    @classmethod
    def fit(cls, X, weights=None):
        "Fit a vMF on input data and return an instance with fitted parameters"
        if weights is None:
            weights = np.ones(len(X))
        mu, kappa = _fit_vMF(X, weights)
        return cls(mu, kappa)

    def _check_params(self):
        "Check that the parameters of the distribution are valid."
        assert self.kappa >= 0
        assert np.abs(np.linalg.norm(self.mu) - 1) < 1e-6

    def __call__(self, X):
        "Return the density for a range of 3-dimensional unit vectors"
        X = utils.normalize_rows(X)
        return self.C * np.exp(self.kappa * X @ self.mu).flatten()

    def __repr__(self):
        return(f"""vMF distribution with parameters:
        μ = {self.mu.flatten()}
        κ = {self.kappa}""")


class Kent():
    "3-dimensional Kent distribution"

    def __init__(self, gamma, kappa, beta):
        self.gamma = gamma
        self.kappa = kappa
        self.beta = beta
        try:
            self._check_params()
        except AssertionError:
            raise ValueError("Invalid parameters\n" + self.__repr__())

    @property
    @store_value_on_first_computation
    def C(self):
        "Normalization constant for the S2 sphere"
        # Approximation of the normalization constant (Kent 1982)
        C_numerator = 2 * np.pi * np.exp(self.kappa)
        C_denominator = np.sqrt(self.kappa**2 - 4 * self.beta**2)
        return C_numerator / C_denominator

    @classmethod
    def fit(cls, X):
        "Fit the distribution on input data and return an instance"
        gamma, kappa, beta = _fit_kent(X)
        return cls(gamma, kappa, beta)

    def _check_params(self):
        "Check that the parameters of the distribution are valid."
        assert self.kappa >= 0
        assert self.beta >= 0
        assert self.kappa > 2 * self.beta
        # Check that gamma is orthogonal
        np.testing.assert_allclose(
             self.gamma.T @ self.gamma, np.eye(3), atol=1e-6
        )

    def __call__(self, X):
        "Return the density for a range of 3-dimensional unit vectors"
        gamma1, gamma2, gamma3 = self.gamma[:, :, np.newaxis]
        exponent = self.kappa * X @ gamma1 +\
                   self.beta * (X @ gamma2)**2 -\
                   self.beta * (X @ gamma3)**2
        return np.exp(exponent.flatten()) / self.C

    def __repr__(self):
        return(f"""Kent distribution with parameters:
        γ1 = {self.gamma[0]}
        γ2 = {self.gamma[1]}
        γ3 = {self.gamma[2]}
        κ = {self.kappa}
        β = {self.beta}""")


if __name__ == "__main__":
    random_seed = np.random.RandomState(42)
    print("Examples for Von Mises Fisher distribution\n" + "-" * 40)
    dummy_data = np.array([[0, 0, 1.], [0, 0.01, 1.01]])
    print(VonMisesFisher.fit(dummy_data))
    dummy_data = random_seed.random((100000, 3)) - np.array([[.5, .5, .5]])
    print(VonMisesFisher.fit(dummy_data))
    dummy_data = np.array([[0, 0, 1.], [0, 0, -1]])
    print(VonMisesFisher.fit(dummy_data, weights=np.array([1e3, 1])))

    print("\nTesting Kent distribution\n" + "-" * 40)
    dummy_data = random_seed.random((10000, 3)) - np.array([[.5, .3, .1]])
    print(Kent.fit(dummy_data))
