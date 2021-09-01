# SphereProba

## Installation
Clone the repository and use pip to install the package.
```bash
git clone https://github.com/rfayat/SphereProba.git
cd SphereProba
pip install .
```
## Generating the example figures
Code for generating the figures shown here can be found in the [generate_examples.py](examples/generate_examples.py) script. To generate the example figures, install seaborn and [angle_visualization](https://github.com/rfayat/angle_visualization) and run:

```bash
python -m examples.generate_examples
```

## von Mises-Fisher distribution on the unit sphere
The [von Mises-Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution) in dimension 3 (or Fisher distribution) is an isotropic probability distribution on the unit sphere. It is defined by a mean direction `mu` and a concentration parameter `kappa` (respectively analogue to the mean and invert variance of a Gaussian).

`SphereProba.distributions.VonMisesFisher` allows to fit such a distribution on an input set of 3-dimensional arrays (which are normalized if needed). `VonMisesFisher` objects support a `__call__` method returning the value of the probability distribution function of input vectors.

### Initialization
A `VonMisesFisher` object can be created by providing an array of length 3 representing its mean direction `mu` and a strictly positive float `kappa` corresponding to its concentration parameter:

```python
>>> import numpy as np
>>> from SphereProba.distributions import VonMisesFisher
>>> vmf = VonMisesFisher(np.array([0, 0, 1]), 10.)
>>> print(vmf)
vMF distribution with parameters:
        μ = [0 0 1]
        κ = 10.0
```

Alternatively, it can be fitted on a 2-dimensional array of shape `(n_arrays, 3)` containing the x, y and z coordinates of a set of 3-dimensional vectors:

```python
>>> dummy_data = np.array([[0, 0, 1.], [0, 0.01, 1.01]])
>>> vmf = VonMisesFisher.fit(dummy_data)
>>> print(vmf)
vMF distribution with parameters:
        μ = [0.         0.00495031 0.99998775]
        κ = 81613.99993282309
```
### Example for different values of kappa
A visualization of the impact of the concentration parameter kappa on the shape of the distribution is shown below:

![vmf example](examples/vmf.png)

### Converting kappa to an equivalent angle
Taking advantage of the isotropy of the von Mises-Fisher distribution (cylindrical symmetry around `mu`), we can convert the concentration parameter `kappa` in an equivalent angle `theta` such that integrating over the spherical cap centered on `mu` and having polar angle `theta` yields a user-defined value.

For instance, the angle `theta` for which 95% of the distribution is contained within angle `theta` of `mu` can be computed as follows:

```python
>>> vmf = VonMisesFisher(np.array([0, 0, 1]), 10.)
>>> vmf.kappa_to_thetamax(.95)  # result in degrees by default
45.53874561998862
```

The proof for the explicit formula for computing the value of `theta` given both `kappa` and the expected value of the integral on the corresponding spherical cap is defined in [this short pdf](ressources/vmf_integration.pdf).

## Kent distribution on the unit sphere
The Kent distribution (or 5-parameter Fisher-Bingham distribution) extends the von Mises-Fisher by making it anisotropic (ellipsoid on the sphere). It has parameters `gamma`, an orthonormal matrix whose components define the mean direction of the distribution (`gamma1`) and directions of major and minor axes of the ellipsoid (respectively `gamma2` and `gamma3`).

Similarly to the von Mises-Fisher distribution, a parameter `kappa`rules the concentration of the distribution while `beta` defines the extent to which the ellipsoid is stretched.

### Fit from a range of 3D vectors
Similarly to `SphereProba.distributions.VonMisesFisher`, `SphereProba.distributions.Kent` supports initialization from user-defined parameters and pdf value using a `__call__` method. Fitting the distribution on a set of 3-dimensional arrays (which are normalized if needed) can be done as follows:

```python
>>> dummy_data = np.array([[ 0.        , -0.4472136 ,  0.89442719],
                           [-0.81649658,  0.40824829,  0.40824829],
                           [-0.85811633, -0.19069252,  0.47673129],
                           [ 0.30151134, -0.90453403,  0.30151134],
                           [ 0.53452248, -0.80178373, -0.26726124],
                           [-0.53452248, -0.80178373, -0.26726124]])
>>> kent = Kent.fit(dummy_data)
>>> print(kent)
Kent distribution with parameters:
   γ1 = [-0.19689796 -0.64910057  0.73477864]
   γ2 = [-0.69428639  0.621472    0.36295864]
   γ3 = [-0.692241   -0.43868099 -0.57302826]
   κ = 3.6618129030645763
   β = 0.572062833756161
```

### Example for different values of kappa and beta
A visualization of the impact of kappa and beta on the shape of the distribution is shown below:

![kent example](examples/kent.png)
