# SphereProba

## Installation
Clone the repository and use pip to install the package.
```bash
git clone https://github.com/rfayat/SphereProba.git
cd SphereProba
pip install -e .  # -e: editable
```
## Generating the example figures
To generate the example figures, install seaborn and [angle_visualization](https://github.com/rfayat/angle_visualization) and run:

```bash
python -m examples.generate_examples
```

## Von Mises Fisher
### Fit from a range of 3D vectors
```python
>>> dummy_data = np.array([[0, 0, 1.], [0, 0.01, 1.01]])
>>> print(VonMisesFisher.fit(dummy_data))
vMF distribution with parameters:
        μ = [0.         0.00495031 0.99998775]
        κ = 81613.99993282309

>>> dummy_data = np.random.random((100000, 3)) - np.array([[.5, .5, .5]])
>>> print(VonMisesFisher.fit(dummy_data))
vMF distribution with parameters:
        μ = [ 0.27893476 -0.72848701 -0.62570127]
        κ = 0.006484827495462602

>>> dummy_data = np.array([[0, 0, 1.], [0, 0, -1]])
>>> print(VonMisesFisher.fit(dummy_data, weights=np.array([1e3, 1])))
vMF distribution with parameters:
        μ = [0. 0. 1.]
        κ = 500.9975019979858
```
### Example for different kappas
![vmf example](examples/vmf.png)

## Kent (Fisher-Bingham 5)
### Fit from a range of 3D vectors
```python
>>> dummy_data = np.random.random((10000, 3)) - np.array([[.5, .3, .1]])
>>> print(Kent.fit(dummy_data))
Kent distribution with parameters:
        γ1 = [-0.01238757  0.43245837  0.9015688 ]
        γ2 = [-0.99658471  0.06827784 -0.04644414]
        γ3 = [-0.08164233 -0.89906501  0.4301356 ]
        κ = 2.769705279148834
        β = 0.05539125253921717
>>> dummy_data = np.array([[ 0.        , -0.4472136 ,  0.89442719],
                           [-0.81649658,  0.40824829,  0.40824829],
                           [-0.85811633, -0.19069252,  0.47673129],
                           [ 0.30151134, -0.90453403,  0.30151134],
                           [ 0.53452248, -0.80178373, -0.26726124],
                           [-0.53452248, -0.80178373, -0.26726124]])
Kent distribution with parameters:
   γ1 = [-0.19689796 -0.64910057  0.73477864]
   γ2 = [-0.69428639  0.621472    0.36295864]
   γ3 = [-0.692241   -0.43868099 -0.57302826]
   κ = 3.6618129030645763
   β = 0.572062833756161
```

### Example for different values of kappa and beta
![kent example](examples/kent.png)
