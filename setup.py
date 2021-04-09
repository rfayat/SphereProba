"""Installation of SphereProba
Author: Romain Fayat, April 2021
"""
import os
from setuptools import setup


def read(fname):
    "Read a file in the current directory."
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="SphereProba",
    version="0.1",
    author="Romain Fayat",
    author_email="r.fayat@gmail.com",
    description="Code for fitting distribution functions on the unit sphere",
    install_requires=["numpy",
                      "scipy",
                      ],
    packages=["SphereProba"],
    long_description=read('README.md'),
)
