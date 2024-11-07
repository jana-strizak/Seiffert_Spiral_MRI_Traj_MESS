#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file to set up the cython nyquist test kernel
'''
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("SeiffertSpiralMRITraj/nyquistTest.pyx")
)
'''

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Specify your extension module
extensions = [
    Extension(
        name="SeiffertSpiralMRITraj.nyquistTest",
        sources=["SeiffertSpiralMRITraj/nyquistTest.pyx"],
        include_dirs=[np.get_include()]  # Include NumPy headers if needed
    )
]

setup(
    name="SeiffertSpiralMRITraj",
    version="0.1.0",
    description="A package for making Seiffert Spiral MRI trajectories",
    install_requires=[
        "sigpy>=0.1.26",
        "matplotlib>=3.9.2",
        "scipy>=1.14.1",
        "cupy>=13.3.0",
        "Cython==0.29.36",
        "nibabel>=5.3.2",
        "pyMapVBVD>=0.6.1",
        "matplotlib-inline",
        "ipython"
    ],
    ext_modules=cythonize(extensions),
    packages=["SeiffertSpiralMRITraj"]
)
