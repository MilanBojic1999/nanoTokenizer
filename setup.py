from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')

ext = Extension(
    "two_max_pairs",
    sources=["wrapper.pyx"],
    libraries=["twomaxpairs"],
    library_dirs=[".", os.path.join(CUDA_HOME, "lib64")],
    include_dirs=[np.get_include(), os.path.join(CUDA_HOME, "include")],
    language="c++",
)

setup(
    name="two_max_pairs",
    version="0.1",
    description="Cython wrapper for the two_max_pairs C++ library",
    ext_modules=cythonize([ext]),
    zip_safe=False,
    install_requires=[
        "Cython>=0.29.21",
    ],
)
