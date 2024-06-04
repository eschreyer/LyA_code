from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(name = 'xsection_cy_new', ext_modules = cythonize("xsection_cy_new.pyx", language_level = "3"), zip_safe = False, include_dirs = [numpy.get_include()])
