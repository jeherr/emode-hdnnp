# To make TensorMol available.
# sudo pip install -e .
#
# to make and upload a source dist
# python setup.py sdist
# twine upload dist/*
# And of course also be me.
#

from __future__ import absolute_import,print_function
from distutils.core import setup, Extension
import numpy
import os

NL = Extension(
'NL',
sources=['./C_API/NL.cpp'],
extra_compile_args=['-std=c++0x','-g','-fopenmp','-w'],
extra_link_args=['-lgomp'],
include_dirs=[numpy.get_include()]+['./C_API/'])


# run the setup
setup(name='emode_hdnnp',
      license='GPL3',
      packages=['emode_hdnnp'],
      ext_modules=[NL])
