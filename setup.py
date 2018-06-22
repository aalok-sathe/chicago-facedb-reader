#!/bin/env python

from setuptools import setup
from setuptools import find_packages

setup(name='cfd-reader',
      version='0.1',
      description='A Python module to access the Chicago Face Database',
      url='http://gitlab.com/aalok-sathe/cfd-reader',
      author='Aalok Sathe',
      author_email='aalok.sathe@richmond.edu',
      license='GPL-3',
      packages=find_packages(),
      install_requires=['numpy>=1.13.0',
                        'opencv-python>=3.4.0.12'],)
