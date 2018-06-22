#!/bin/env python

from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(name='cfd-reader',
      version='0.1.1',
      description='A Python module to access the Chicago Face Database',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://gitlab.com/aalok-sathe/cfd-reader',
      author='Aalok Sathe',
      author_email='aalok.sathe@richmond.edu',
      license='GPL-3',
      packages=find_packages(),
      python_requires='>=3.4',
      install_requires=['numpy>=1.13.0',
                        'opencv-python>=3.4.0.12',
                        'progressbar2>=3.35.0',
                        'xlrd>=1.1.0',],
      classifiers=[
          'Development Status :: 1 - Planning',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries :: Python Modules'],)
