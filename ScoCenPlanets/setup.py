# -*- coding: utf-8 -*-
"""
setup.py - G Bouma (luke@astro.princeton.edu) - Mar 2021
"""
__version__ = '0.0.0'

import sys
from setuptools import setup

###############
## RUN SETUP ##
###############

# run setup.
setup(
    name='ScoCenPlanets',
    version=__version__,
    description=('ScoCenPlanets'),
    long_description_content_type="text/markdown",
    keywords='astronomy',
    url='https://github.com/gurmehersk/Sco-Cen-Planets',
    author='Gurmeher Kathuria',
    author_email='gurmehersk@gmail.com',
    packages=[
        'ScoCenPlanets',
    ],
    include_package_data=True,
    zip_safe=False,
)
