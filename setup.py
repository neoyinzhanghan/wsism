# the most basic setup.py file
from setuptools import setup, find_packages

setup(
    name='wsism',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'tqdm'
    ]
)