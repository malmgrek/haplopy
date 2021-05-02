import os
from setuptools import setup, find_packages


with open(os.path.join(os.path.dirname(__file__), "README.md")) as fh:
    long_description = fh.read()


setup(
    name='haplopy',
    version='0.1.0',
    description='Haplotype reconstruction from unphased diplotypes',
    author='Stratos Staboulis',
    url="https://github.com/malmgrek/haplopy",
    packages=find_packages(exclude=["contrib", "doc", "tests"]),
    install_requires=[
        "numpy>=1.10.0",
        "scipy>=0.13.0",
        "matplotlib",
    ],
    extras_require={
        "test": ["pytest"]
    },
    keywords="statistics modeling population genetics",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
