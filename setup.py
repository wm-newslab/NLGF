#!/usr/bin/env python

from setuptools import setup, find_packages

desc = """NLGF: Geo-Focus Identification of US Local News Articles"""

__appversion__ = None

exec(open('nlgf/version.py').read())


setup(
    name='nlgf',
    version=__appversion__,
    description=desc,
    long_description='See: Anonymous',
    author='Anonymous',
    author_email='Anonymous',
    url='Anonymous',
    packages=find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    install_requires=[
        'numpy', 
        'transformers', 
        'spacy', 
        'tqdm', 
        'openai', 
        'shapely', 
        'storysniffer',
        'seaborn',
        'xgboost', 
        'shap',
        'scikit-learn',
        'matplotlib',
        'pandas',
        'imbalanced-learn',
        'newspaper3k',
        'lxml-html-clean'
    ],
    scripts=[
        'bin/nlgf'
    ]
)
