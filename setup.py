#!/usr/bin/env python
"""Setup file for feature_generator package"""

from setuptools import setup, find_packages

setup(
    name='feature_generator',
    version='0.0.13',
    py_modules=['data_handling', 'feature_generator'],
    author='Paul Wormuth',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'io',
        'sys',
        'os',
        're',
        'json',
        'pandas',
        'numpy',
        'abc',
        'copy',
        'collections',
        'sklearn.ensemble',
        'sklearn.preprocessing',
        'sklearn.feature_extraction.text',
        'sklearn.linear_model',
        'sklearn.metrics',
        'sklearn.pipeline',
        'sklearn.grid_search'
        ],
    entry_points={
        'console_scripts': [
            'data_handling=feature_generator.data_handling:input_args_handler_fg',
            'data_cleaner=feature_generator.data_handling:data_cleaner_fg',
            'feature_generator=feature_generator.feature_generator:feature_generator_classes',
            'compare_scores=feature_generator.feature_generator:compare_scores',
        ]
    },
    include_package_data=True,
    # test_suite='nose.collector',
    # tests_require=['nose'],
)