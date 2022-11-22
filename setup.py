#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>
#
# Code based on the paper:
# "Unsupervised Voice Activity Detection by Modeling Source
# and System Information using Zero Frequency Filtering"
# Authors: Eklavya Sarkar, RaviShankar Prasad, Mathew Magimai Doss

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="ZFF-VAD",
    version=open("version.txt").read().rstrip(),
    description="""
    ZFF-VAD, a tool for segmenting audio files into segments
    which only contain speech, based on the voice source
    and vocal tract system information.
    """,
    url="",
    license="GPLv3",
    author="Eklavya Sarkar",
    author_email="eklavya.sarkar@idiap.ch",
    keywords="segmentation",
    long_description=open("README.rst").read(),
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "segment = zff.segment:segment",
        ]
    },
    test_suite="tests",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
