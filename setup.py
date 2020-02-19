#!/usr/bin/env python
import setuptools

with open("README",'r') as fh:
	long_description = fh.read()

setuptools.setup(
	name="pocketSearch",
	version="1.1",
	author="Matt Sinclair",
	author_email="mts7@illinois.edu",
	description="Package to find surrogate structure for catalytic site of a protein",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/msinclair/bioinformatics"
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"Operating System :: Linux",
	],
	python_requires='>=3.6',
)

