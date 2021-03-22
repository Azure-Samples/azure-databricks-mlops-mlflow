.PHONY: clean clean-test clean-pyc clean-build
SHELL=/bin/bash

## remove Python file artifacts
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

## remove test and coverage artifacts
clean-test:
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

## remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

## remove all build, test, coverage and Python artifacts
clean: clean-build clean-pyc clean-test

## pcakage ml
dist-ml: clean
	python ml_source/src/setup.py bdist_wheel

## pcakage mlops
dist-mlops: clean
	python ml_ops/src/setup.py bdist_wheel

## pcakage all
dist: dist-ml dist-mlops

## install ml locally
install-ml: clean
	python ml_source/src/setup.py install

## install mlops locally
install-mlops: clean
	python ml_ops/src/setup.py install

## install all locally
install: install-ml install-mlops

## unit test ml locally
test-ml: install-ml
	cd ml_source && python -m unittest discover
	cd ml_source && coverage run --source=diabetes -m unittest discover
	cd ml_source && coverage report -m

## unit test mlops locally
test-mlops: install-mlops
	cd ml_ops && python -m unittest discover
	cd ml_ops && coverage run --source=diabetes_mlops -m unittest discover
	cd ml_ops && coverage report -m

## unit test all locally
test: test-ml test-mlops
	coverage combine ml_source/.coverage ml_ops/.coverage
	coverage report