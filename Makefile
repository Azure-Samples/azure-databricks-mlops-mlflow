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
	rm -fr build/

## pcakage mlops
dist-mlops: clean
	python ml_ops/src/setup.py bdist_wheel
	rm -fr build/

## pcakage all
dist: dist-ml dist-mlops

## install ml locally
install-ml: clean
	python ml_source/src/setup.py install
	rm -fr build/

## install mlops locally
install-mlops: clean
	python ml_ops/src/setup.py install
	rm -fr build/

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

## lint all python src and tests
lint:
	flake8 --max-line-length=88 ml_ops/src ml_ops/tests ml_source/src ml_source/tests

## databricks authenticate
databricks-authenticate:
	$(info Authenticate Databricks CLI)
	$(info Follow https://docs.microsoft.com/en-us/azure/databricks/dev-tools/cli/ for getting Host and token value)
	databricks configure --token
	$(info Taking Backup of .databrickscfg file in .env/databrickscfg)
	cp ~/.databrickscfg .env/.databrickscfg
	$(info Creating env script file for mlflow)
	DATABRICKS_HOST="$$(cat ~/.databrickscfg | grep '^host' | cut -d' ' -f 3)"; \
	DATABRICKS_TOKEN="$$(cat ~/.databrickscfg | grep '^token' | cut -d' ' -f 3)"; \
	echo "export MLFLOW_TRACKING_URI=databricks"> .env/.databricks_env.sh; \
	echo "export DATABRICKS_HOST=$$DATABRICKS_HOST" >> .env/.databricks_env.sh; \
	echo "export DATABRICKS_TOKEN=$$DATABRICKS_TOKEN" >> .env/.databricks_env.sh

## databricks init (create cluster, base workspace, mlflow experiment, secret scope)
databricks-init:
	$(info Creating databricks cluster)
	databricks clusters create --json-file ml_ops/deployment/databricks/cluster_template.json
	$(info Creating databricks workspace root directory)
	databricks workspace mkdirs /azure-databricks-mlops-mlflow
	$(info Creating databricks dbfs root directory)
	databricks fs mkdirs dbfs:/FileStore/libraries/azure-databricks-mlops-mlflow
	$(info Creating databricks secret scope)
	databricks secrets create-scope --scope azure-databricks-mlops-mlflow --initial-manage-principal users
	$(info Creating mlflow experiment in databricks workspace root directory)
	source .env/.databricks_env.sh && mlflow experiments create --experiment-name /azure-databricks-mlops-mlflow/Experiment

## databricks secrets put
databricks-secrets-put:
	$(info Put databricks secret azure-blob-storage-account-name)
	@read -p "Enter Azure Blob storage Account Name: " stg_account_name; \
	databricks secrets put --scope azure-databricks-mlops-mlflow --key azure-blob-storage-account-name --string-value $$stg_account_name
	$(info Put databricks secret azure-blob-storage-container-name)
	@read -p "Enter Azure Blob storage Container Name: " stg_container_name; \
	databricks secrets put --scope azure-databricks-mlops-mlflow --key azure-blob-storage-container-name --string-value $$stg_container_name
	$(info Put databricks secret azure-shared-access-key)
	$(info Mount Blob Storage https://docs.microsoft.com/en-gb/azure/databricks/data/data-sources/azure/azure-storage)
	@read -p "Enter Azure Blob storage Shared Access Key: " shared_access_key; \
	databricks secrets put --scope azure-databricks-mlops-mlflow --key azure-blob-storage-shared-access-key --string-value $$shared_access_key

## databricks deploy (upload wheel pacakges to databricks DBFS workspace)
databricks-deploy: dist
	$(info Upload wheel packages into databricks dbfs root directory)
	databricks fs cp --overwrite --recursive dist/ dbfs:/FileStore/libraries/azure-databricks-mlops-mlflow/
	$(info Importing orchestrator notebooks into databricks workspace root directory)
	databricks workspace import_dir --overwrite ml_ops/orchestrator/ /azure-databricks-mlops-mlflow/
	$(info Create or update databricks jobs)