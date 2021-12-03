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
	cd ml_source && coverage run --source=diabetes,monitoring -m unittest discover
	cd ml_source && coverage report -m

## unit test mlops locally
test-mlops: install-mlops
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
	mkdir -p .env
	cp ~/.databrickscfg .env/.databrickscfg
	$(info Creating env script file for mlflow)
	DATABRICKS_HOST="$$(cat ~/.databrickscfg | grep '^host' | cut -d' ' -f 3)"; \
	DATABRICKS_TOKEN="$$(cat ~/.databrickscfg | grep '^token' | cut -d' ' -f 3)"; \
	echo "export MLFLOW_TRACKING_URI=databricks"> .env/.databricks_env.sh; \
	echo "export DATABRICKS_HOST=$$DATABRICKS_HOST" >> .env/.databricks_env.sh; \
	echo "export DATABRICKS_TOKEN=$$DATABRICKS_TOKEN" >> .env/.databricks_env.sh

## databricks init (create cluster, base workspace, mlflow experiment, secret scope)
databricks-init:
	echo "Creating databricks workspace root directory"; \
	databricks workspace mkdirs /azure-databricks-mlops-mlflow; \
	echo "Creating databricks dbfs root directory"; \
	databricks fs mkdirs dbfs:/FileStore/libraries/azure-databricks-mlops-mlflow; \
	CLUSTER_ID="$$(databricks clusters list --output json | \
				   jq ".clusters[] | select(.cluster_name == \"azure-databricks-mlops-mlflow\") | .cluster_id")"; \
	echo "Got existing cluster azure-databricks-mlops-mlflow with id: $$CLUSTER_ID"; \
	if [[ $$CLUSTER_ID == "" ]]; then \
		echo "Creating databricks cluster azure-databricks-mlops-mlflow"; \
		databricks clusters create --json-file ml_ops/deployment/databricks/cluster_template.json; \
	fi; \
	SECRET_SCOPE_NAME="$$(databricks secrets list-scopes --output json | \
				   jq ".scopes[] | select(.name == \"azure-databricks-mlops-mlflow\") | .name")"; \
	echo "Got existing secret scope $$SECRET_SCOPE_NAME"; \
	if [[ $$SECRET_SCOPE_NAME == "" ]]; then \
		echo "Creating databricks secret scope azure-databricks-mlops-mlflow"; \
		databricks secrets create-scope --scope azure-databricks-mlops-mlflow --initial-manage-principal users; \
	fi; \
	MLFLOW_EXPERIMENT_ID="$$(source .env/.databricks_env.sh && mlflow experiments list | \
							 grep '/azure-databricks-mlops-mlflow/Experiment' | \
							 cut -d' ' -f 1)"; \
	echo "Got existing mlflow experiment id: $$MLFLOW_EXPERIMENT_ID"; \
	if [[ "$$MLFLOW_EXPERIMENT_ID" == "" ]]; then \
		echo "Creating mlflow experiment in databricks workspace /azure-databricks-mlops-mlflow/Experiment directory"; \
		source .env/.databricks_env.sh && mlflow experiments create --experiment-name /azure-databricks-mlops-mlflow/Experiment; \
	fi; \

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
	
## databricks secrets put application insights key
databricks-add-app-insights-key:
	$(info Put app insights key)
	@read -p "Enter App insights key: " app_insights_key; \
	if [[ "$$app_insights_key" != '' ]]; then \
		echo "Setting app insights key : $$app_insights_key "; \
		databricks secrets put --scope azure-databricks-mlops-mlflow --key app_insights_key --string-value "$$app_insights_key"; \
	fi; \

## databricks deploy (upload wheel pacakges to databricks DBFS workspace)
databricks-deploy-code: dist
	$(info Upload wheel packages into databricks dbfs root directory)
	databricks fs cp --overwrite --recursive dist/ dbfs:/FileStore/libraries/azure-databricks-mlops-mlflow/
	$(info Importing orchestrator notebooks into databricks workspace root directory)
	databricks workspace import_dir --overwrite ml_ops/orchestrator/ /azure-databricks-mlops-mlflow/
	$(info Create or update databricks jobs)

## databricks deploy jobs (create databricks jobs)
databricks-deploy-jobs: databricks-deploy-code
	$(info Getting required values from databricks)
	CLUSTER_ID="$$(databricks clusters list --output json | \
				   jq ".clusters[] | select(.cluster_name == \"azure-databricks-mlops-mlflow\") | .cluster_id")"; \
	echo "Got existing cluster id: $$CLUSTER_ID"; \
	TRAINING_JOB_ID="$$(databricks jobs list --output json | \
						jq ".jobs[] | select(.settings.name == \"diabetes_model_training\") | .job_id")"; \
	echo "Got existing diabetes_model_training job id: $$TRAINING_JOB_ID"; \
	if [[ "$$TRAINING_JOB_ID" == "" ]]; then \
		databricks jobs create --json "{\"name\": \"diabetes_model_training\", \"existing_cluster_id\": $$CLUSTER_ID}"; \
		TRAINING_JOB_ID="$$(databricks jobs list --output json | \
							jq ".jobs[] | select(.settings.name == \"diabetes_model_training\") | .job_id")"; \
		echo "Created diabetes_model_training with job id: $$TRAINING_JOB_ID"; \
	fi; \
	BATCH_SCORING_JOB_ID="$$(databricks jobs list --output json | \
							 jq ".jobs[] | select(.settings.name == \"diabetes_batch_scoring\") | .job_id")"; \
	echo "Got existing diabetes_batch_scoring job id: $$BATCH_SCORING_JOB_ID"; \
	if [[ "$$BATCH_SCORING_JOB_ID" == "" ]]; then \
		databricks jobs create --json "{\"name\": \"diabetes_batch_scoring\", \"existing_cluster_id\": $$CLUSTER_ID}"; \
		BATCH_SCORING_JOB_ID="$$(databricks jobs list --output json | \
								 jq ".jobs[] | select(.settings.name == \"diabetes_batch_scoring\") | .job_id")"; \
		echo "Created diabetes_batch_scoring with job id: $$BATCH_SCORING_JOB_ID"; \
	fi; \
	MLFLOW_EXPERIMENT_ID="$$(source .env/.databricks_env.sh && mlflow experiments list | \
							 grep '/azure-databricks-mlops-mlflow/Experiment' | \
							 cut -d' ' -f 1)"; \
	echo "Got existing mlflow experiment id: $$MLFLOW_EXPERIMENT_ID"; \
	echo "Updating diabetes_model_training by using template ml_ops/deployment/databricks/job_template_diabetes_training.json"; \
	TRAINING_JOB_UPDATE_JSON="$$(cat ml_ops/deployment/databricks/job_template_diabetes_training.json | \
								 sed "s/\"FILL_JOB_ID\"/$$TRAINING_JOB_ID/" | \
								 sed "s/FILL_MLFLOW_EXPERIMENT_ID/$$MLFLOW_EXPERIMENT_ID/" | \
								 sed "s/\"FILL_CLUSTER_ID\"/$$CLUSTER_ID/")"; \
	databricks jobs reset --job-id $$TRAINING_JOB_ID --json "$$TRAINING_JOB_UPDATE_JSON"; \
	echo "Updating diabetes_batch_scoring by using template ml_ops/deployment/databricks/job_template_diabetes_batch_scoring.json"; \
	BATCH_SCORING_JOB_UPDATE_JSON="$$(cat ml_ops/deployment/databricks/job_template_diabetes_batch_scoring.json | \
								 sed "s/\"FILL_JOB_ID\"/$$BATCH_SCORING_JOB_ID/" | \
								 sed "s/FILL_MLFLOW_EXPERIMENT_ID/$$MLFLOW_EXPERIMENT_ID/" | \
								 sed "s/\"FILL_CLUSTER_ID\"/$$CLUSTER_ID/")"; \
	databricks jobs reset --job-id $$BATCH_SCORING_JOB_ID --json "$$BATCH_SCORING_JOB_UPDATE_JSON"; \

## deploy databricks all
deploy: databricks-deploy-jobs

## run databricks diabetes_model_training job
run-diabetes-model-training:
	$(info Triggering model training job)
	TRAINING_JOB_ID="$$(databricks jobs list --output json | \
						jq ".jobs[] | select(.settings.name == \"diabetes_model_training\") | .job_id")"; \
	RUN_ID="$$(databricks jobs run-now --job-id $$TRAINING_JOB_ID | \
			   jq ".number_in_job")"; \
	DATABRICKS_HOST="$$(cat ~/.databrickscfg | grep '^host' | cut -d' ' -f 3)"; \
	DATABRICKS_ORG_ID="$$(echo $$DATABRICKS_HOST | cut -d'-' -f 2 | cut -d'.' -f 1)"; \
	echo "Open the following link in browser to check result -"; \
	echo "$$DATABRICKS_HOST/?o=$$DATABRICKS_ORG_ID/#job/$$TRAINING_JOB_ID/run/$$RUN_ID"; \

	
## run databricks diabetes_batch_scoring job
run-diabetes-batch-scoring:
	$(info Triggering batch scoring job)
	BATCH_SCORING_JOB_ID="$$(databricks jobs list --output json | \
							 jq ".jobs[] | select(.settings.name == \"diabetes_batch_scoring\") | .job_id")"; \
	RUN_ID="$$(databricks jobs run-now --job-id $$BATCH_SCORING_JOB_ID | \
			   jq ".number_in_job")"; \
	DATABRICKS_HOST="$$(cat ~/.databrickscfg | grep '^host' | cut -d' ' -f 3)"; \
	DATABRICKS_ORG_ID="$$(echo $$DATABRICKS_HOST | cut -d'-' -f 2 | cut -d'.' -f 1)"; \
	echo "Open the following link in browser to check result -"; \
	echo "$$DATABRICKS_HOST/?o=$$DATABRICKS_ORG_ID/#job/$$BATCH_SCORING_JOB_ID/run/$$RUN_ID"; \

# continuous integration (CI)
ci: lint test dist

# continuous deployment (CD)
cd: deploy