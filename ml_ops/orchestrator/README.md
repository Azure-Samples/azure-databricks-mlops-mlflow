# Orchestrator

## Overview

This document covers the design guide of the following orchestrators -

1. [diabetes_orchestrator_train.py](diabetes_orchestrator_train.py)
2. [diabetes_orchestrator_batch_score.py](diabetes_orchestrator_batch_score.py)

## Considerations

- It will be a Databricks notebook in Databricks workspace.
- It will be stored in GIT as a python file.
- It will use `dbutils` widgets for parametrization
- It will use `pip magic commands` for managing libraries.
- It will be executed from a Databricks Job.
- It will perform logging in Application Insights
- It will log artifacts, metrics, parameters, trained model into MLflow.

## Parameters

### Define Parameters

Parameters are defined using `dbutils.widgets.text`, example

```py
dbutils.widgets.text("<param_name>", "<default_value>")
```

### Read Parameters

Parameters are read using `dbutils.widgets.get`, example

```py
param_value = dbutils.widgets.get("<param_name>")
```

## Installation of libraries

### How to enable %pip magic commands

Starting with Databricks Runtime ML version 6.4 this feature can be enabled when creating a cluster.
To perform this set `spark.databricks.conda.condaMagic.enabled` to `true` under “Spark Config” (Edit > Advanced Options > Spark).

### How to install libraries using pip

Libraries are installed as [Notebook-scoped Python libraries](https://docs.microsoft.com/en-us/azure/databricks/libraries/notebooks-python-libraries), example

```sh
%pip install dbfs/<path>/<package_name>.whl
```

## Calling MLOps Python Functions

MLOps Python Functions are packaged as a wheel package and orchestrator notebook calls the python functions from wheel package.

## Execution of Orchestrator

Orchestrator are executed from DataBricks Job.

## Error handling

For error handling `try..catch` block is used to handle exceptions -

```py
try:
  model = run_training()
except(Exception ex):
  logger.error(f"Encountered error: {ex.Message}") # To log exception in Application Insights
  raise Exception(f"Encountered error - {ex}") from ex # To fail the Databricks Job Run
```

## Observability

[OpenCensus](https://docs.microsoft.com/en-us/azure/azure-monitor/app/opencensus-python) library is used to capture logs and metrics and send it to Application Insights.

## Secret Management

The following secrets need to be stored in [Databricks Secret Scope](https://docs.microsoft.com/en-us/azure/databricks/security/secrets/):

- Application Insights Instrumentation Key
- Azure ADLS Gen2 Storage Details (account name, container name, shared access key)

Secrets are read using `dbutils.secrets.get`, example

```py
secret_value = dbutils.secrets.get(scope = "<scope-name>", key = "<secret-name>")
```

## References

1. [Enable pip magic commands](https://databricks.com/blog/2020/06/17/simplify-python-environment-management-on-databricks-runtime-for-machine-learning-using-pip-and-conda.html)
2. [OpenCensus](https://docs.microsoft.com/en-us/azure/azure-monitor/app/opencensus-python)
3. [DataBricks Job API](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/jobs)
4. [DataBricks Cluster API](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/clusters)
5. [DataBricks CLI](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/cli/)
