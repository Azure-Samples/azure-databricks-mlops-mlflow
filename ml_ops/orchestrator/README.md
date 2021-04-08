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
- It will log artifacts, metrics, parameters, trained model into MlFlow.

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
To perform this set `spark.databricks.conda.condaMagic.enabled` to `true` under “Spark Config” (Edit > Advanced Options > Spark).- `[^1]`

### How to install libraries using pip

The following example can be used to install any wheel package stored DBFS folder.

```sh
%pip install dbfs/<path>/<package_name>.whl
```

## Calling MLOps Python Functions

MLOps Python Functions will be packaged as a wheel package and orchestrator notebook will call the python functions from wheel package. - `[^3]`

## Execution of Orchestrator

Orchestrator will be executed from DataBricks Job.-`[^3]`

## Error handling

`try..catch` block to handle exceptions -

```py
try:
  model = clustering()
except(Exception ex):
  logger.error(f"Encountered error: {ex.Message}")
  dbutils.notebook.exit(f"Encountered error :{ex.Message}")
```

## Observability

`Opencensus` library is used to capture logs and metrics and send it to Application Insights. - `[^2]`

## Secret Management

The following secrets need to be stored in [Databricks Secret Scope](https://docs.microsoft.com/en-us/azure/databricks/security/secrets/):

- Application Insights Instrumentation Key
- Azure ADLS Gen2 Storage Details

The following example can be used to fetch secrets from Databricks Secret Scope.

```py
secret_value = dbutils.secrets.get(scope = "<scope-name>", key = "<secret-name>")
```

## References

1. [Enable pip magic commands](https://databricks.com/blog/2020/06/17/simplify-python-environment-management-on-databricks-runtime-for-machine-learning-using-pip-and-conda.html
2. [OpenCensus](https://docs.microsoft.com/en-us/azure/azure-monitor/app/opencensus-python)
3. [DataBricks Job API](https://docs.databricks.com/dev-tools/api/latest/jobs.html)
4. [DataBricks Cluster API](https://docs.databricks.com/dev-tools/api/latest/clusters.html)
5. [DataBricks CLI](https://docs.databricks.com/dev-tools/cli/index.html)
