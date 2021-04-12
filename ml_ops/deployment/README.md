# Deployment

## Overview

This document covers the deployment guide for MLOps.

## Databricks Cluster

For Orchestrator job, either an existing cluster can be used or a new cluster can be created. However, we need to be sure to set following [properties](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/clusters#--request-structure-of-the-cluster-definition) in the cluster.

- Cluster Mode: High Concurrency
- DataBricks Runtime Version : 8.1 LTS ML (includes Apache Spark 3.0.1, Scala 2.12)
- Enable Autoscaling: True
- Worker Type: Standard_F4s
- Driver Type: Standard_F4s
- Spark Settings under “Spark Config” (Edit > Advanced Options > Spark)
  
  ```configuration
  spark.databricks.cluster.profile serverless
  spark.databricks.repl.allowedLanguages sql,python,r
  spark.databricks.conda.condaMagic.enabled true
  ```

## Databricks Job

Orchestrator DataBricks Job from a [Databricks Job create template](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/jobs#--create) can be created using following example CLI command -

```sh
databricks jobs create --json-file <job-template-file>.json
```

Orchestrator DataBricks Job from a [Databricks Job reset template](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/jobs#--reset) can be updated using following example CLI command -

```sh
databricks jobs reset --job-id <job-id of existing job> --json-file <job-template-file>.json
```

## Databricks MLflow Experiment

MLflow Experiment can be created using [Databricks Workspace Portal](https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/tracking#workspace-experiments) or using following CLI commands -

```sh
export MLFLOW_TRACKING_URI=databricks
export DATABRICKS_HOST=<databricks host>
export DATABRICKS_TOKEN=<databricks token>
mlflow experiments create --experiment-name /<path in databricks workspace>/<experiment name>
```

Get `DATABRICKS_HOST` and `DATABRICKS_TOKEN` from [Databricks CLI Reference](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/cli/)

## Databricks DBFS Upload

The following CLI command can be used to upload Wheel package into DataBricks DBFS.

```sh
databricks fs cp --overwrite python-package.whl <dbfs-path>
```

## Databricks Notebook Import

The following CLI command can be used to import orchestrator python file as a DataBricks notebook into DataBricks workspace.

```sh
databricks workspace import -l PYTHON -f SOURCE -o <orchestrator-notebook-python-file>.py <databricks-workspace-path>
```

## Orchestrator DataBricks Job trigger

Orchestrator databricks job can be triggered using following ways -

- Scheduled :  
  - Cron based scheduling.
- Manual :  
  - Databricks workspace portal but clicking on `Run Now With Different Parameters`.
  - Via [Databricks-CLI](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/cli/jobs-cli).
  - Via [Databricks-API](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/jobs#--run-now).
