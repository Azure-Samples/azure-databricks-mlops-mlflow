# Registered Models Stages and Transitioning

This document describes a possible way to transitioning a model from different stages available in [Mlflow Model Registry](https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/model-registry#model-registry-concepts).

1. In this demo setup, currently [Continuous Integration (CI)](cicd.md) step does [register](../ml_ops/src/diabetes_mlops/../../../../ml_ops/src/diabetes_mlops/publish_model.py) the model in MLflow model registry in `None` stage.
2. Now the registered model can be [transitioned](https://www.mlflow.org/docs/latest/model-registry.html#transitioning-an-mlflow-models-stage) to next stage `Staging` post Integration test step.
3. Finally the model can be transitioned to stage `Production` during Continuous Deployment (CD) step.

## References

- [MLflow Model Registry on Azure Databricks](https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/model-registry)
- [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html)
