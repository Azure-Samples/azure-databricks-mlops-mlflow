import logging

import mlflow
from mlflow.entities.model_registry import ModelVersion
from sklearn.linear_model import Ridge


def run(trained_model: Ridge, mlflow: mlflow, model_name="diabetes") -> ModelVersion:
    """MLOps publish model in mlflow model registry - entry point.

    Args:
        trained_model (Ridge): trained Ridge model
        mlflow (mlflow): mlflow object that is having an active run
                         initiated by mlflow.start_run
        model_name (str, optional): model name in mlflow model registry.
                                    Defaults to "diabetes".

    Returns:
        mlflow.entities.model_registry.ModelVersion: registered model details
    """
    logger = logging.getLogger(__name__)
    logger.info("Running MLOps publish model")

    mlflow.sklearn.save_model(trained_model, model_name)
    mlflow.log_artifact(model_name)
    model_uri = "runs:/{run_id}/{artifact_path}".format(
        run_id=mlflow.active_run().info.run_id, artifact_path=model_name
    )

    logger.info("Publishing trained model into mlflow model registry")
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    model_versions = model_details.version

    mlflow.log_param("model_versions", model_versions)
    logger.info(f"published model name: {model_name}, version: {model_versions}")
    logger.info("Completed MLOps publish model")

    return model_details
