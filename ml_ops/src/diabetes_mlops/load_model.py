import logging

import mlflow
from sklearn.linear_model import Ridge


def run(
    mlflow: mlflow, model_version: str = None, model_name: str = "diabetes"
) -> Ridge:
    """Load trained model from mlflow model registry.

    Args:
        mlflow (mlflow): mlflow object that is having an active run
                         initiated by mlflow.start_run
        model_version (str, optional): model version. Defaults to latest.
        model_name (str, optional): model name in mlflow model registry.
                                    Defaults to "diabetes".
    Returns:
        Ridge: trained model

    """
    logger = logging.getLogger(__name__)
    logger.info("Running MLOps load model")

    client = mlflow.tracking.MlflowClient()
    if model_version is None:
        model_version_object_list = client.get_latest_versions(
            model_name, stages=["None"]
        )
        if len(model_version_object_list) == 0:
            logger.error(f"There is no Model registered with this name: {model_name}")
            return None
        model_version = model_version_object_list[0].version

    mlflow.log_param("model_version", model_version)
    mlflow.log_param("model_name", model_name)

    model_uri = "models:/{model_name}/{model_version}".format(
        model_name=model_name, model_version=model_version
    )
    trained_model = mlflow.sklearn.load_model(model_uri)

    logger.info("Completed MLOps load model")
    return trained_model
