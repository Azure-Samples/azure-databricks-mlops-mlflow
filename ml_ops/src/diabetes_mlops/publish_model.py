import logging
import os
import tempfile

import mlflow
from mlflow.entities.model_registry import ModelVersion
from monitoring.app_logger import AppLogger, get_disabled_logger
from opencensus.trace.tracer import Tracer
from sklearn.linear_model import Ridge


def run(
    trained_model: Ridge,
    mlflow: mlflow,
    model_name: str = "diabetes",
    app_logger: AppLogger = get_disabled_logger(),
    parent_tracer: Tracer = None,
) -> ModelVersion:
    """MLOps publish model in mlflow model registry - entry point.

    Args:
        trained_model (Ridge): trained Ridge model
        mlflow (mlflow): mlflow object that is having an active run
                         initiated by mlflow.start_run
        model_name (str, optional): model name in mlflow model registry.
                                    Defaults to "diabetes".
        app_logger (monitoring.app_logger.AppLogger): AppLogger object deafult
                                        to monitoring.app_logger.get_disabled_logger
        parent_tracer (Tracer): OpenCensus parent tracer for correlation
    Returns:
        mlflow.entities.model_registry.ModelVersion: registered model details
    """
    logger = logging.getLogger(__name__)
    try:
        component_name = "Diabetes_Publish_Model"

        # mlflow tracking
        mlflow_run = mlflow.active_run()
        mlflow_run_id = mlflow_run.info.run_id
        mlflow_experiment_id = mlflow_run.info.experiment_id

        logger = app_logger.get_logger(
            component_name=component_name,
            custom_dimensions={
                "mlflow_run_id": mlflow_run_id,
                "mlflow_experiment_id": mlflow_experiment_id,
            },
        )
        tracer = app_logger.get_tracer(
            component_name=component_name, parent_tracer=parent_tracer
        )

        logger.info("Running MLOps publish model")

        temp_model_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_model_dir, model_name)
        with tracer.span("save_model"):
            mlflow.sklearn.save_model(trained_model, model_path)
        mlflow.log_artifact(model_path)
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=model_name
        )

        logger.info("Publishing trained model into mlflow model registry")
        with tracer.span("register_model"):
            model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        model_version = model_details.version

        mlflow.log_param("model_version", model_version)
        mlflow.log_param("model_name", model_name)

        logger.info(f"published model name: {model_name}, version: {model_version}")
        logger.info("Completed MLOps publish model")

        return model_details
    except Exception as exp:
        logger.error("an exception occurred in publish model")
        raise Exception("an exception occurred in publish model") from exp
