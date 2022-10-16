import logging

import lightgbm as lgb
import mlflow
from databricks import feature_store
from databricks.feature_store.training_set import TrainingSet
from mlflow.entities.model_registry import ModelVersion
from monitoring.app_logger import AppLogger, get_disabled_logger
from opencensus.trace.tracer import Tracer

from taxi_fares_mlops.utils import get_latest_model_version


def run(
    trained_model: lgb.Booster,
    training_set: TrainingSet,
    mlflow: mlflow,
    model_name: str = "taxi_fares",
    app_logger: AppLogger = get_disabled_logger(),
    parent_tracer: Tracer = None,
) -> ModelVersion:
    """MLOps publish model in mlflow model registry - entry point.

    Args:
        trained_model (Ridge): trained Ridge model
        mlflow (mlflow): mlflow object that is having an active run
                         initiated by mlflow.start_run
        model_name (str, optional): model name in mlflow model registry.
                                    Defaults to "taxi_fares".
        app_logger (monitoring.app_logger.AppLogger): AppLogger object deafult
                                        to monitoring.app_logger.get_disabled_logger
        parent_tracer (Tracer): OpenCensus parent tracer for correlation
    Returns:
        mlflow.entities.model_registry.ModelVersion: registered model details
    """
    logger = logging.getLogger(__name__)
    try:
        component_name = "Taxi_Fares_Publish_Model"

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

        logger.info("Publishing trained model into mlflow model registry")
        with tracer.span("register_model"):
            fs = feature_store.FeatureStoreClient()
            fs.log_model(
                trained_model,
                artifact_path="model_packaged",
                flavor=mlflow.lightgbm,
                training_set=training_set,
                registered_model_name=model_name,
            )
        model_version = get_latest_model_version(model_name)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("model_name", model_name)

        logger.info(f"published model name: {model_name}, version: {model_version}")
        logger.info("Completed MLOps publish model")
    except Exception as exp:
        logger.error("an exception occurred in publish model")
        raise Exception("an exception occurred in publish model") from exp
