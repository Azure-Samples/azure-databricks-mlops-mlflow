import logging

import mlflow
from monitoring.app_logger import AppLogger, get_disabled_logger
from opencensus.trace.tracer import Tracer
from sklearn.linear_model import Ridge


def run(
    mlflow: mlflow,
    model_version: str = None,
    model_name: str = "diabetes",
    app_logger: AppLogger = get_disabled_logger(),
    parent_tracer: Tracer = None,
) -> Ridge:
    """Load trained model from mlflow model registry.

    Args:
        mlflow (mlflow): mlflow object that is having an active run
                         initiated by mlflow.start_run
        model_version (str, optional): model version. Defaults to latest.
        model_name (str, optional): model name in mlflow model registry.
                                    Defaults to "diabetes".
        app_logger (monitoring.app_logger.AppLogger): AppLogger object deafult
                                        to monitoring.app_logger.get_disabled_logger
        parent_tracer (Tracer): OpenCensus parent tracer for correlation
    Returns:
        Ridge: trained model

    """
    logger = logging.getLogger(__name__)
    try:
        component_name = "Diabetes_Load_Model"
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

        logger.info("Running MLOps load model")

        client = mlflow.tracking.MlflowClient()
        if model_version is None:
            model_version_object_list = client.get_latest_versions(
                model_name, stages=["None"]
            )
            if len(model_version_object_list) == 0:
                logger.error(
                    f"There is no Model registered with this name: {model_name}"
                )
                return None
            model_version = model_version_object_list[0].version

        mlflow.log_param("model_version", model_version)
        mlflow.log_param("model_name", model_name)

        model_uri = "models:/{model_name}/{model_version}".format(
            model_name=model_name, model_version=model_version
        )
        with tracer.span("sklearn.load_model"):
            trained_model = mlflow.sklearn.load_model(model_uri)

        logger.info("Completed MLOps load model")
        return trained_model
    except Exception as exp:
        logger.error("an exception occurred in load model")
        raise Exception("an exception occurred in load model") from exp
