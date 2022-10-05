import logging

import lightgbm as lgb
import mlflow
import pandas as pd
from monitoring.app_logger import AppLogger, get_disabled_logger
from opencensus.trace.tracer import Tracer
from taxi_fares.training.evaluate import get_model_metrics, split_data
from taxi_fares.training.train import train


def run(
    train_df: pd.DataFrame,
    mlflow: mlflow,
    app_logger: AppLogger = get_disabled_logger(),
    parent_tracer: Tracer = None,
) -> lgb.Booster:
    """MLOps training entry point.

    Args:
        train_df (pd.DataFrame): data for training, output of feature engineering
        mlflow (mlflow): mlflow object that is having an active run
                         initiated by mlflow.start_run
        app_logger (monitoring.app_logger.AppLogger): AppLogger object deafult
                                        to monitoring.app_logger.get_disabled_logger
        parent_tracer (Tracer): OpenCensus parent tracer for correlation
    Returns:
        Ridge: trained model
    """
    logger = logging.getLogger(__name__)
    try:
        component_name = "Taxi_Fare_Training"

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

        logger.info("Running MLOps training")

        params = {"num_leaves": 32, "objective": "regression", "metric": "rmse"}
        num_rounds = 100
        for k, v in params.items():
            logger.info(f"Training parameter {k}: {v}")
            mlflow.log_param("training_param_" + k, v)
        mlflow.log_param("training_param_num_rounds", num_rounds)

        logger.info("Spliting data for train and test")
        data = split_data(train_df)

        logger.info("Train the model")
        with tracer.span("train_model"):
            model = train(data["train"], params, num_rounds)

        logger.info("Log the metrics for the model")
        metrics = get_model_metrics(model, data["test"])
        for (k, v) in metrics.items():
            logger.info(f"Metric {k}: {v}")
            mlflow.log_metric(k, v)

        logger.info("Completed MLOps training")
        return model
    except Exception as exp:
        logger.error("an exception occurred in training")
        raise Exception("an exception occurred in training") from exp
