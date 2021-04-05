import logging

import mlflow
import pandas as pd
from diabetes.training.evaluate import get_model_metrics, split_data
from diabetes.training.train import train_model
from sklearn.linear_model import Ridge


def run(train_df: pd.DataFrame, mlflow: mlflow) -> Ridge:
    """MLOps training entry point.

    Args:
        train_df (pd.DataFrame): data for training, output of feature engineering
        mlflow (mlflow): mlflow object that is having an active run
                         initiated by mlflow.start_run

    Returns:
        Ridge: trained model
    """
    logger = logging.getLogger(__name__)
    logger.info("Running MLOps training")

    ridge_args = {"alpha": 0.5}
    mlflow.log_param("training_param_alpha", 0.5)
    logger.info(f"Defining training parameters {ridge_args}")

    logger.info("Spliting data for train and test")
    data = split_data(train_df)

    logger.info("Train the model")
    model = train_model(data["train"], ridge_args)

    logger.info("Log the metrics for the model")
    metrics = get_model_metrics(model, data["test"])
    for (k, v) in metrics.items():
        logger.info(f"Metric {k}: {v}")
        mlflow.log_metric(k, v)

    logger.info("Completed MLOps training")
    return model
