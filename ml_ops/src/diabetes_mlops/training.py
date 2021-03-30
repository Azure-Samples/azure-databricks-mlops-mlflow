import logging

import pandas as pd
from diabetes.training.evaluate import get_model_metrics, split_data
from diabetes.training.train import train_model
from sklearn.linear_model import Ridge


def run(train_df: pd.DataFrame) -> Ridge:
    """MLOps training entry point.

    Args:
        train_df (pd.DataFrame): data for training, output of feature engineering

    Returns:
        Ridge: trained model
    """
    logging.info("Running MLOps training")

    ridge_args = {"alpha": 0.5}
    logging.debug(f"Defining training parameters {ridge_args}")

    logging.debug("Spliting data for train and test")
    data = split_data(train_df)

    logging.debug("Train the model")
    model = train_model(data["train"], ridge_args)

    logging.debug("Log the metrics for the model")
    metrics = get_model_metrics(model, data["test"])
    for (k, v) in metrics.items():
        logging.info(f"Metric {k}: {v}")

    logging.info("Completed MLOps training")
    return model
