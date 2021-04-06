import logging
from pathlib import Path

import mlflow
import pandas as pd
from diabetes.scoring.batch.run import batch_scoring
from diabetes.training.evaluate import get_model_metrics
from sklearn.linear_model import Ridge


def run(
    trained_model: Ridge,
    df_input: pd.DataFrame,
    mlflow: mlflow,
    mlflow_log_tmp_dir: str,
) -> None:
    """[summary]

    Args:
        trained_model (Ridge): trained Ridge model
        df_input (pd.DataFrame): input dataframe for batch scoring,
                                 feature engineeringered.
        mlflow (mlflow): mlflow object that is having an active run
                         initiated by mlflow.start_run
        mlflow_log_tmp_dir (str): directory for puting files to be logged
                                  in mlflow artifacts

    """
    logger = logging.getLogger(__name__)
    logger.info("Running MLOps batch scoring")

    df_input["Y"] = batch_scoring(model=trained_model, df=df_input)

    df_input.to_html(
        Path(
            mlflow_log_tmp_dir,
            "batch_scoring_result.html",
        ),
        justify="center",
        na_rep="",
    )
    df_input.to_csv(
        Path(
            mlflow_log_tmp_dir,
            "batch_scoring_result.csv",
        ),
        index=False,
    )

    X = df_input.drop("Y", axis=1).values
    y = df_input["Y"].values
    metrics = get_model_metrics(trained_model, {"X": X, "y": y})
    for (k, v) in metrics.items():
        logger.info(f"Metric {k}: {v}")
        mlflow.log_metric(k, v)

    logger.info("Completed MLOps batch scoring")
