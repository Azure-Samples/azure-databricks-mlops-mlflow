import logging
from pathlib import Path

import mlflow
import pandas as pd
from diabetes.scoring.batch.run import batch_scoring
from diabetes.training.evaluate import get_model_metrics
from monitoring.app_logger import AppLogger, get_disabled_logger
from opencensus.trace.tracer import Tracer
from sklearn.linear_model import Ridge


def run(
    trained_model: Ridge,
    df_input: pd.DataFrame,
    mlflow: mlflow,
    mlflow_log_tmp_dir: str,
    app_logger: AppLogger = get_disabled_logger(),
    parent_tracer: Tracer = None,
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
        app_logger (monitoring.app_logger.AppLogger): AppLogger object deafult
                                        to monitoring.app_logger.get_disabled_logger
        parent_tracer (Tracer): OpenCensus parent tracer for correlation
    """
    logger = logging.getLogger(__name__)
    try:
        component_name = "Diabetes_Scoring_Batch"
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

        logger.info("Running MLOps batch scoring")
        with tracer.span("batch_scoring"):
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
    except Exception as exp:
        logger.error("an exception occurred in scoring batch")
        raise Exception("an exception occurred in scoring batch") from exp
