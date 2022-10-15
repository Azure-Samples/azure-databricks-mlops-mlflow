import logging
from pathlib import Path

import mlflow
import pandas as pd
import pyspark.sql.functions as func
from databricks import feature_store
from monitoring.app_logger import AppLogger, get_disabled_logger
from opencensus.trace.tracer import Tracer

from taxi_fares_mlops.utils import get_latest_model_version


def run(
    trained_model_name: str,
    score_df: pd.DataFrame,
    mlflow: mlflow,
    mlflow_log_tmp_dir: str,
    trained_model_version: str = None,
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
            cols = [
                "fare_amount",
                "trip_distance",
                "pickup_zip",
                "dropoff_zip",
                "rounded_pickup_datetime",
                "rounded_dropoff_datetime",
            ]
            score_df_reordered = score_df.select(cols)
            if trained_model_version is None or trained_model_version == "":
                trained_model_version = get_latest_model_version(trained_model_name)
            else:
                trained_model_version = int(trained_model_version)
            model_uri = f"models:/{trained_model_name}/{trained_model_version}"
            mlflow.log_param("trained_model_version", trained_model_version)
            logger.info(f"trained model version {trained_model_version}")
            fs = feature_store.FeatureStoreClient()
            predictions = fs.score_batch(model_uri, score_df_reordered)
            cols = [
                "prediction",
                "fare_amount",
                "trip_distance",
                "pickup_zip",
                "dropoff_zip",
                "rounded_pickup_datetime",
                "rounded_dropoff_datetime",
                "mean_fare_window_1h_pickup_zip",
                "count_trips_window_1h_pickup_zip",
                "count_trips_window_30m_dropoff_zip",
                "dropoff_is_weekend",
            ]

            with_predictions_reordered = (
                predictions.select(
                    cols,
                )
                .withColumnRenamed(
                    "prediction",
                    "predicted_fare_amount",
                )
                .withColumn(
                    "predicted_fare_amount",
                    func.round("predicted_fare_amount", 2),
                )
            )
        with_predictions_reordered.toPandas().to_html(
            Path(
                mlflow_log_tmp_dir,
                "batch_scoring_result.html",
            ),
            justify="center",
            na_rep="",
        )
        with_predictions_reordered.toPandas().to_csv(
            Path(
                mlflow_log_tmp_dir,
                "batch_scoring_result.csv",
            ),
            index=False,
        )
        logger.info("Completed MLOps batch scoring")
    except Exception as exp:
        logger.error("an exception occurred in scoring batch")
        raise Exception("an exception occurred in scoring batch") from exp
