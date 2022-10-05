import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from monitoring.app_logger import AppLogger, get_disabled_logger
from opencensus.trace.tracer import Tracer
from pyspark.sql.dataframe import DataFrame
from taxi_fares.feature_eng.features import dropoff_features_fn, pickup_features_fn


def run(
    df_input: DataFrame,
    start_date: datetime,
    end_date: datetime,
    mlflow: mlflow,
    mlflow_log_tmp_dir: str,
    explain_features: bool = True,
    app_logger: AppLogger = get_disabled_logger(),
    parent_tracer: Tracer = None,
) -> Tuple[DataFrame, DataFrame]:
    """MLOps feature engineering entry point.

    Args:
        df_input (pd.DataFrame): input data - raw
        mlflow (mlflow): mlflow object that is having an active run
                         initiated by mlflow.start_run
        mlflow_log_tmp_dir (str): directory for putting files to be logged
                                  in mlflow artifacts
        explain_features (bool, optional): explain features, possible only with
                                           training data. Defaults to True.
        app_logger (monitoring.app_logger.AppLogger): AppLogger object default
                                        to monitoring.app_logger.get_disabled_logger
        parent_tracer (Tracer): OpenCensus parent tracer for correlation
    Returns:
        pd.DataFrame: clean and feature engineered data
    """
    logger = logging.getLogger(__name__)
    try:
        component_name = "Taxi_Fare_Feature_Eng"
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

        logger.info("Running MLOps feature engineering")
        logger.info(
            f"Shape of input dataframe, rows: {df_input.count()}, cols: {len(df_input.columns)}"  # noqa: E501
        )

        logger.info("Getting pickup features")
        with tracer.span("pickup_features"):
            pickup_features = pickup_features_fn(
                df_input,
                ts_column="tpep_pickup_datetime",
                start_date=start_date,
                end_date=end_date,
            )
        logger.info(
            f"Shape of pickup features dataframe, rows: {pickup_features.count()}, cols: {len(pickup_features.columns)}"  # noqa: E501
        )
        mlflow.log_param(
            "feature_engineering_pickup_features",
            (pickup_features.count(), len(pickup_features.columns)),
        )

        logger.info("Getting drop off features")
        with tracer.span("dropoff_features"):
            dropoff_features = dropoff_features_fn(
                df_input,
                ts_column="tpep_dropoff_datetime",
                start_date=start_date,
                end_date=end_date,
            )
        logger.info(
            f"Shape of dropoff features dataframe, rows: {dropoff_features.count()}, cols: {len(dropoff_features.columns)}"  # noqa: E501
        )
        mlflow.log_param(
            "feature_engineering_dropoff_features",
            (dropoff_features.count(), len(dropoff_features.columns)),
        )

        with tracer.span("explain_features"):
            if explain_features:
                logger.info("Getting feature explanations - statistics")
                feature_statistic_pickup_features = (
                    pickup_features.describe().toPandas()
                )
                feature_statistic_pickup_features.to_html(
                    Path(
                        mlflow_log_tmp_dir,
                        "feature_statistic_pickup_features.html",
                    ),
                    justify="center",
                    na_rep="",
                )
                feature_statistic_dropoff_features = (
                    dropoff_features.describe().toPandas()
                )
                feature_statistic_dropoff_features.to_html(
                    Path(
                        mlflow_log_tmp_dir,
                        "feature_statistic_dropoff_features.html",
                    ),
                    justify="center",
                    na_rep="",
                )
                logger.info("Getting feature explanations - box plot")
                pickup_features_pandas = pickup_features.toPandas()[
                    [
                        "mean_fare_window_1h_pickup_zip",
                        "count_trips_window_1h_pickup_zip",
                    ]
                ]
                numeric_cols = pickup_features_pandas.columns
                plot_data = pickup_features_pandas.copy()
                select_top_k = len(numeric_cols)
                n_col = 2
                n_row = math.ceil(select_top_k / n_col)
                s_col = 5
                s_row = 3
                fig, axs = plt.subplots(
                    n_row, n_col, figsize=(s_col * n_col, s_row * n_row), sharey=False
                )
                axs = axs.flatten()
                for index, col in enumerate(numeric_cols[:select_top_k]):
                    ax = sns.boxplot(
                        x="count_trips_window_1h_pickup_zip",
                        y=col,
                        data=plot_data,
                        ax=axs[index],
                    )
                    ax.set(title=col, ylabel="")
                fig.tight_layout()
                fig.savefig(
                    Path(mlflow_log_tmp_dir, "feature_pickup_features_boxplot.png")
                )
                dropoff_features_pandas = dropoff_features.toPandas()[
                    ["count_trips_window_30m_dropoff_zip", "dropoff_is_weekend"]
                ]
                numeric_cols = dropoff_features_pandas.columns
                plot_data = dropoff_features_pandas.copy()
                select_top_k = len(numeric_cols)
                n_col = 2
                n_row = math.ceil(select_top_k / n_col)
                s_col = 5
                s_row = 3
                fig, axs = plt.subplots(
                    n_row, n_col, figsize=(s_col * n_col, s_row * n_row), sharey=False
                )
                axs = axs.flatten()
                for index, col in enumerate(numeric_cols[:select_top_k]):
                    ax = sns.boxplot(
                        x="dropoff_is_weekend", y=col, data=plot_data, ax=axs[index]
                    )
                    ax.set(title=col, ylabel="")
                fig.tight_layout()
                fig.savefig(
                    Path(mlflow_log_tmp_dir, "feature_dropoff_features_boxplot.png")
                )

        logger.info("Completed MLOps feature engineering")
        return (pickup_features, dropoff_features)
    except Exception as exp:
        logger.error("an exception occurred in Feature Eng")
        raise Exception("an exception occurred in Feature Eng") from exp
