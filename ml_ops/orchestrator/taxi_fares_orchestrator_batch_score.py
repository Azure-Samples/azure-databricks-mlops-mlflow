# Databricks notebook source
"""Orchestrator notebook for taxifares training."""
# Initialization of dbutils to avoid linting errors during developing in vscode
from pyspark.sql import SparkSession


def get_dbutils(spark):
    """Return dbutils for databricks."""
    if spark.conf.get("spark.databricks.service.client.enabled") == "true":
        from pyspark.dbutils import DBUtils

        return DBUtils(spark)
    else:
        import IPython

        return IPython.get_ipython().user_ns["dbutils"]


spark = SparkSession.builder.appName("Pipeline").getOrCreate()
dbutils = get_dbutils(spark)

# COMMAND ----------

# Define parameters
dbutils.widgets.text(
    "taxi_fares_raw_data", "/databricks-datasets/nyctaxi-with-zipcodes/subsampled"
)
dbutils.widgets.text("taxi_fares_mount_point", "/mnt/data")
dbutils.widgets.text("mlflow_experiment_id", "")
dbutils.widgets.text("wheel_package_dbfs_base_path", "")
dbutils.widgets.text("wheel_package_taxi_fares_version", "")
dbutils.widgets.text("wheel_package_taxi_fares_mlops_version", "")
dbutils.widgets.text("execute_feature_engineering", "true")
dbutils.widgets.text("trained_model_version", "")
dbutils.widgets.text("scoring_data_start_date", "2016-02-01")
dbutils.widgets.text("training_data_end_date", "2016-02-29")

# COMMAND ----------

# Get wheel package parameters
wheel_package_dbfs_base_path = dbutils.widgets.get("wheel_package_dbfs_base_path")
wheel_package_taxi_fares_version = dbutils.widgets.get(
    "wheel_package_taxi_fares_version"
)
wheel_package_taxi_fares_mlops_version = dbutils.widgets.get(
    "wheel_package_taxi_fares_mlops_version"
)

# COMMAND ----------

# MAGIC %pip install $wheel_package_dbfs_base_path/taxi_fares-$wheel_package_taxi_fares_version-py3-none-any.whl # noqa: E501
# MAGIC %pip install $wheel_package_dbfs_base_path/taxi_fares_mlops-$wheel_package_taxi_fares_mlops_version-py3-none-any.whl # noqa: E501

# COMMAND ----------

# Imports
import os  # noqa: E402
import shutil  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import mlflow  # noqa: E402
from databricks import feature_store  # noqa: E402
from monitoring.app_logger import AppLogger, get_disabled_logger  # noqa: E402
from taxi_fares.utils.pyspark_utils import rounded_taxi_data  # noqa: E402
from taxi_fares_mlops.feature_engineering import run as run_feature_engineering  # noqa
from taxi_fares_mlops.scoring_batch import run as run_scoring_batch  # noqa: E402

# COMMAND ----------

# Get other parameters
mlflow_experiment_id = dbutils.widgets.get("mlflow_experiment_id")
execute_feature_engineering = dbutils.widgets.get("execute_feature_engineering")
taxi_fares_raw_data = dbutils.widgets.get("taxi_fares_raw_data")
taxi_fares_mount_point = dbutils.widgets.get("taxi_fares_mount_point")
trained_model_version = dbutils.widgets.get("trained_model_version")
scoring_data_start_date = dbutils.widgets.get("scoring_data_start_date")
scoring_data_end_date = dbutils.widgets.get("scoring_data_end_date")

# COMMAND ----------

# Initiate mlflow experiment
mlflow.start_run(experiment_id=int(mlflow_experiment_id), run_name="batch_scoring")
mlflow_run = mlflow.active_run()
mlflow_run_id = mlflow_run.info.run_id
mlflow_log_tmp_dir = "/tmp/" + str(mlflow_run_id)  # nosec: B108
Path(mlflow_log_tmp_dir).mkdir(parents=True, exist_ok=True)

# initiate app logger
if any(
    [
        True
        for secret in dbutils.secrets.list(scope="azure-databricks-mlops-mlflow")
        if "app_insights_key" in secret.key
    ]
):
    app_insights_key = dbutils.secrets.get(
        scope="azure-databricks-mlops-mlflow", key="app_insights_key"
    )
    config = {"app_insights_key": app_insights_key}
    app_logger = AppLogger(config=config)
else:
    app_logger = get_disabled_logger()
try:
    logger = app_logger.get_logger(
        component_name="Batch_Score_Orchestrator",
        custom_dimensions={
            "mlflow_run_id": mlflow_run_id,
            "mlflow_experiment_id": int(mlflow_experiment_id),
        },
    )
    tracer = app_logger.get_tracer(
        component_name="Batch_Score_Orchestrator",
    )
except Exception as ex:
    print(ex)
    mlflow.end_run()
    shutil.rmtree(mlflow_log_tmp_dir, ignore_errors=True)
    raise Exception(f"ERROR - in initializing app logger - {ex}") from ex

logger.info(f"Stating batch scoring with mlflow run id {mlflow_run_id}")

# COMMAND ----------

# Mount ADLS Gen2 storage container
try:
    logger.info(f"Mounting {taxi_fares_mount_point}")
    if any(mount.mountPoint == taxi_fares_mount_point for mount in dbutils.fs.mounts()):
        logger.info(f"Mount point exists {taxi_fares_mount_point}")
    else:
        storage_account_name = dbutils.secrets.get(
            scope="azure-databricks-mlops-mlflow", key="azure-blob-storage-account-name"
        )
        storage_container_name = dbutils.secrets.get(
            scope="azure-databricks-mlops-mlflow",
            key="azure-blob-storage-container-name",
        )
        storage_shared_key_name = dbutils.secrets.get(
            scope="azure-databricks-mlops-mlflow",
            key="azure-blob-storage-shared-access-key",
        )
        dbutils.fs.mount(
            source=f"wasbs://{storage_container_name}@{storage_account_name}.blob.core.windows.net",  # noqa: E501
            mount_point=taxi_fares_mount_point,
            extra_configs={
                f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_shared_key_name  # noqa: E501
            },
        )
except Exception as ex:
    print(ex)
    mlflow.end_run()
    shutil.rmtree(mlflow_log_tmp_dir, ignore_errors=True)
    logger.exception(f"ERROR - in mounting adls - {ex}")
    raise Exception(f"ERROR - in mounting adls - {ex}") from ex

# COMMAND ----------

# Clean up function


def clean():
    dbutils.fs.unmount(taxi_fares_mount_point)
    mlflow.log_artifacts(mlflow_log_tmp_dir)
    shutil.rmtree(mlflow_log_tmp_dir)
    mlflow.end_run()


# COMMAND ----------

# Get batch scoring raw data
try:
    logger.info("Reading training raw data")
    raw_data_file = taxi_fares_raw_data
    raw_data = spark.read.format("delta").load(raw_data_file)
    mlflow.log_param("data_raw_rows", raw_data.count())
    mlflow.log_param("data_raw_cols", len(raw_data.columns))
except Exception as ex:
    clean()
    logger.exception(f"ERROR - in reading raw data - {ex}")
    raise Exception(f"ERROR - in reading raw data - {ex}") from ex

# COMMAND ----------


# Run feature engineering on batch scoring raw data
if execute_feature_engineering == "true":
    try:
        logger.info("Starting feature engineering")
        with tracer.span("run_feature_engineering"):
            feature_engineered_data = run_feature_engineering(
                df_input=raw_data,
                start_date=datetime.strptime(scoring_data_start_date, "%Y-%m-%d"),
                end_date=datetime.strptime(scoring_data_end_date, "%Y-%m-%d"),
                mlflow=mlflow,
                mlflow_log_tmp_dir=mlflow_log_tmp_dir,
                explain_features=True,
                app_logger=app_logger,
                parent_tracer=tracer,
            )
    except Exception as ex:
        clean()
        logger.exception(f"ERROR - in feature engineering - {ex}")
        raise Exception(f"ERROR - in feature engineering - {ex}") from ex
else:
    logger.info("Skipping feature engineering")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS feature_store_taxi_example;

# COMMAND ----------

# Save features to feature store
fs = feature_store.FeatureStoreClient()
if execute_feature_engineering == "true":
    try:
        spark.conf.set("spark.sql.shuffle.partitions", "5")

        fs.create_table(
            name="feature_store_taxi_example.trip_pickup_features",
            primary_keys=["zip", "ts"],
            df=feature_engineered_data[0],
            partition_columns="yyyy_mm",
            description="Taxi Fares. Pickup Features",
        )
        fs.create_table(
            name="feature_store_taxi_example.trip_dropoff_features",
            primary_keys=["zip", "ts"],
            df=feature_engineered_data[1],
            partition_columns="yyyy_mm",
            description="Taxi Fares. Dropoff Features",
        )

        # Write the pickup features DataFrame to the feature store table
        fs.write_table(
            name="feature_store_taxi_example.trip_pickup_features",
            df=feature_engineered_data[0],
            mode="merge",
        )
        # Write the dropoff features DataFrame to the feature store table
        fs.write_table(
            name="feature_store_taxi_example.trip_dropoff_features",
            df=feature_engineered_data[1],
            mode="merge",
        )
    except Exception as ex:
        clean()
        logger.exception(f"ERROR - in feature saving into feature store - {ex}")
        raise Exception(f"ERROR - in feature saving into feature store - {ex}") from ex
else:
    logger.info("Skipping feature saving into feature store")

# COMMAND ----------

# Batch scoring
try:
    logger.info("Starting batch scoring")
    with tracer.span("run_scoring_batch"):
        run_scoring_batch(
            trained_model_name="taxi_fares",
            score_df=rounded_taxi_data(raw_data),
            mlflow=mlflow,
            mlflow_log_tmp_dir=mlflow_log_tmp_dir,
            trained_model_version=trained_model_version,
            app_logger=app_logger,
            parent_tracer=tracer,
        )
except Exception as ex:
    clean()
    logger.exception(f"ERROR - in batch scoring - {ex}")
    raise Exception(f"ERROR - in batch scoring - {ex}") from ex


# COMMAND ----------

# Batch scoring result publish
try:
    logger.info("Starting batch scoring result publish to adls")
    with tracer.span("run_scoring_batch"):
        result_path = "/".join(
            [
                "/dbfs",
                taxi_fares_mount_point,
                "batch_scoring_result",
                str(mlflow_run_id),
            ]
        )
        Path(result_path).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            os.path.join(mlflow_log_tmp_dir, "batch_scoring_result.html"),
            os.path.join(
                result_path,
                "batch_scoring_result.html",
            ),
        )
        shutil.copyfile(
            os.path.join(mlflow_log_tmp_dir, "batch_scoring_result.csv"),
            os.path.join(
                result_path,
                "batch_scoring_result.csv",
            ),
        )
        logger.info(f"Published score result in {result_path}")
except Exception as ex:
    clean()
    logger.exception(f"ERROR - in batch scoring result publish to adls - {ex}")
    raise Exception(f"ERROR - in batch scoring result publish to adls - {ex}") from ex


# COMMAND ----------

# End
logger.info(f"Completed batch scoring with mlflow run id {mlflow_run_id}")
clean()
