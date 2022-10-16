# Databricks notebook source
"""Orchestrator notebook for diabetes training."""
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
dbutils.widgets.text("mlflow_experiment_id", "")
dbutils.widgets.text("wheel_package_dbfs_base_path", "")
dbutils.widgets.text("wheel_package_taxi_fares_version", "")
dbutils.widgets.text("wheel_package_taxi_fares_mlops_version", "")
dbutils.widgets.text("execute_feature_engineering", "true")
dbutils.widgets.text("training_data_start_date", "2016-01-01")
dbutils.widgets.text("training_data_end_date", "2016-01-31")
dbutils.widgets.text("training_num_leaves", "32")
dbutils.widgets.text("training_objective", "regression")
dbutils.widgets.text("training_metric", "rmse")
dbutils.widgets.text("training_num_rounds", "100")

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
import shutil  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import mlflow  # noqa: E402
from databricks import feature_store  # noqa: E402
from databricks.feature_store import FeatureLookup  # noqa: E402
from monitoring.app_logger import AppLogger, get_disabled_logger  # noqa: E402
from taxi_fares.utils.pyspark_utils import rounded_taxi_data  # noqa: E402
from taxi_fares_mlops.feature_engineering import run as run_feature_engineering  # noqa
from taxi_fares_mlops.publish_model import run as run_publish_model  # noqa: E402
from taxi_fares_mlops.training import run as run_training  # noqa: E402

# COMMAND ----------

# Get other parameters
mlflow_experiment_id = dbutils.widgets.get("mlflow_experiment_id")
execute_feature_engineering = dbutils.widgets.get("execute_feature_engineering")
training_data_start_date = dbutils.widgets.get("training_data_start_date")
training_data_end_date = dbutils.widgets.get("training_data_end_date")
taxi_fares_raw_data = dbutils.widgets.get("taxi_fares_raw_data")
training_num_leaves = int(dbutils.widgets.get("training_num_leaves"))
training_objective = dbutils.widgets.get("training_objective")
training_metric = dbutils.widgets.get("training_metric")
training_num_rounds = int(dbutils.widgets.get("training_num_rounds"))

# COMMAND ----------

# Initiate mlflow experiment
mlflow.start_run(experiment_id=int(mlflow_experiment_id), run_name="training")
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
        component_name="Train_Orchestrator",
        custom_dimensions={
            "mlflow_run_id": mlflow_run_id,
            "mlflow_experiment_id": int(mlflow_experiment_id),
        },
    )
    tracer = app_logger.get_tracer(
        component_name="Train_Orchestrator",
    )
except Exception as ex:
    print(ex)
    mlflow.end_run()
    shutil.rmtree(mlflow_log_tmp_dir, ignore_errors=True)
    raise Exception(f"ERROR - in initializing app logger - {ex}") from ex

logger.info(f"Stating training with mlflow run id {mlflow_run_id}")

# COMMAND ----------

# Clean up function


def clean():
    mlflow.log_artifacts(mlflow_log_tmp_dir)
    shutil.rmtree(mlflow_log_tmp_dir)
    mlflow.end_run()


# COMMAND ----------

# Get training raw data
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

# Run feature engineering
if execute_feature_engineering == "true":
    try:
        logger.info("Starting feature engineering")
        with tracer.span("run_feature_engineering"):
            feature_engineered_data = run_feature_engineering(
                df_input=raw_data,
                start_date=datetime.strptime(training_data_start_date, "%Y-%m-%d"),
                end_date=datetime.strptime(training_data_end_date, "%Y-%m-%d"),
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

# Load features from feature store
try:
    pickup_features_table = "feature_store_taxi_example.trip_pickup_features"
    dropoff_features_table = "feature_store_taxi_example.trip_dropoff_features"

    pickup_feature_lookups = [
        FeatureLookup(
            table_name=pickup_features_table,
            feature_names=[
                "mean_fare_window_1h_pickup_zip",
                "count_trips_window_1h_pickup_zip",
            ],
            lookup_key=["pickup_zip", "rounded_pickup_datetime"],
        ),
    ]

    dropoff_feature_lookups = [
        FeatureLookup(
            table_name=dropoff_features_table,
            feature_names=["count_trips_window_30m_dropoff_zip", "dropoff_is_weekend"],
            lookup_key=["dropoff_zip", "rounded_dropoff_datetime"],
        ),
    ]

    # unless additional feature engineering was performed,
    # exclude them to avoid training on them.
    exclude_columns = ["rounded_pickup_datetime", "rounded_dropoff_datetime"]

    # Create the training set that includes the raw input data merged with
    # corresponding features from both feature tables
    training_set = fs.create_training_set(
        rounded_taxi_data(raw_data),
        feature_lookups=pickup_feature_lookups + dropoff_feature_lookups,
        label="fare_amount",
        exclude_columns=exclude_columns,
    )

    # Load the TrainingSet into a dataframe which can be passed into
    # sklearn for training a model
    training_df = training_set.load_df()

    logger.info(
        f"Shape of training dataframe, rows: {training_df.count()}, cols: {len(training_df.columns)}"  # noqa: E501
    )
    mlflow.log_param("training_data_rows", training_df.count())
    mlflow.log_param("training_data_columns", len(training_df.columns))
except Exception as ex:
    clean()
    logger.exception(f"ERROR - in feature loading from feature store - {ex}")
    raise Exception(f"ERROR - in feature loading from feature store - {ex}") from ex

# COMMAND ----------

# Run training
try:
    logger.info("Starting model training")
    params = {
        "num_leaves": training_num_leaves,
        "objective": training_objective,
        "metric": training_metric,
    }
    num_rounds = training_num_rounds
    with tracer.span("run_training"):
        trained_model = run_training(
            training_df,
            mlflow,
            params=params,
            num_rounds=num_rounds,
            app_logger=app_logger,
            parent_tracer=tracer,
        )
except Exception as ex:
    clean()
    logger.exception(f"ERROR - in model training - {ex}")
    raise Exception(f"ERROR - in model training - {ex}") from ex

# COMMAND ----------

# Publish trained model
try:
    logger.info("Starting publish model")
    with tracer.span("run_publish_model"):
        run_publish_model(
            trained_model=trained_model,
            training_set=training_set,
            mlflow=mlflow,
            model_name="taxi_fares",
            app_logger=app_logger,
            parent_tracer=tracer,
        )
except Exception as ex:
    clean()
    logger.exception(f"ERROR - in publish trained model - {ex}")
    raise Exception(f"ERROR - in publish trained model - {ex}") from ex

# COMMAND ----------

# End
logger.info(f"Completed training with mlflow run id {mlflow_run_id}")
clean()
