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
dbutils.widgets.text("diabetes_mount_point", "/mnt/data")
dbutils.widgets.text("diabetes_training_data_file", "data_raw.csv")
dbutils.widgets.text("mlflow_experiment_id", "")
dbutils.widgets.text("wheel_package_dbfs_base_path", "")
dbutils.widgets.text("wheel_package_diabetes_version", "")
dbutils.widgets.text("wheel_package_diabetes_mlops_version", "")

# COMMAND ----------

# Get wheel package parameters
wheel_package_dbfs_base_path = dbutils.widgets.get("wheel_package_dbfs_base_path")
wheel_package_diabetes_version = dbutils.widgets.get("wheel_package_diabetes_version")
wheel_package_diabetes_mlops_version = dbutils.widgets.get(
    "wheel_package_diabetes_mlops_version"
)

# COMMAND ----------

# MAGIC %pip install $wheel_package_dbfs_base_path/diabetes-$wheel_package_diabetes_version-py3-none-any.whl # noqa: E501
# MAGIC %pip install $wheel_package_dbfs_base_path/diabetes_mlops-$wheel_package_diabetes_mlops_version-py3-none-any.whl # noqa: E501

# COMMAND ----------

# Imports
import logging  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from diabetes_mlops.feature_engineering import run as run_feature_engineering  # noqa
from diabetes_mlops.publish_model import run as run_publish_model  # noqa: E402
from diabetes_mlops.training import run as run_training  # noqa: E402

# COMMAND ----------

# Get other parameters
mlflow_experiment_id = dbutils.widgets.get("mlflow_experiment_id")
diabetes_mount_point = dbutils.widgets.get("diabetes_mount_point")
diabetes_training_data_file = dbutils.widgets.get("diabetes_training_data_file")

# COMMAND ----------

# Initiate mlflow experiment
mlflow.start_run(experiment_id=int(mlflow_experiment_id), run_name="training")
mlflow_run = mlflow.active_run()
mlflow_run_id = mlflow_run.info.run_id
mlflow_log_tmp_dir = "/tmp/" + str(mlflow_run_id)  # nosec: B108
Path(mlflow_log_tmp_dir).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    format="%(asctime)s %(module)s %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logging.getLogger("py4j").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.info(f"Stating training with mlflow run id {mlflow_run_id}")

# COMMAND ----------

# Mount ADLS Gen2 storage container
logger.info(f"Mounting {diabetes_mount_point}")
if any(mount.mountPoint == diabetes_mount_point for mount in dbutils.fs.mounts()):
    logger.info(f"Mount point exists {diabetes_mount_point}")
else:
    storage_account_name = dbutils.secrets.get(
        scope="azure-databricks-mlops-mlflow", key="azure-blob-storage-account-name"
    )
    storage_container_name = dbutils.secrets.get(
        scope="azure-databricks-mlops-mlflow", key="azure-blob-storage-container-name"
    )
    storage_shared_key_name = dbutils.secrets.get(
        scope="azure-databricks-mlops-mlflow",
        key="azure-blob-storage-shared-access-key",
    )
    dbutils.fs.mount(
        source=f"wasbs://{storage_container_name}@{storage_account_name}.blob.core.windows.net",  # noqa: E501
        mount_point=diabetes_mount_point,
        extra_configs={
            f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_shared_key_name  # noqa: E501
        },
    )


# COMMAND ----------

# Get training raw data
logger.info("Reading training raw data")
raw_data_file = "/dbfs/" + diabetes_mount_point + "/" + diabetes_training_data_file
raw_data = pd.read_csv(raw_data_file)
mlflow.log_param("data_raw_rows", raw_data.shape[0])
mlflow.log_param("data_raw_cols", raw_data.shape[1])

# COMMAND ----------

# Run feature engineering
logger.info("Stating feature engineering")
feature_engineered_data = run_feature_engineering(
    df_input=raw_data, mlflow=mlflow, mlflow_log_tmp_dir=mlflow_log_tmp_dir
)
mlflow.log_param("data_feature_engineered_rows", feature_engineered_data.shape[0])
mlflow.log_param("data_feature_engineered_cols", feature_engineered_data.shape[1])

# COMMAND ----------

# Run training
trained_model = run_training(feature_engineered_data, mlflow)

# COMMAND ----------

# Publish trained model
run_publish_model(trained_model=trained_model, mlflow=mlflow, model_name="diabetes")

# COMMAND ----------

# End
logger.info(f"Completed training with mlflow run id {mlflow_run_id}")
dbutils.fs.unmount(diabetes_mount_point)
mlflow.log_artifacts(mlflow_log_tmp_dir)
shutil.rmtree(mlflow_log_tmp_dir)
mlflow.end_run()
