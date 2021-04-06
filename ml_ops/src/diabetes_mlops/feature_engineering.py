import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from diabetes.feature_engineering.data_cleansing import (
    cal_missing_values,
    fill_missing_values,
    remove_categorical_column,
)


def run(
    df_input: pd.DataFrame,
    mlflow: mlflow,
    mlflow_log_tmp_dir: str,
    explain_features: bool = True,
) -> pd.DataFrame:
    """MLOps feature engineering entry point.

    Args:
        df_input (pd.DataFrame): input data - raw
        mlflow (mlflow): mlflow object that is having an active run
                         initiated by mlflow.start_run
        mlflow_log_tmp_dir (str): directory for puting files to be logged
                                  in mlflow artifacts
        explain_features (bool, optional): explain features, possible only with
                                           training data. Defaults to True.

    Returns:
        pd.DataFrame: clean and feature enginered data
    """
    logger = logging.getLogger(__name__)
    logger.info("Running MLOps feature engineering")

    logger.info(
        f"Shape of input dataframe, rows: {df_input.shape[0]}, cols: {df_input.shape[1]}"  # noqa: E501
    )

    percentage_missing = cal_missing_values(df_input)
    logger.info(f"Percentage of Missing Values: {percentage_missing}")
    mlflow.log_param("feature_engineering_percentage_missing", percentage_missing)

    logger.info("Filling missing values")
    df_output = fill_missing_values(df_input)

    logger.info("One-hot-encoding of categorical columns")
    df_output = remove_categorical_column(df_output)

    logger.info(
        f"Shape of feature enginered dataframe, rows: {df_output.shape[0]}, cols: {df_output.shape[1]}"  # noqa: E501
    )

    if explain_features:
        logger.info("Getting feature explanations - statistics")
        feature_statistic = df_output.describe()
        feature_statistic.to_html(
            Path(
                mlflow_log_tmp_dir,
                "feature_statistics.html",
            ),
            justify="center",
            na_rep="",
        )
        logger.info("Getting feature explanations - box plot")
        numeric_cols = df_output.columns
        plot_data = df_output.copy()
        select_top_k = len(numeric_cols)
        n_col = 4
        n_row = math.ceil(select_top_k / n_col)
        s_col = 5
        s_row = 3
        fig, axs = plt.subplots(
            n_row, n_col, figsize=(s_col * n_col, s_row * n_row), sharey=False
        )
        axs = axs.flatten()
        for index, col in enumerate(numeric_cols[:select_top_k]):
            ax = sns.boxplot(x="Y", y=col, data=plot_data, ax=axs[index])
            ax.set(title=col, ylabel="")
        fig.tight_layout()
        fig.savefig(Path(mlflow_log_tmp_dir, "feature_boxplot.png"))

    logger.info("Completed MLOps feature engineering")
    return df_output
