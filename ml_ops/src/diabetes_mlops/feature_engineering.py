import logging

import pandas as pd
from diabetes.feature_engineering.data_cleansing import (
    cal_missing_values,
    fill_missing_values,
    remove_categorical_column,
)


def run(df_input: pd.DataFrame) -> pd.DataFrame:
    """MLOps feature engineering entry point.

    Args:
        df_input (pd.DataFrame): input data - raw

    Returns:
        pd.DataFrame: clean and feature enginered data
    """
    logging.info("Running MLOps feature engineering")

    logging.info(
        f"Shape of input dataframe, rows: {df_input.shape[0]}, cols: {df_input.shape[1]}"  # noqa: E501
    )

    percentage_missing = cal_missing_values(df_input)
    logging.info(f"Percentage of Missing Values: {percentage_missing}")

    logging.info("Filling missing values")
    df_output = fill_missing_values(df_input)

    logging.info("One-hot-encoding of categorical columns")
    df_output = remove_categorical_column(df_output)

    logging.info(
        f"Shape of feature enginered dataframe, rows: {df_output.shape[0]}, cols: {df_output.shape[1]}"  # noqa: E501
    )

    logging.info("Completed MLOps feature engineering")
    return df_output
