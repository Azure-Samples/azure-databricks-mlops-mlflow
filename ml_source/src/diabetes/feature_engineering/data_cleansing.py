import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def read_data(csv_path):
    """Read CSV file and returns a dataframe

    Args:
        csv_path ([str]): [Path of csv file]

    Returns:
        [pd.DataFrame]: [Pandas Dataframe]
    """
    print(f"clean_data: path: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Shape of dataframe, rows: {df.shape[0]}, cols: {df.shape[1]}")
    return df


def cal_missing_values(df: pd.DataFrame):
    """Function to print % of missing values

    Args:
        df (pd.DataFrame): [Pandas dataframe]
    """
    total_cells = np.product(df.shape)
    missing_values_count = df.isnull().sum()
    total_missing = missing_values_count.sum()
    percentage_missing = (total_missing / total_cells) * 100
    return percentage_missing


def fill_missing_values(df: pd.DataFrame):
    """Fill missing values with mean

    Args:
        df (pd.DataFrame): [Pandas DataFrame]

    Returns:
        [pd.DataFrame]: [Returns processed dataframe]
    """
    filled_df = df.fillna(df.mean())
    return filled_df


def remove_categorical_column(df: pd.DataFrame):
    """Removes Categorical Column in raw data

    Args:
        df (pd.DataFrame): [Input Pandas DataFrame]

    Returns:
        [pd.DataFrame]: [Returns processed dataframe]
    """
    ohe = OneHotEncoder(sparse=False)
    transform_df = ohe.fit_transform(df["SEX"].values.reshape(-1, 1))
    sex_cat = ["MALE", "FEMALE"]
    df_one_hot = pd.DataFrame(
        transform_df, columns=[sex_cat[i] for i in range(len(sex_cat))]
    )
    complete_df = pd.concat([df, df_one_hot], axis=1).drop(["SEX"], axis=1)
    return complete_df
