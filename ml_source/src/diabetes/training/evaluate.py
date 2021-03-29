from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame) -> dict:
    """Split the dataframe into test and train data.

    Args:
        df (pd.DataFrame): processed dataframe for train and evaluate

    Returns:
        dict: splitted data for train and test -
                {
                    "train":
                        "X": np.array,
                        "y": np.array,
                    },
                    "test":
                        "X": np.array,
                        "y": np.array,
                    }
                }
    """
    X = df.drop("Y", axis=1).values
    y = df["Y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}
    return data


def get_model_metrics(model: Ridge, test_data: Dict[str, np.ndarray]) -> dict:
    """Evaluate the metrics for the model.

    Args:
        model (Ridge): trained ridge model
        test_data (Dict[np.array]): test data with X key for features and y key labels

    Returns:
        dict: mse metrics
    """
    preds = model.predict(test_data["X"])
    mse = mean_squared_error(preds, test_data["y"])
    metrics = {"mse": mse}
    return metrics
