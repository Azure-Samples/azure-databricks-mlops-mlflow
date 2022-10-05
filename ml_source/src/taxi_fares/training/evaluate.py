from typing import Dict

import lightgbm as lgb
import numpy as np
from pyspark.sql.dataframe import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def split_data(df: DataFrame) -> dict:
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
    features_and_label = df.columns

    # Collect data into a Pandas array for training
    data = df.toPandas()[features_and_label]

    train, test = train_test_split(data, random_state=123)
    X_train = train.drop(["fare_amount"], axis=1)
    y_train = train.fare_amount
    X_test = test.drop(["fare_amount"], axis=1)
    y_test = test.fare_amount

    data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}
    return data


def get_model_metrics(model: lgb.Booster, test_data: Dict[str, np.ndarray]) -> dict:
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
