from typing import Any, Dict

import numpy as np
from sklearn.linear_model import Ridge


def train_model(train_data: Dict[str, np.ndarray], ridge_args: Dict[str, Any]) -> Ridge:
    """Train the model, return the model.

    Args:
        train_data (Dict[np.array]): training data with X key for
                                     features and y key labels
        ridge_args (Dict[Any]): ridge classifier arguments

    Returns:
        Ridge: trained ridge model
    """
    reg_model = Ridge(**ridge_args)
    reg_model.fit(train_data["X"], train_data["y"])
    return reg_model
