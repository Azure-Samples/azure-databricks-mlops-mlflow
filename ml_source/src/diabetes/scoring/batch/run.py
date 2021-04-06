import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def batch_scoring(model: Ridge, df: pd.DataFrame) -> np.array:
    """[Batch scoring method]

    Args:
        model (Ridge): [Model]
        df (pd.DataFrame): [Input dataframe for prediction]

    Returns:
        np.array : Returns predicted values, shape (n_samples,)
    """
    result = model.predict(df.values)
    return result
