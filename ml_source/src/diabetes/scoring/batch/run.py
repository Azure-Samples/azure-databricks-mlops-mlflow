import pandas as pd
from sklearn.linear_model import Ridge


def batch_scoring(model: Ridge, df: pd.DataFrame) -> str:
    """[Batch scoring method]

    Args:
        model (Ridge): [Model]
        df (pd.DataFrame): [Input dataframe for prediction]

    Returns:
        str: [description]
    """
    # TODO -- Add data cleansing here
    # df = perform_data_cleansing(df)
    result = model.predict(df.values)
    return result
