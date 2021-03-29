import pandas as pd
from diabetes.training.evaluate import get_model_metrics, split_data
from diabetes.training.train import train_model


def training(train_df: pd.DataFrame) -> None:
    print("Running train.py")

    # Define training parameters
    ridge_args = {"alpha": 0.5}

    # data for train and test
    data = split_data(train_df)

    # Train the model
    model = train_model(data["train"], ridge_args)

    # Log the metrics for the model
    metrics = get_model_metrics(model, data["test"])
    for (k, v) in metrics.items():
        print(f"{k}: {v}")
