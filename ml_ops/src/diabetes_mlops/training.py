import os
import pandas as pd
from diabetes.training.train import train_model
from diabetes.training.evaluate import split_data
from diabetes.training.evaluate import get_model_metrics

def training():
    print("Running train.py")

    # Define training parameters
    ridge_args = {"alpha": 0.5}

    # Load the training data as dataframe
    data_dir = "data"
    data_file = os.path.join(data_dir, 'diabetes.csv')
    train_df = pd.read_csv(data_file)

    data = split_data(train_df)

    # Train the model
    model = train_model(data, ridge_args)

    # Log the metrics for the model
    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")