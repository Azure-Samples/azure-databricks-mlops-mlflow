import os
import unittest

import pandas as pd
from diabetes.training.evaluate import get_model_metrics, split_data
from diabetes.training.train import train_model


class TestEvaluateMethods(unittest.TestCase):
    def test_get_model_metrics(self):
        ridge_args = {"alpha": 0.5}
        data_file = os.path.join("tests/diabetes/data", "diabetes_unit_test.csv")
        train_df = pd.read_csv(data_file).drop(columns=["SEX"])
        data = split_data(train_df)
        model = train_model(data["train"], ridge_args)
        metrics = get_model_metrics(model, data["test"])
        self.assertEqual(len(metrics), 1)


if __name__ == "__main__":
    unittest.main()
