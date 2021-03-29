import os
import unittest

import pandas as pd
from diabetes.training.evaluate import split_data
from diabetes.training.train import train_model
from sklearn.linear_model import Ridge


class TestTrainMethods(unittest.TestCase):
    def test_train_model(self):
        ridge_args = {"alpha": 0.5}
        data_file = os.path.join("tests/diabetes/data", "diabetes_unit_test.csv")
        train_df = pd.read_csv(data_file).drop(columns=["SEX"])
        data = split_data(train_df)
        model = train_model(data["train"], ridge_args)
        self.assertIsInstance(model, Ridge)


if __name__ == "__main__":
    unittest.main()
