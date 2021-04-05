import os
import unittest

import pandas as pd
from diabetes.scoring.batch.run import batch_scoring
from diabetes.training.evaluate import split_data
from diabetes.training.train import train_model


class TestScoringBatchMethods(unittest.TestCase):
    def test_batch_scoring(self):
        ridge_args = {"alpha": 0.5}
        data_file = os.path.join("tests/diabetes/data", "diabetes_unit_test.csv")
        train_df = pd.read_csv(data_file).drop(columns=["SEX"])
        data = split_data(train_df)
        model = train_model(data["train"], ridge_args)

        score_data_file = os.path.join("tests/diabetes/data", "scoring_dataset.csv")
        score_df = pd.read_csv(score_data_file).drop(columns=["SEX"])
        scores = batch_scoring(model, score_df)
        self.assertAlmostEqual(scores[0], 60.75743442)
        self.assertAlmostEqual(scores[1], 67.10061271)
