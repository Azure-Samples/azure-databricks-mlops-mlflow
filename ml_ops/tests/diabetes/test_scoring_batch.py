import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import pandas as pd
from diabetes_mlops.scoring_batch import run

from diabetes.training.evaluate import split_data
from diabetes.training.train import train_model


class TestEvaluateMethods(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s %(module)s %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )

    def test_scoring_batch(self):
        self.logger.info("unittest test_scoring_batch")
        ridge_args = {"alpha": 0.5}
        data_file = os.path.join(
            "tests/diabetes/data", "diabetes_unit_test_training.csv"
        )
        score_df = pd.read_csv(data_file)
        data = split_data(score_df)
        model = train_model(data["train"], ridge_args)

        score_df = score_df.drop(columns=["Y"])
        run(model, score_df, MagicMock(), tempfile.mkdtemp())
        assert True

    def test_scoring_batch_exception(self):
        self.logger.info("unittest test_scoring_batch exception")
        with self.assertRaises(Exception):
            run(MagicMock(), MagicMock(), MagicMock(), tempfile.mkdtemp())
            assert True


if __name__ == "__main__":
    unittest.main()
