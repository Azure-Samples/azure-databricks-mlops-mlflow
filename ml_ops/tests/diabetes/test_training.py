import logging
import os
import unittest
from unittest.mock import MagicMock

import pandas as pd
from diabetes_mlops.training import run
from sklearn.linear_model import Ridge


class TestEvaluateMethods(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s %(module)s %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )

    def test_training(self):
        self.logger.info("unittest test_training")
        data_file = os.path.join(
            "tests/diabetes/data", "diabetes_unit_test_training.csv"
        )
        train_df = pd.read_csv(data_file)
        model = run(train_df, MagicMock())

        assert isinstance(model, Ridge)

    def test_training_exception(self):
        self.logger.info("unittest test_training exception")
        with self.assertRaises(Exception):
            model = run(MagicMock(), MagicMock())
            assert model is not None


if __name__ == "__main__":
    unittest.main()
