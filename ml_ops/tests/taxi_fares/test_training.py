import logging
import os
import unittest
from unittest.mock import MagicMock

import lightgbm as lgb
import pandas as pd
from pyspark.sql import SparkSession
from taxi_fares_mlops.training import run


class TestEvaluateMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[*]").appName("Unit-tests").getOrCreate()
        )

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s %(module)s %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )

    def test_training(self):
        self.logger.info("unittest test_training")
        data_file = os.path.join(
            "tests/taxi_fares/data", "taxi_fares_unit_test_training.csv"
        )
        train_df_pandas = pd.read_csv(data_file)
        train_df = self.spark.createDataFrame(train_df_pandas)
        model = run(train_df, MagicMock())

        assert isinstance(model, lgb.Booster)

    def test_training_exception(self):
        self.logger.info("unittest test_training exception")
        with self.assertRaises(Exception):
            model = run(MagicMock(), MagicMock())
            assert model is not None


if __name__ == "__main__":
    unittest.main()
