import logging
import os
import unittest

import pandas as pd
from diabetes_mlops.feature_engineering import run


class TestEvaluateMethods(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s %(module)s %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )

    def test_feature_engineering(self):
        self.logger.info("unittest test_feature_engineering")
        data_file = os.path.join("tests/diabetes/data", "diabetes_unit_test.csv")
        df_input = pd.read_csv(data_file)
        df_output = run(df_input)

        assert df_input.shape[0] == df_output.shape[0]
        assert df_input.shape[1] + 1 == df_output.shape[1]

        df_output_numeric = df_output.select_dtypes(include=["float64", "int64"])
        assert df_output_numeric.shape == df_output.shape


if __name__ == "__main__":
    unittest.main()
