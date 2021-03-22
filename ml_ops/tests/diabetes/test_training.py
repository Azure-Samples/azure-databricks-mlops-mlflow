from diabetes_mlops.training import training
import unittest
import os
import pandas as pd

class TestEvaluateMethods(unittest.TestCase):
    def test_get_model_metrics(self):
        data_file = os.path.join('tests/diabetes/data', 'diabetes_unit_test.csv')
        train_df = pd.read_csv(data_file).drop(columns=['SEX'])
        training(train_df)

if __name__ == '__main__':
    unittest.main()