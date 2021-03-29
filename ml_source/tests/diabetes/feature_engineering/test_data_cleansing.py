import unittest

from diabetes.feature_engineering.data_cleansing import (
    cal_missing_values,
    fill_missing_values,
    read_data,
    remove_categorical_column,
)


class TestFeatureEngMethods(unittest.TestCase):
    def test_read_csv(self):
        csv_path = "./tests/diabetes/data/diabetes_unit_test.csv"
        df = read_data(csv_path)
        print(f"Rows: {df.shape[0]}, cols: {df.shape[1]}")
        self.assertEqual(df.shape[1], 11)
        self.assertEqual(df.shape[0], 9)

    def test_cal_missing_values(self):
        csv_path = "./tests/diabetes/data/diabetes_unit_test.csv"
        df = read_data(csv_path)
        percentage_missing = cal_missing_values(df)
        self.assertEqual(percentage_missing, 0.0)

    def test_fill_missing_values(self):
        csv_path = "./tests/diabetes/data/diabetes_unit_test.csv"
        df = read_data(csv_path)
        processed_df = fill_missing_values(df)
        self.assertEqual(processed_df.shape[1], 11)

    def test_remove_categorical_column(self):
        csv_path = "./tests/diabetes/data/diabetes_unit_test.csv"
        df = read_data(csv_path)
        processed_df = remove_categorical_column(df)
        self.assertEqual(processed_df.shape[1], 12)


if __name__ == "__main__":
    unittest.main()
