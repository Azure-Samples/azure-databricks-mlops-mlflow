import unittest

from pyspark.sql import SparkSession
from src.taxi_fares.utils.pyspark_utils import filter_df_by_ts


class TestPysparkUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[*]").appName("Unit-tests").getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_if_df_is_getting_filtered_by_ts(self):
        df = self.spark.createDataFrame(
            [
                ("2019-01-01 00:00:00", 1),
                ("2019-01-01 00:15:00", 2),
                ("2019-01-01 00:30:00", 3),
                ("2019-01-01 00:45:00", 4),
                ("2019-01-01 01:00:00", 5),
                ("2019-01-01 01:15:00", 6),
                ("2019-01-01 01:30:00", 7),
                ("2019-01-01 01:45:00", 8),
                ("2019-01-01 02:00:00", 9),
                ("2019-01-01 02:15:00", 10),
                ("2019-01-01 02:30:00", 11),
                ("2019-01-01 02:45:00", 12),
            ],
            ["tpep_pickup_datetime", "fare_amount"],
        )
        df = filter_df_by_ts(
            df, "tpep_pickup_datetime", "2019-01-01 00:00:00", "2019-01-01 01:45:00"
        )
        self.assertEqual(df.count(), 7)
