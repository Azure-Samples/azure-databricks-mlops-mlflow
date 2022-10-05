import unittest

from pyspark.sql import SparkSession
from src.taxi_fares.feature_eng.features import pickup_features_fn


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[*]").appName("Unit-tests").getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_if_pickup_features_are_computed(self):
        df = self.spark.createDataFrame(
            [
                ("2019-01-01 00:00:00", "2019-01-01 01:00:00", 1.0, 1, 10000, 10001),
                ("2019-01-01 00:15:00", "2019-01-01 01:15:00", 2.0, 2, 10002, 10003),
                ("2019-01-01 00:30:00", "2019-01-01 01:30:00", 3.0, 3, 10004, 10005),
                ("2019-01-01 00:45:00", "2019-01-01 01:45:00", 4.0, 4, 10006, 10007),
                ("2019-01-01 01:00:00", "2019-01-01 02:00:00", 5.0, 5, 10008, 10009),
                ("2019-01-01 01:15:00", "2019-01-01 02:15:00", 6.0, 6, 10010, 10011),
                ("2019-01-01 01:30:00", "2019-01-01 02:30:00", 7.0, 7, 10012, 10013),
                ("2019-01-01 01:45:00", "2019-01-01 02:45:00", 8.0, 8, 10014, 10015),
                ("2019-01-01 02:00:00", "2019-01-01 03:00:00", 9.0, 9, 10016, 10017),
                ("2019-01-01 02:15:00", "2019-01-01 03:15:00", 10.0, 10, 10018, 10019),
                ("2019-01-01 02:30:00", "2019-01-01 03:30:00", 11.0, 11, 10020, 10021),
                ("2019-01-01 02:45:00", "2019-01-01 03:45:00", 12.0, 12, 10022, 10023),
            ],
            [
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "trip_distance",
                "fare_amount",
                "pickup_zip",
                "dropoff_zip",
            ],
        )
        df = pickup_features_fn(
            df, "tpep_pickup_datetime", "2019-01-01 00:00:00", "2019-01-01 01:45:00"
        )
        self.assertEqual(df.count(), 28)
