from datetime import datetime

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, count, mean, to_timestamp, unix_timestamp, window
from pyspark.sql.types import FloatType, IntegerType
from taxi_fares.utils.pyspark_utils import filter_df_by_ts, is_weekend, partition_id


def pickup_features_fn(
    df: DataFrame, ts_column: str, start_date: datetime, end_date: datetime
) -> DataFrame:
    """
    Computes the pickup_features feature group.
    To restrict features to a time range, pass in ts_column, start_date,
    and/or end_date as kwargs.
    """
    df = filter_df_by_ts(df, ts_column, start_date, end_date)
    pickupzip_features = (
        df.groupBy(
            "pickup_zip", window("tpep_pickup_datetime", "1 hour", "15 minutes")
        )  # 1 hour window, sliding every 15 minutes
        .agg(
            mean("fare_amount").alias("mean_fare_window_1h_pickup_zip"),
            count("*").alias("count_trips_window_1h_pickup_zip"),
        )
        .select(
            col("pickup_zip").alias("zip"),
            unix_timestamp(col("window.end")).alias("ts").cast(IntegerType()),
            partition_id(to_timestamp(col("window.end"))).alias("yyyy_mm"),
            col("mean_fare_window_1h_pickup_zip").cast(FloatType()),
            col("count_trips_window_1h_pickup_zip").cast(IntegerType()),
        )
    )
    return pickupzip_features


def dropoff_features_fn(
    df: DataFrame, ts_column: str, start_date: datetime, end_date: datetime
) -> DataFrame:
    """
    Computes the dropoff_features feature group.
    To restrict features to a time range, pass in ts_column, start_date,
    and/or end_date as kwargs.
    """
    df = filter_df_by_ts(df, ts_column, start_date, end_date)
    dropoffzip_features = (
        df.groupBy("dropoff_zip", window("tpep_dropoff_datetime", "30 minute"))
        .agg(count("*").alias("count_trips_window_30m_dropoff_zip"))
        .select(
            col("dropoff_zip").alias("zip"),
            unix_timestamp(col("window.end")).alias("ts").cast(IntegerType()),
            partition_id(to_timestamp(col("window.end"))).alias("yyyy_mm"),
            col("count_trips_window_30m_dropoff_zip").cast(IntegerType()),
            is_weekend(col("window.end")).alias("dropoff_is_weekend"),
        )
    )
    return dropoffzip_features
