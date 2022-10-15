import unittest
from unittest.mock import patch

from taxi_fares_mlops.utils import get_latest_model_version


class TestUtils(unittest.TestCase):
    @patch("taxi_fares_mlops.utils.MlflowClient")
    def test_get_latest_model_version(self, mock_mlflow_client):
        assert get_latest_model_version("taxi_fares") == 1
