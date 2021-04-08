import logging
import unittest
from unittest.mock import MagicMock, patch

import mlflow
from diabetes_mlops.load_model import run
from sklearn.linear_model import Ridge


class TestEvaluateMethods(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s %(module)s %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )

    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.sklearn.load_model")
    @patch("mlflow.log_param")
    @patch("mlflow.active_run")
    def test_load_model(
        self,
        mock_mlflow_active_run,
        mock_mlflow_log_param,
        mock_mlflow_load_model,
        mock_mlflow_client,
    ):
        self.logger.info("unittest test_load_model")
        mock_mlflow_active_run.return_value = MagicMock()
        mock_mlflow_load_model.return_value = Ridge()
        mock_mlflow_client.return_value.get_latest_versions.return_value = [MagicMock()]
        model = run(mlflow)
        assert isinstance(model, Ridge)

    def test_load_model_none_version(self):
        self.logger.info("unittest test_load_model")
        model = run(MagicMock())
        assert model is None

    def test_load_model_exception(self):
        self.logger.info("unittest test_load_model exception")
        with self.assertRaises(Exception):
            run(None)
            assert True


if __name__ == "__main__":
    unittest.main()
