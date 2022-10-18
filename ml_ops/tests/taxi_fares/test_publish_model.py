import logging
import unittest
from unittest.mock import MagicMock, patch

from taxi_fares_mlops.publish_model import run


class TestEvaluateMethods(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s %(module)s %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )

    @patch("taxi_fares_mlops.publish_model.feature_store")
    @patch("taxi_fares_mlops.publish_model.get_latest_model_version")
    def test_publish_model(self, mock_feature_store, mock_get_latest_model_version):
        self.logger.info("unittest test_publish_model")
        run(MagicMock(), MagicMock(), MagicMock())
        assert True

    def test_publish_model_exception(self):
        self.logger.info("unittest test_publish_model exception")
        with self.assertRaises(Exception):
            run(None, None, None)
            assert True


if __name__ == "__main__":
    unittest.main()
