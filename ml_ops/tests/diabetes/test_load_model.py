import logging
import unittest
from unittest.mock import MagicMock

from diabetes_mlops.load_model import run


class TestEvaluateMethods(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s %(module)s %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )

    def test_load_model(self):
        self.logger.info("unittest test_load_model")
        run(MagicMock(), 1)
        assert True

    def test_load_model_none_version(self):
        self.logger.info("unittest test_load_model")
        run(MagicMock())
        assert True


if __name__ == "__main__":
    unittest.main()
