import logging
import unittest
from unittest.mock import MagicMock

from diabetes_mlops.publish_model import run


class TestEvaluateMethods(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s %(module)s %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )

    def test_publish_model(self):
        self.logger.info("unittest test_publish_model")
        run(MagicMock(), MagicMock())
        assert True


if __name__ == "__main__":
    unittest.main()
