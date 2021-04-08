"""Test src/monitoring/app_logger.py."""

import logging
import unittest
import uuid

from monitoring.app_logger import AppLogger, get_disabled_logger

test_instrumentation_key = str(uuid.uuid1())
test_invalid_instrumentation_key = "invalid_instrumentation_key"


class TestAppLogger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid_config = {
            "log_level": "DEBUG",
            "logging_enabled": "true",
            "app_insights_key": test_instrumentation_key,
        }
        cls.invalid_config = {
            "log_level": "DEBUG",
            "logging_enabled": "false",
            "app_insights_key": test_invalid_instrumentation_key,
        }

    def test_logger_creation_valid_instrumentation_key(self):
        """Test with valid formatted instrumentation key."""
        global test_instrumentation_key
        try:
            app_logger = AppLogger(
                config=self.valid_config,
            )
            assert app_logger is not None
        except Exception:
            assert False

    def test_logger_creation_invalid_instrumentation_key(self):
        """Test with invalid instrumentation key."""
        global test_invalid_instrumentation_key
        with self.assertRaises(Exception):
            logging.disable(logging.CRITICAL)
            app_logger = AppLogger(
                config=self.invalid_config,
            )
            app_logger.get_logger()
            assert app_logger is not None

    def test_logger_creation_no_instrumentation_key(self):
        """Test with no instrumentation key."""
        with self.assertRaises(Exception):
            logging.disable(logging.CRITICAL)
            config = {"log_level": logging.DEBUG, "logging_enabled": "false"}
            app_logger = AppLogger(config=config)
            app_logger.get_logger()
            assert app_logger is not None

    def test_logging(self):
        """Test to use logging functions."""
        global test_instrumentation_key
        try:
            component_name = "TestComponent"
            app_logger = AppLogger(config=self.valid_config)
            assert app_logger is not None
            test_logger = app_logger.get_logger(
                component_name=component_name,
            )

            assert test_logger is not None
            test_logger.info("Test Logging")
        except Exception:
            assert False

    def test_tracing(self):
        """Test for Tracer."""
        global test_instrumentation_key
        try:
            component_name = "TestComponent"
            app_logger = AppLogger(config=self.valid_config)
            assert app_logger is not None

            tracer = app_logger.get_tracer(
                component_name=component_name,
            )
            tracer_with_parent = app_logger.get_tracer(
                component_name=component_name, parent_tracer=tracer
            )
            test_logger = app_logger.get_logger(
                component_name=component_name,
            )

            assert test_logger is not None
            assert tracer is not None
            assert tracer_with_parent is not None

            with tracer.span(name="testspan"):
                test_logger.info("in test span")
        except Exception:
            assert False

    def test_tracing_with_disabled_logger(self):
        """Test with no instrumentation key."""
        app_logger = get_disabled_logger()
        tracer = app_logger.get_tracer()
        assert tracer is not None

    def test_exception(self):
        """Test for calling logger.exception method."""
        global test_instrumentation_key
        try:
            component_name = "TestComponent"
            app_logger = AppLogger(
                config=self.valid_config,
            )
            assert app_logger is not None

            test_logger = app_logger.get_logger(
                component_name=component_name,
            )
            assert test_logger is not None
            try:
                raise Exception("Testing exception logging")
            except Exception as exp:
                test_logger.exception(exp)
        except Exception:
            assert False

    def test_logging_level(self):
        """Test for changing logger level in config."""
        try:
            global test_instrumentation_key
            component_name = "TestComponent"
            valid_config = self.valid_config.copy()
            valid_config["log_level"] = logging.ERROR
            app_logger = AppLogger(
                config=valid_config,
            )
            assert app_logger.config["log_level"] == logging.ERROR
            test_logger = app_logger.get_logger(
                component_name=component_name,
            )

            test_logger.error("Testing logging level")
        except Exception:
            assert False

    def test_logging_extra_params(self):
        """Test logging extra params."""
        try:
            global test_instrumentation_key
            component_name = "TestComponent"
            app_logger = AppLogger(
                config=self.valid_config,
            )
            test_logger = app_logger.get_logger(
                component_name=component_name,
            )
            extra_params = {"custom_dimensions": {"key1": "value1"}}
            test_logger.info("Logging extra params", extra=extra_params)
        except Exception:
            assert False

    def test_disabled_logger(self):
        """Test disabled logger."""
        try:

            def do_work(app_logger=get_disabled_logger()):
                component_name = "TestComponent"
                test_logger = app_logger.get_logger(
                    component_name=component_name,
                )
                extra_params = {"custom_dimensions": {"key1": "value1"}}
                test_logger.info("Logging extra params", extra=extra_params)

            do_work()
        except Exception:
            assert False


if __name__ == "__main__":
    unittest.main()
