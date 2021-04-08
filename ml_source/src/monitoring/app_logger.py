"""This module is used to log traces into Azure Application Insights."""
import logging
import uuid
from os import getenv

from opencensus.ext.azure.common import utils
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace import config_integration
from opencensus.trace.samplers import AlwaysOffSampler, AlwaysOnSampler
from opencensus.trace.tracer import Tracer


class CustomDimensionsFilter(logging.Filter):
    """Add custom-dimensions like run_id in each log by using filters."""

    def __init__(self, custom_dimensions=None):
        """Initialize CustomDimensionsFilter."""
        self.custom_dimensions = custom_dimensions or {}

    def filter(self, record):
        """Add the default custom_dimensions into the current log record."""
        dim = {**self.custom_dimensions, **getattr(record, "custom_dimensions", {})}
        record.custom_dimensions = dim
        return True


class AppLogger:
    """Logger wrapper that attach the handler to Application Insights."""

    HANDLER_NAME = "Azure Application Insights Handler"

    def __init__(self, config=None):
        """Create an instance of the Logger class.

        Args:
            config:([dict], optional):
                Contains the setting for logger {"log_level": logging.debug,"env":"dev",
                                    "app_insights_key":"<app insights key>"}
            parent:tracer([opencensus.trace.tracer], optional):
                Contains parent tracer required for setting coorelation.
        """
        self.config = {"log_level": logging.INFO, "logging_enabled": "true"}
        self.APPINSIGHTS_INSTRUMENTATION_KEY = "APPINSIGHTS_INSTRUMENTATION_KEY"
        self.update_config(config)
        pass

    def _initialize_azure_log_handler(self, component_name, custom_dimensions):
        """Initialize azure log handler."""
        # Adding logging to trace_integrations
        # This will help in adding trace and span ids to logs
        # https://github.com/census-instrumentation/opencensus-python/tree/master/contrib/opencensus-ext-logging

        config_integration.trace_integrations(["logging"])
        logging.basicConfig(
            format="%(asctime)s name=%(name)s level=%(levelname)s "
            "traceId=%(traceId)s spanId=%(spanId)s %(message)s"
        )
        app_insights_cs = "InstrumentationKey=" + self._get_app_insights_key()
        log_handler = AzureLogHandler(
            connection_string=app_insights_cs, export_interval=0.0
        )
        log_handler.add_telemetry_processor(self._get_callback(component_name))
        log_handler.name = self.HANDLER_NAME
        log_handler.addFilter(CustomDimensionsFilter(custom_dimensions))
        return log_handler

    def _initialize_azure_log_exporter(self, component_name):
        """Initialize azure log exporter."""
        app_insights_cs = "InstrumentationKey=" + self._get_app_insights_key()
        log_exporter = AzureExporter(
            connection_string=app_insights_cs, export_interval=0.0
        )
        log_exporter.add_telemetry_processor(self._get_callback(component_name))
        return log_exporter

    def _initialize_logger(self, log_handler, component_name):
        """Initialize Logger."""
        logger = logging.getLogger(component_name)
        logger.setLevel(self.log_level)
        if self.config.get("logging_enabled") == "true":
            if not any(x for x in logger.handlers if x.name == self.HANDLER_NAME):
                logger.addHandler(log_handler)
        return logger

    def get_logger(self, component_name="DiabetesMlOps", custom_dimensions={}):
        """Get Logger Object.

        Args:
            component_name (str, optional): Name of logger. Defaults to "DiabetesMlOps".
            custom_dimensions (dict, optional): {"key":"value"}
                                                to capture with every log.
                                                Defaults to {}.

        Returns:
            Logger: A logger.
        """
        self.update_config(self.config)
        handler = self._initialize_azure_log_handler(component_name, custom_dimensions)
        return self._initialize_logger(handler, component_name)

    def get_tracer(self, component_name="DiabetesMlOps", parent_tracer=None):
        """Get Tracer Object.

        Args:
            component_name (str, optional): Name of logger. Defaults to "DiabetesMlOps".
            parent_tracer([opencensus.trace.tracer], optional):
                Contains parent tracer required for setting coorelation.

        Returns:
            opencensus.trace.tracer: A Tracer.
        """
        self.update_config(self.config)
        sampler = AlwaysOnSampler()
        exporter = self._initialize_azure_log_exporter(component_name)
        if self.config.get("logging_enabled") != "true":
            sampler = AlwaysOffSampler()
        if parent_tracer is None:
            tracer = Tracer(exporter=exporter, sampler=sampler)
        else:
            tracer = Tracer(
                span_context=parent_tracer.span_context,
                exporter=exporter,
                sampler=sampler,
            )
        return tracer

    def _get_app_insights_key(self):
        """Get Application Insights Key."""
        try:
            if self.app_insights_key is None:
                self.app_insights_key = getenv(
                    self.APPINSIGHTS_INSTRUMENTATION_KEY, None
                )
            if self.app_insights_key is not None:
                utils.validate_instrumentation_key(self.app_insights_key)
                return self.app_insights_key
            else:
                raise Exception("ApplicationInsights Key is not set")
        except Exception as exp:
            raise Exception(f"Exception is getting app insights key-> {exp}")

    def _get_callback(self, component_name):
        def _callback_add_role_name(envelope):
            """Add role name for logger."""
            envelope.tags["ai.cloud.role"] = component_name
            envelope.tags["ai.cloud.roleInstance"] = component_name

        return _callback_add_role_name

    def update_config(self, config=None):
        """Update logger configuration."""
        if config is not None:
            self.config.update(config)
        self.app_insights_key = self.config.get("app_insights_key")
        self.log_level = self.config.get("log_level")


def get_disabled_logger():
    """Get a disabled AppLogger.

    Returns:
        AppLogger: A disabled AppLogger
    """
    return AppLogger(
        config={"logging_enabled": "false", "app_insights_key": str(uuid.uuid1())}
    )
