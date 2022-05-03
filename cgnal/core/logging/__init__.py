"""Logging module."""
from abc import ABC, abstractmethod
from logging import Logger
from typing_extensions import Literal, TypedDict
from cgnal.core.typing import PathLike
from cgnal.core.config import BaseConfig

LevelTypes = Literal[
    "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET", 50, 40, 30, 20, 10, 0
]
StrLevelTypes = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]


class LevelsDict(TypedDict):
    """Dictionary of logging levels."""

    CRITICAL: Literal[50]
    ERROR: Literal[40]
    WARNING: Literal[30]
    INFO: Literal[20]
    DEBUG: Literal[10]
    NOTSET: Literal[0]


DEFAULT_LOG_LEVEL: StrLevelTypes = "INFO"


class WithLoggingABC(ABC):
    """Abstract class providing logging capabilities."""

    @property
    @abstractmethod
    def logger(self) -> Logger:
        """Logger instance to be used to output logs within a class."""
        raise NotImplementedError


class LoggingConfig(BaseConfig):
    """Logging configuration."""

    @property
    def level(self) -> str:
        """
        Returnn logging level.

        :return: level
        """
        return self.getValue("level")

    @property
    def filename(self) -> PathLike:
        """
        Name of the file where logs are stored.

        :return: filename
        """
        return self.getValue("filename")

    @property
    def default_config_file(self) -> PathLike:
        """
        Return default logging configuration file.

        :return: default config file
        """
        return self.getValue("default_config_file")

    @property
    def capture_warnings(self) -> bool:
        """
        Flag that determines whether waring are captured.

        :return: capture warnings
        """
        return self.getValue("capture_warnings")
