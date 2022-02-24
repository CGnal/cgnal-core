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
    CRITICAL: Literal[50]
    ERROR: Literal[40]
    WARNING: Literal[30]
    INFO: Literal[20]
    DEBUG: Literal[10]
    NOTSET: Literal[0]


DEFAULT_LOG_LEVEL: StrLevelTypes = "INFO"


class WithLoggingABC(ABC):
    @property
    @abstractmethod
    def logger(self) -> Logger:
        """
        Logger instance to be used to output logs within a class
        :return: None, outputs logs
        """
        pass


class LoggingConfig(BaseConfig):
    @property
    def level(self) -> str:
        return self.getValue("level")

    @property
    def filename(self) -> PathLike:
        return self.getValue("filename")

    @property
    def default_config_file(self) -> PathLike:
        return self.getValue("default_config_file")

    @property
    def capture_warnings(self) -> bool:
        return self.getValue("capture_warnings")
