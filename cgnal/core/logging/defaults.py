"""Module for general logging functionalities and abstractions."""

import os
import sys
from types import TracebackType
from importlib import import_module
from typing import Optional, List, Callable, Union, Any, Type
from logging import getLogger, basicConfig, config, captureWarnings, Logger, FileHandler

from cgnal.core.typing import PathLike
from cgnal.core.config import merge_confs
from cgnal.core.logging import (
    WithLoggingABC,
    DEFAULT_LOG_LEVEL,
    LevelsDict,
    LevelTypes,
    StrLevelTypes,
)
from cgnal.core.utils.fs import create_dir_if_not_exists


levels: LevelsDict = {
    "CRITICAL": 50,
    "ERROR": 40,
    "WARNING": 30,
    "INFO": 20,
    "DEBUG": 10,
    "NOTSET": 0,
}


class WithLogging(WithLoggingABC):
    """Base class to be used for providing a logger embedded in the class."""

    @property
    def logger(self) -> Logger:
        """
        Create logger.

        :return: default logger
        """
        nameLogger = str(self.__class__).replace("<class '", "").replace("'>", "")
        return getLogger(nameLogger)

    def logResult(
        self, msg: Union[Callable[..., str], str], level: StrLevelTypes = "INFO"
    ) -> Callable[..., Any]:
        """
        Return a decorator to allow logging of inputs/outputs.

        :param msg: message to log
        :param level: logging level
        :return: wrapped method
        """

        def wrap(x: Any) -> Any:
            if isinstance(msg, str):
                self.logger.log(levels[level], msg)
            else:
                self.logger.log(levels[level], msg(x))
            return x

        return wrap


def getDefaultLogger(level: LevelTypes = levels[DEFAULT_LOG_LEVEL]) -> Logger:
    """
    Create default logger.

    :param level: logging level
    :return: logger
    """
    basicConfig(level=level)
    return getLogger()


def configFromFiles(
    config_files: List[PathLike],
    capture_warnings: bool = True,
    catch_exceptions: Optional[str] = None,
) -> None:
    """
    Configure loggers from configuration obtained merging configuration files.

    If any handler inherits from FileHandler create the directory for its output files if it does not exist yet.

    :param config_files: list of configuration files
    :param capture_warnings: whether to capture warnings with logger
    :param catch_exceptions: name of the logger used to catch exceptions. If None do not catch exception with loggers.
    :return: None
    """
    captureWarnings(capture_warnings)

    configuration = merge_confs(filenames=config_files, default=None)
    for v in configuration.to_dict()["handlers"].values():
        splitted = v["class"].split(".")
        if issubclass(
            getattr(import_module(".".join(splitted[:-1])), splitted[-1]), FileHandler
        ):
            create_dir_if_not_exists(os.path.dirname(v["filename"]))
    config.dictConfig(configuration)

    if catch_exceptions is not None:
        except_logger = getLogger(catch_exceptions)
        print(
            f"Catching excetptions with {except_logger.name} logger using handlers "
            f'{", ".join([x.name for x in except_logger.handlers if x.name is not None])}'
        )

        def handle_exception(
            exc_type: Type[BaseException],
            exc_value: BaseException,
            exc_traceback: Optional[TracebackType],
        ) -> Any:
            if issubclass(exc_type, KeyboardInterrupt) and exc_traceback is not None:
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
            else:
                except_logger.error(
                    f"{exc_type.__name__}: {exc_value}",
                    exc_info=(exc_type, exc_value, exc_traceback),
                )

        sys.excepthook = handle_exception


def logger(name: Optional[str] = None) -> Logger:
    """
    Return a logger with the specified name, creating it if necessary.

    :param name: name to be used for the logger. If None return root logger

    :return: named logger
    """
    return getLogger(name)
