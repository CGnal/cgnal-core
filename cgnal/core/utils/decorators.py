"""Module that gathers useful decorators."""

import inspect
import os
from functools import wraps, partial
from glob import glob
from typing import Callable, Any, Iterable, Tuple

import pandas as pd
from deprecated import deprecated

from cgnal.core.typing import PathLike, T
from cgnal.core.utils.dict import union
from cgnal.core.utils.fs import create_dir_if_not_exists


def cache(func: Callable[..., T]) -> Callable[..., T]:
    """
    Return a decorator to cache function return values.

    :param func: input function
    :return: function wrapper
    """

    @wraps(func)
    def _wrap(obj):
        try:
            return obj.__dict__[func.__name__]
        except KeyError:
            score = func(obj)
            obj.__dict__[func.__name__] = score
            return score

    _wrap.__name__ = func.__name__
    return _wrap


def lazyproperty(obj: Any) -> property:
    """
    Return a lazy property, i.e. a property that is computed if needed and then cached.

    :param obj: method do decorate
    :return: wrapped method
    """
    return property(cache(obj))


class Cached(object):
    """Class to cache results and export them as pickle to be later reloaded."""

    @property
    def __cache__(self):
        """Hidden property that stores the custom cache that can be exported and imported."""
        try:
            return self.__cache
        except AttributeError:
            self.__cache = {}
            return self.__cache__

    @staticmethod
    def cache(func: Callable[["Cached"], T]) -> property:
        """
        Return a decorator to cache function return values.

        :param func: input function
        :return: function wrapper
        """

        @wraps(func)
        def _wrap(obj: "Cached"):
            try:
                return obj.__cache__[func.__name__]
            except KeyError:
                score = func(obj)
                obj.__cache__[func.__name__] = score
                return score

        return property(_wrap)

    def clear_cache(self) -> None:
        """
        Clear cache of the object.

        :return: None
        """
        self.__cache__.clear()

    def save_pickles(self, path: PathLike) -> None:
        """
        Save pickle in given path.

        :param path: saving path
        :return: None
        """
        path = create_dir_if_not_exists(path)

        for k, data in self.__cache__.items():
            Cached.save_element(os.path.join(path, k), data)

    @staticmethod
    def save_element(filename: PathLike, obj: Any) -> None:
        """
        Save given object in given file.

        :param filename: saving path
        :param obj: object to be saved
        :return: None
        """
        if isinstance(obj, dict):
            create_dir_if_not_exists(filename)
            [Cached.save_element(os.path.join(filename, k), v) for k, v in obj.items()]
        elif isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            pd.to_pickle(obj, "%s.p" % filename)
        else:
            try:
                pd.to_pickle(obj, "%s.p" % filename)
            except Exception:
                raise ValueError("Cannot save input of type %s" % str(obj.__class__))

    def load(self, filename: PathLike) -> None:
        """
        Load pickle at given path (or all pickles in given folder).

        :param filename: path to pickles
        :return: None
        """
        self.__cache__.update(dict(self.load_element(filename, "")))
        return None

    @classmethod
    def load_element(
        cls, filename: PathLike, prefix: str = ""
    ) -> Iterable[Tuple[PathLike, Any]]:
        """
        Load given element/property of the object.

        :param filename: name of the file/directory where the data should be taken from
        :param prefix: prefix to add to elements in filename in case it is a directory
        :return: iterable of couples (path, loaded object)
        """
        if os.path.isdir(filename):
            for path in glob(os.path.join(filename, "*")):
                for name, obj in cls.load_element(
                    path, f"{cls.__reformat_name(filename)}_"
                ):
                    yield f"{prefix}{name}", obj
        else:
            yield str(cls.__reformat_name(filename)), pd.read_pickle(filename)

    @staticmethod
    def __reformat_name(filename: PathLike) -> PathLike:
        """
        Reformat file name to make it readable with load_element.

        :param filename: file name to reformat
        :return: reformatted filename
        """
        if os.path.isfile(filename):
            return os.path.basename(filename).replace(".p", "")
        else:
            return os.path.basename(filename)


def paramCheck(function: Callable[..., T], allow_none: bool = True) -> Callable[..., T]:
    """
    Return a decorator that performs runtime checks on the input types.

    :param function: function to be checked against its typing annotations
    :param allow_none: whether None values are allowed
    :return: wrapped function
    """

    @wraps(function)
    def check(*arguments, **kwargs):

        default_values = inspect.getfullargspec(function).defaults
        annotations = inspect.getfullargspec(function).annotations
        _args = (
            inspect.getfullargspec(function).args[: -len(default_values)]
            if default_values
            else inspect.getfullargspec(function).args
        )

        non_default_args = [
            arg for arg in _args if (arg in annotations.keys() and arg != "self")
        ]
        default_args = (
            inspect.getfullargspec(function).args[-len(default_values) :]
            if default_values
            else []
        )

        arg_dict = union(
            {
                argument: {"type": annotations[argument], "value": None}
                for argument in non_default_args
            },
            {
                argument: {
                    "type": annotations.get(argument)
                    if annotations.get(argument)
                    else type(default_values[index]),
                    "value": default_values[index],
                }
                for index, argument in enumerate(default_args)
            },
        )

        NoneType = type(None)

        for index, arg in enumerate(inspect.getfullargspec(function).args):

            if arg != "self":
                argIn = (
                    arguments[index]
                    if index < len(arguments)
                    else kwargs.get(arg, arg_dict[arg]["value"])
                )
                if argIn is None:
                    if allow_none is False:
                        raise ValueError(f"{arg} cannot be None")
                else:
                    if (not isinstance(argIn, arg_dict[arg]["type"])) and (
                        arg_dict[arg]["type"] != NoneType
                    ):
                        raise TypeError(
                            f"{arg} parameter must be of type {str(arg_dict[arg]['type'])}"
                        )

        return function(*arguments, **kwargs)

    return check


@deprecated(
    "This decorator is deprecated and will be removed in future versions. Use typeguard library instead"
)
def param_check(with_none: bool) -> Callable[..., Any]:
    """
    Return a decorator that performs runtime checks on the typing of the inputs.

    :param with_none: whether None values are allowed
    :return: decorator
    """
    return partial(paramCheck, allow_none=with_none)
