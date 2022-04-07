"""Typing utilities."""
import os
from typing import Union, Any, TypeVar, Hashable
from typing_extensions import Protocol

PathLike = Union[str, "os.PathLike[str]"]
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

# To be used as hashable key
K = TypeVar("K", bound=Hashable)


class SupportsLessThan(Protocol):
    """Protocol for a class that must support the less than operator."""

    def __lt__(self, __other: Any) -> bool:
        """
        Less than operator.

        :param Any __other: other operand
        :return bool: whether self is less than __other
        """
        ...
