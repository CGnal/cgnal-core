"""Data layer module."""
import pandas as pd
from abc import abstractmethod, ABC
from bson.objectid import ObjectId
from typing import (
    Any,
    Callable,
    Optional,
    Iterator,
    TypeVar,
    Union,
    Hashable,
    Dict,
    List,
    Generic,
)
from pymongo.collection import UpdateResult

from cgnal.core.typing import T
from cgnal.core.data.model.text import Document
from cgnal.core.data.model.core import IterGenerator
from cgnal.core.data.exceptions import NoTableException

DataVal = Union[Document, pd.DataFrame, pd.Series]

V = TypeVar("V")


class DAO(Generic[T, V], ABC):
    """Data Access Object."""

    @abstractmethod
    def computeKey(self, obj: T) -> Union[Hashable, Dict[str, ObjectId]]:
        """
        Compute the key of a given object.

        :param T obj: object
        :return Union[Hashable, Dict[str, ObjectId]]: hashable key
        """
        ...

    @abstractmethod
    def get(self, obj: T) -> V:
        """
        Return the record from the persistance layer corresponding to a object.

        :param T obj: object
        :return V: value
        """
        ...

    @abstractmethod
    def parse(self, row: V) -> T:
        """
        Parse the record of the persistance layer into a object.

        :param V row: record of the persistance layer
        :return T: object
        """
        ...


class Archiver(Generic[T], ABC):
    """Object that retrieve data from source and stores it in memory."""

    @abstractmethod
    def retrieve(self, condition: Any, sort_by: Any) -> Iterator[T]:
        """
        Retrieve objects under given filter conditions and sorting options.

        :param Any condition: filter to be applied in the persistance layer
        :param Any sort_by: sorting options
        :yield Iterator[T]: iterator over the objects retrieved from the persistance layer
        """
        ...

    @abstractmethod
    def archive(
        self, obj: T
    ) -> Union["Archiver", None, UpdateResult, List[UpdateResult]]:
        """
        Archive an object in the persistance layer.

        :param T obj: object to be archived
        :return: archiver or update result
        """
        ...

    def map(self, f: Callable[[T], V], *args: Any, **kwargs: Any) -> Iterator[V]:
        """
        Apply a tranformation to each object and return an iterator.

        :param Callable[[T], V] f: function
        :yield Iterator[V]: iterator over transformed objects
        """
        for obj in self.retrieve(*args, **kwargs):  # type: T
            yield f(obj)

    def foreach(self, f: Callable[[T], None], *args, **kwargs) -> None:
        """
        Apply a function to each object.

        :param Callable[[T], None] f: function
        """
        for obj in self.retrieve(*args, **kwargs):  # type: T
            f(obj)

    def retrieveGenerator(self, condition: Any, sort_by: Any) -> IterGenerator[T]:
        """
        Wrap the retrieve method to return an iterator generator.

        :param Any condition: filter condition
        :param Any sort_by: sorting options
        :return IterGenerator[T]: iterator generator
        """
        def __iterator__():
            return self.retrieve(condition=condition, sort_by=sort_by)

        return IterGenerator(__iterator__)


class TableABC(ABC):
    """Abstract class for tables."""

    @abstractmethod
    def to_df(self, query: str) -> pd.DataFrame:
        """
        Transform the underling data into a pandas dataframe.

        :param str query: query
        :return pd.DataFrame: pandas dataframe
        """
        ...

    @abstractmethod
    def write(self, df: pd.DataFrame) -> None:
        """
        Write the pandas dataframe in the persistance layer.

        :param pd.DataFrame df: pandas dataframe
        """
        ...


class DatabaseABC(ABC):
    """Abstract class for databases."""

    @abstractmethod
    def table(self, table_name: str) -> Optional[TableABC]:
        """
        Return the table with a given name.

        :param str table_name: name of the table
        :return Optional[TableABC]: table or None if table not found
        """
        ...


class Writer(ABC):
    """Abstract class to write Tables."""

    @property
    @abstractmethod
    def table(self) -> TableABC:
        """Return table object."""
        ...

    @abstractmethod
    def push(self, df: pd.DataFrame) -> None:
        """Push a pandas dataframe."""
        ...


class EmptyDatabase(DatabaseABC):
    """Class for empty Databases."""

    def table(self, table_name: str) -> None:
        """
        Raise an exception, since the database is empty.

        :param str table_name: name of the table
        :raises NoTableException: no table is present in an empty database
        """
        raise NoTableException(f"No table found with name {table_name}")
