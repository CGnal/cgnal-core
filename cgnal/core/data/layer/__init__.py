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
    """Data Access Object"""

    @abstractmethod
    def computeKey(self, obj: T) -> Union[Hashable, Dict[str, ObjectId]]:
        ...

    @abstractmethod
    def get(self, obj: T) -> V:
        ...

    @abstractmethod
    def parse(self, row: V) -> T:
        ...


class Archiver(Generic[T], ABC):
    """Object that retrieve data from source and stores it in memory"""

    @abstractmethod
    def retrieve(self, condition: Any, sort_by: Any) -> Iterator[T]:
        ...

    @abstractmethod
    def archive(
        self, obj: T
    ) -> Union["Archiver", None, UpdateResult, List[UpdateResult]]:
        ...

    def map(self, f: Callable[[T], V], *args: Any, **kwargs: Any) -> Iterator[V]:
        for obj in self.retrieve(*args, **kwargs):  # type: T
            yield f(obj)

    def foreach(self, f: Callable[[T], None], *args, **kwargs) -> None:
        for obj in self.retrieve(*args, **kwargs):  # type: T
            f(obj)

    def retrieveGenerator(self, condition: Any, sort_by: Any) -> IterGenerator[T]:
        def __iterator__():
            return self.retrieve(condition=condition, sort_by=sort_by)

        return IterGenerator(__iterator__)


class TableABC(ABC):
    """
    Abstract class for tables
    """

    @abstractmethod
    def to_df(self, query: str) -> pd.DataFrame:
        ...

    @abstractmethod
    def write(self, df: pd.DataFrame) -> None:
        ...


class DatabaseABC(ABC):
    """
    Abstract class for databases
    """

    @abstractmethod
    def table(self, table_name: str) -> Optional[TableABC]:
        ...


class Writer(ABC):
    """
    Abstract class to write Tables
    """

    @property
    @abstractmethod
    def table(self) -> TableABC:
        ...

    @abstractmethod
    def push(self, df: pd.DataFrame) -> None:
        ...


class EmptyDatabase(DatabaseABC):
    """
    Class for empty Databases
    """

    def table(self, table_name: str) -> None:
        raise NoTableException(f"No table found with name {table_name}")
