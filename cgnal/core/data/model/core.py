"""Module with base abstraction of common objects."""

import pickle
import sys
from abc import ABC, abstractmethod
from functools import reduce
from itertools import islice
from typing import List, Iterable, Iterator, Tuple, Union, Type, Any, TypeVar, Optional

import dill
import numpy as np
import pandas as pd
from pandas.core.tools.datetimes import DatetimeScalar, Timestamp

from cgnal.core.typing import PathLike, T_co  # , T, T_co
from cgnal.core.utils.dict import groupIterable

if sys.version_info[0] < 3:
    pass

from typing import Generic, Callable, Sequence

T = TypeVar("T")


class Serializable(ABC):
    """Abstract Class to be used to extend objects that can be serialised."""

    @abstractmethod
    def write(self, filename: PathLike) -> None:
        """
        Write class to a file.

        :param filename: filename
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, filename: PathLike) -> "Serializable":
        """
        Load class from a file.

        :param filename: filename
        """
        ...


class PickleSerialization(Serializable):
    """Serialization based on pickle package."""

    def write(self, filename: PathLike) -> None:
        """
        Write instance as pickle.

        :param filename: Name of the file where to save the instance
        """
        with open(filename, "wb") as fid:
            pickle.dump(self, fid)

    @classmethod
    def load(cls, filename: PathLike) -> "PickleSerialization":
        """
        Load instance from pickle.

        :param filename: Name of the file to be read
        :return: Instance of the read Model
        """
        with open(filename, "rb") as fid:
            return pickle.load(fid)


class DillSerialization(Serializable):
    """Serialization based on dill package."""

    def write(self, filename: PathLike) -> None:
        """
        Write instance as pickle.

        :param filename: Name of the file where to save the instance
        """
        with open(filename, "wb") as fid:
            dill.dump(self, fid)

    @classmethod
    def load(cls, filename: PathLike) -> "DillSerialization":
        """
        Load instance from file.

        :param filename: Name of the file to be read
        :return: Instance of the read Model
        """
        with open(filename, "rb") as fid:
            return dill.load(fid)


class IterGenerator(Generic[T]):
    """Base class representing any generator."""

    def __init__(
        self,
        generator_function: Callable[[], Iterator[T]],
        _type: Optional[Type[T]] = None,
    ):
        """
        Class that allows a given generator to be accessed as an Iterator via .iterator property.

        :param generator_function: function that outputs a generator
        :param _type: type returned by the generartor, required when the generator is empty
        :raises TypeError: when type mismatch happens between generator and provided type
        :raises ValueError: when an empty generator is provided without _type specification
        """
        self.generator_function = generator_function

        try:
            inferred_type = type(next(self.iterator))
            if _type is not None:
                if not issubclass(inferred_type, _type):
                    raise TypeError(
                        f"Provided type {_type} not compliant with type provided by the generator function"
                    )
                self.type = _type
            else:
                self.type = inferred_type
        except StopIteration:
            if _type is None:
                raise ValueError(
                    "_type argument must be provided when generator is empty."
                )
            self.type = _type

    @property
    def iterator(self) -> Iterator[T]:
        """
        Return an iterator over the given generator function.

        :return: an iterator
        """
        return self.generator_function()


VarIterable = TypeVar("VarIterable", bound="_BaseIterable")
LazyVarIterable = TypeVar("LazyVarIterable", bound="_LazyIterable")
CachedVarIterable = TypeVar("CachedVarIterable", bound="_CachedIterable")


class _BaseIterable(Generic[T], ABC):
    """
    Class to provide base interfaces and methods for enhancing iterables classes and enable more functional approaches.

    In particular, the class provides among others implementation for map, filter and foreach methods.
    """

    @property
    @abstractmethod
    def type(self) -> Type[T]:
        """Return the type of the objects in the Iterable."""

    @property
    @abstractmethod
    def items(self) -> Iterable[T]:
        """
        Return an iterator over the items.

        :return: Iterable[T]
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def cached(self) -> bool:
        """
        Whether the iterable is cached in memory or lazy.

        :return: boolean indicating whether iterable is fully-stored in memory
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def empty(cls: Type[VarIterable]) -> VarIterable:
        """Return an empty iterable instance."""


class _LazyIterable(_BaseIterable[T], Generic[T]):
    """Base class to be used for implementing lazy iterables."""

    def __init__(self, items: IterGenerator):
        """
        Return an instance of the class to be used for implementing lazy iterables.

        :param items: IterGenerator containing the generator of items
        """
        self._items = items

    @property
    def items(self) -> Iterator[T]:
        """Return an iterator over the items.

        :return: Iterable[T]
        """
        return self._items.iterator

    @property
    def cached(self) -> bool:
        """
        Whether the iterable is cached in memory or lazy.

        :return: boolean indicating whether iterable is fully-stored in memory
        """
        return False

    @classmethod
    def empty(cls: Type[LazyVarIterable]) -> LazyVarIterable:
        """Return an empty lazy iterable.

        :return: Empty instance
        """

        def empty():
            return iter(())

        return cls(IterGenerator(empty))


class _CachedIterable(_BaseIterable[T], Generic[T]):
    """Base class to be used for implementing cached iterables."""

    def __init__(self, items: Sequence[T]):
        """
        Return instance of a class to be used for implementing cached iterables.

        :param items: sequence or iterable of elements
        """
        self._items = list(items)

    def __len__(self) -> int:
        """
        Return the size of the list of elements.

        :return: size of the list of elements
        """
        return len(self.items)

    @property
    def items(self) -> Sequence[T]:
        """
        Return an iterator over the items.

        :return: Iterable[T]
        """
        return self._items

    def __getitem__(self, item: int) -> T:
        """
        Get the item by position index.

        :param item: integer representing the position.
        :return: item
        """
        return self.items[item]

    @property
    def cached(self) -> bool:
        """
        Whether the iterable is cached in memory or lazy.

        :return: boolean indicating whether iterable is fully-stored in memory
        """
        return True

    @classmethod
    def empty(cls: Type[CachedVarIterable]) -> CachedVarIterable:
        """Return an empty cached iterable.

        :return: Empty instance
        """
        return cls([])


L = TypeVar("L", bound=_LazyIterable)
C = TypeVar("C", bound=_CachedIterable)


class _IterableUtils(_BaseIterable[T], Generic[T, C, L], ABC):
    @property
    @abstractmethod
    def _lazyType(self) -> Type[L]:
        ...

    @property
    @abstractmethod
    def _cachedType(self) -> Type[C]:
        ...

    @property
    def asLazy(self) -> L:
        """
        Provide a lazy representation of the iterable.

        :return: lazy iterable
        """

        def generator():
            for item in self:
                yield item

        return self._lazyType(IterGenerator(generator))

    @property
    def asCached(self) -> C:
        """
        Provide an in-memory cached representation of the iterable.

        :return: cached iterable
        """
        return self._cachedType(list(self.items))

    def take(self, size: int) -> C:
        """
        Take the first n elements of the iterables.

        :param size: number of elements to be taken
        :return: cached iterable with the first elements
        """
        return self._cachedType(list(islice(self, size)))

    def filter(self, f: Callable[[T], bool]) -> L:
        """
        Return an iterable where elements have been filtered based on a boolean function.

        :param f: boolean function that selects items
        :return: lazy iterable with elements filtered
        """

        def generator():
            for item in self:
                if f(item):
                    yield item

        return self._lazyType(IterGenerator(generator))

    def __iter__(self) -> Iterator[T]:
        """
        Return an iterator over the items.

        :yield: items
        """
        for item in self.items:
            yield item

    def batch(self, size: int = 100) -> Iterator[C]:
        """
        Return an iterator of batches of size *size*.

        :param size: dimension of the batch
        :yield: iterator of batches
        """
        for batch in groupIterable(self.items, batch_size=size):
            yield self._cachedType(batch)

    def map(self, f: Callable[[T], T_co]) -> L:
        """
        Map all elements of an iterable with the provided function.

        :param f: function to be used to map the elements
        :return: mapped iterable
        """

        def generator():
            for item in self:
                yield f(item)

        return self._lazyType(IterGenerator(generator))

    def foreach(self, f: Callable[[T], Any]):
        """
        Execute the provided function on each element of the iterable.

        :param f: function to be executed for each element
        """
        for doc in self.items:
            f(doc)

    def fromElement(self, value: T, cached=True):
        if cached:
            return self._cachedType([value])
        else:

            def generator():
                yield value

            return self._lazyType(IterGenerator(generator))


class BaseIterable(
    _IterableUtils[T, "CachedIterable[T]", "LazyIterable[T]"], Generic[T]
):
    """Basic class for extending iterable classes with boosted functionalities."""

    @property
    def _lazyType(self) -> "Type[LazyIterable[T]]":
        """Pre-defined lazy type to cast lazy outputs.

        :return: Lazy Iterable Class
        """
        return LazyIterable[T]

    @property
    def _cachedType(self) -> "Type[CachedIterable[T]]":
        """Pre-defined cached type to cast lazy outputs.

        :return: Cached Iterable Class
        """
        return CachedIterable[T]


class CachedIterable(
    _CachedIterable[T], BaseIterable[T], Generic[T], DillSerialization
):
    """Basic class for extending cached iterable class with boosted functionalities."""

    def type(self) -> Type[T]:
        """
        Return the type of the objects in the Iterable.

        :return: type of the object of the iterable
        """
        return self.__type

    def __init__(self, items: Sequence[T], _type: Optional[Type[T]] = None):
        """
        Return instance of a class to be used for implementing cached iterables.

        :param items: sequence or iterable of elements
        :param _type: type returned by the generartor, required when the generator is empty
        :raises TypeError: when type mismatch happens between sequence elements and provided type
        :raises ValueError: when an empty sequence is provided without _type specification
        """
        try:
            inferred_type = type(items[0])
            if _type is not None:
                if not issubclass(inferred_type, _type):
                    raise TypeError(
                        f"Provided type {_type} not compliant with type provided by the generator function"
                    )
                self.__type = _type
            else:
                self.__type = inferred_type
        except StopIteration:
            if _type is None:
                raise ValueError(
                    "_type argument must be provided when generator is empty."
                )
            self.__type = _type

        super(CachedIterable, self).__init__(items)


class LazyIterable(_LazyIterable[T], BaseIterable[T], Generic[T]):
    """Basic class for extending lazy iterable class with boosted functionalities."""

    def type(self) -> Type[T]:
        """
        Return the type of the objects in the Iterable.

        :return: type of the object of the iterable
        """
        return self._items.type


class BaseRange(ABC):
    """Abstract Range Class."""

    @property
    @abstractmethod
    def start(self) -> Timestamp:
        """
        Return the first timestamp.

        :return: Timestamp
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def end(self) -> Timestamp:
        """
        Return the last timestamp.

        :return: Timestamp
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator["Range"]:
        """
        Return an iterator over continuous ranges.

        :return: Iterator[Range]
        """
        ...

    @abstractmethod
    def __add__(self, other: "BaseRange") -> "BaseRange":
        """
        Return a range composed by two ranges.

        :param other: other range to be merged
        :return: merged range
        """
        ...

    @abstractmethod
    def overlaps(self, other: "BaseRange") -> bool:
        """
        Return whether two ranges overlaps.

        :param other: other range to be compared with
        :return: True if the two ranges intersect, False otherwise
        """
        ...

    @abstractmethod
    def range(self, freq="H") -> List[Timestamp]:
        """
        Return list of timestamps, spaced by given frequency.

        :param freq: frequency of timestamps, valid values are "D" (day), "H" (hours), "M"(minute), "S" (seconds).
        :return: list of timestamps
        """
        ...

    def __str__(self) -> str:
        """
        Return string representation.

        :return: string representation
        """
        return " // ".join([f"{r.start}-{r.end}" for r in self])

    @property
    def days(self) -> List[Timestamp]:
        """
        Create date range with daily frequency.

        :return: list of pd.Timestamp from start to end with daily frequency
        """
        return self.range(freq="1D")

    @property
    def business_days(self) -> List[Timestamp]:
        """
        Create date range with daily frequency.

        :return: list of pd.Timestamp from start to end with daily frequency including only days from Mon to Fri
        """
        return self.range(freq="1B")

    @property
    def minutes_15(self) -> List[Timestamp]:
        """
        Create date range with daily frequency.

        :return: list of pd.Timestamp from start to end with 15 minutes frequency
        """
        return self.range(freq="15T")


class Range(BaseRange):
    """Base class for a continuous range."""

    def __init__(self, start: DatetimeScalar, end: DatetimeScalar) -> None:
        """
        Return a simple Range Class.

        :param start: starting datetime for the range
        :param end: ending datetime for the range
        :raises ValueError: if start > end
        """
        self._start = pd.to_datetime(start)
        self._end = pd.to_datetime(end)

        if self.start > self.end:
            raise ValueError(
                "Start and End values should be consequential: start < end"
            )

    @property
    def start(self) -> Timestamp:
        """
        Return the first timestamp.

        :return: Timestamp
        """
        return self._start

    @property
    def end(self) -> Timestamp:
        """
        Return the last timestamp.

        :return: Timestamp
        """
        return self._end

    def __iter__(self) -> Iterator["Range"]:
        """
        Return an iterator over continuous ranges.

        :yield: Iterator[Range]
        """
        yield Range(self.start, self.end)

    def range(self, freq="H") -> List[Timestamp]:
        """
        Return list of timestamps, spaced by given frequency.

        :param freq: given frequency
        :return: list of timestamps
        """
        return pd.date_range(self.start, self.end, freq=freq).tolist()

    def _overlaps_range(self, other: "Range") -> bool:
        """
        Check whether there is any overlap between current and other range.

        :param other: other range
        :return: bool
        """
        return ((self.start < other.start) and (self.end > other.start)) or (
            (other.start < self.start) and (other.end > self.start)
        )

    def overlaps(self, other: "BaseRange") -> bool:
        """
        Return whether two ranges overlaps.

        :param other: other range to be compared with
        :return: True or False whether the two overlaps
        """
        return any([self._overlaps_range(r) for r in other])

    def __add__(self, other: BaseRange) -> Union["CompositeRange", "Range"]:
        """
        Return a range composed by two ranges.

        :param other: other range to be merged
        :return: merged range
        :raises TypeError: other is not of type Range
        """
        if not isinstance(other, Range):
            raise TypeError(
                f"add operator not defined for argument of type {type(other)}. Argument should be of "
                f"type Range"
            )
        if isinstance(other, Range) and self.overlaps(other):
            return Range(min(self.start, other.start), max(self.end, other.end))
        else:
            return CompositeRange([self] + [span for span in other])


class CompositeRange(BaseRange):
    """Class representing a composition of ranges."""

    def __init__(self, ranges: List[Range]) -> None:
        """
        Return a range made up of multiple ranges.

        :param ranges: List of Ranges
        """
        self.ranges = ranges

    def simplify(self) -> Union["CompositeRange", Range]:
        """
        Simplify the list into disjoint Range objects, aggregating non-disjoint ranges.

        If only one range would be present, a simple Range object is returned.

        :return: BaseRange
        """
        ranges = sorted(self.ranges, key=lambda r: r.start)

        # check overlapping ranges
        overlaps = [
            first.overlaps(second) for first, second in zip(ranges[:-1], ranges[1:])
        ]

        def merge(agg: List[Range], item: Tuple[int, bool]) -> List[Range]:
            ith, overlap = item
            return (
                agg + [ranges[ith + 1]]
                if not overlap
                else agg[:-1] + [agg[-1] + ranges[ith + 1]]  # type: ignore
            )

        # merge ranges
        rangeList = reduce(merge, enumerate(overlaps), [ranges[0]])

        if len(rangeList) == 1:
            return rangeList[0]
        else:
            return CompositeRange(rangeList)

    @property
    def start(self) -> Timestamp:
        """
        Return the first timestamp.

        :return: Timestamp
        """
        return min([range.start for range in self.ranges])

    @property
    def end(self) -> Timestamp:
        """
        Return the last timestamp.

        :return: Timestamp
        """
        return max([range.end for range in self.ranges])

    def __iter__(self) -> Iterator["Range"]:
        """
        Return an iterator over continuous ranges.

        :yield: Iterator[Range]
        """
        for range in self.ranges:
            yield range

    def range(self, freq="H") -> List[Timestamp]:
        """
        Return list of timestamps, spaced by given frequency.

        :param freq: given frequency
        :return: list of timestamps
        """
        items = np.unique(
            [
                item
                for range in self.ranges
                for item in pd.date_range(range.start, range.end, freq=freq)
            ]
        )
        return sorted(items)

    def __add__(self, other: BaseRange) -> BaseRange:
        """
        Return a range composed by two ranges.

        :param other: BaseRange, other range to be merged
        :return: BaseRange, merged range
        :raises TypeError: other is not of type BaseRange
        """
        if not isinstance(other, BaseRange):
            raise TypeError(
                f"add operator not defined for argument of type {type(other)}. Argument should be of "
                f"type BaseRange"
            )
        return CompositeRange(self.ranges + list(other)).simplify()

    def overlaps(self, other: "BaseRange") -> bool:
        """
        Return whether two ranges overlaps.

        :param other: BaseRange, other range to be compared with
        :return: bool, True if the two ranges intersect, False otherwise
        """
        return any([r.overlaps(other) for r in self])
