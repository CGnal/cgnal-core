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


LazyIterableType = TypeVar("LazyIterableType", bound="LazyIterable")
CachedIterableType = TypeVar("CachedIterableType", bound="CachedIterable")
BaseIterableType = TypeVar("BaseIterableType", bound="BaseIterable")


class BaseIterable(Generic[T]):
    """Abstract class defining interface for iterables."""

    @property
    @abstractmethod
    def type(self) -> Type[T]:
        """Return the type of the objects in the Iterable."""
        raise NotImplementedError

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
    def empty(cls: Type[BaseIterableType]) -> BaseIterableType:
        """Return an empty iterable instance."""
        raise NotImplementedError


class LazyIterable(BaseIterable[T]):
    """Base class to be used for implementing lazy iterables."""

    @classmethod
    def from_iterable(
        cls: Type[LazyIterableType], iterable: BaseIterable[T]
    ) -> LazyIterableType:
        """
        Create a new instance of this class from a BaseIterable instance.

        :param iterable: iterable instance
        :return: lazy iterable
        """

        def generator():
            for item in iterable:
                yield item

        return cls(IterGenerator(generator))

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
    def empty(cls: Type[LazyIterableType]) -> LazyIterableType:
        """Return an empty lazy iterable.

        :return: Empty instance
        """

        def empty():
            return iter(())

        return cls(IterGenerator(empty))


class CachedIterable(BaseIterable[T]):
    """Base class to be used for implementing cached iterables."""

    @classmethod
    def from_iterable(
        cls: Type[CachedIterableType], iterable: BaseIterable[T]
    ) -> CachedIterableType:
        """
        Create a new instance of this class from a BaseIterable instance.

        :param iterable: iterable instance
        :return: cached iterable
        """
        return cls(list(iterable.items))

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
    def empty(cls: Type[CachedIterableType]) -> CachedIterableType:
        """Return an empty cached iterable.

        :return: Empty instance
        """
        return cls([])


class IterableUtilsMixin(
    Generic[T, LazyIterableType, CachedIterableType], BaseIterable[T], ABC
):
    """
    Class to provide base interfaces and methods for enhancing iterables classes and enable more functional approaches.

    In particular, the class provides among others implementation for map, filter and foreach methods.
    """

    lazy_type: Type[LazyIterableType]
    cached_type: Type[CachedIterableType]

    @staticmethod
    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of this class.

        :param cls: parent object class
        :param args: passed to the super class __new__ method
        :param kwargs: passed to the super class __new__ method
        :raises RuntimeError: if the cached and lazy versions were not defined before instantiating the class
        :return: an instance of this class
        """
        try:
            if not (
                issubclass(cls.lazy_type, BaseIterable)
                and not issubclass(cls.lazy_type, CachedIterable)
                and issubclass(cls.cached_type, BaseIterable)
                and not issubclass(cls.cached_type, LazyIterable)
            ):
                raise AttributeError
        except AttributeError:
            raise RuntimeError(
                "When extending IterableUtilsMixin, two classes must be defined: "
                "a lazy version extending LazyIterable and a cached version extending CachedIterable.\n"
                "The second class to be defined must register the two classes together "
                "using the RegisterLazyCachedIterables decorator."
            )

        try:
            return super().__new__(cls, *args, **kwargs)
        except TypeError:
            return super().__new__(cls)

    def to_cached(self) -> CachedIterableType:
        """
        Create a new cached instance of this instance.

        :return: cached iterable
        """
        return self.cached_type.from_iterable(self)

    def to_lazy(self) -> LazyIterableType:
        """
        Create a new lazy instance of this instance.

        :return: lazy iterable
        """
        return self.lazy_type.from_iterable(self)

    def take(self, size: int) -> CachedIterableType:
        """
        Take the first n elements of the iterables.

        :param size: number of elements to be taken
        :return: cached iterable with the first elements
        """
        return self.cached_type(list(islice(self, size)))

    def filter(self, f: Callable[[T], bool]) -> LazyIterableType:
        """
        Return an iterable where elements have been filtered based on a boolean function.

        :param f: boolean function that selects items
        :return: lazy iterable with elements filtered
        """

        def generator():
            for item in self:
                if f(item):
                    yield item

        return self.lazy_type(IterGenerator(generator))

    def __iter__(self) -> Iterator[T]:
        """
        Return an iterator over the items.

        :yield: items
        """
        for item in self.items:
            yield item

    def batch(self, size: int = 100) -> Iterator[CachedIterableType]:
        """
        Return an iterator of batches of size *size*.

        :param size: dimension of the batch
        :yield: iterator of batches
        """
        for batch in groupIterable(self.items, batch_size=size):
            yield self.cached_type(batch)

    def map(self, f: Callable[[T], T_co]) -> LazyIterableType:
        """
        Map all elements of an iterable with the provided function.

        :param f: function to be used to map the elements
        :return: mapped iterable
        """

        def generator():
            for item in self:
                yield f(item)

        return self.lazy_type(IterGenerator(generator))

    def foreach(self, f: Callable[[T], Any]):
        """
        Execute the provided function on each element of the iterable.

        :param f: function to be executed for each element
        """
        for doc in self.items:
            f(doc)

    def from_element(
        self, value: T, cached=True
    ) -> Union[LazyIterableType, CachedIterableType]:
        """
        Instantiate a new object of this class from a single element.

        :param value: element
        :param cached: whether a cached iterable should be returned, defaults to True
        :return: iterable object
        """
        if cached:
            return self.cached_type([value])
        else:

            def generator():
                yield value

            return self.lazy_type(IterGenerator(generator))


IterableUtilsMixinType = TypeVar(
    "IterableUtilsMixinType", bound=IterableUtilsMixin, covariant=True
)


class RegisterLazyCachedIterables:
    """Register the lazy and cached version of the iterables."""

    def __init__(
        self,
        class_object_first: Type[IterableUtilsMixin],
        unidirectional_link: bool = False,
    ):
        """
        Initialize an instance of this class.

        :param class_object_first: the first iterable class object (the lazy or chached version)
        :param unidirectional_link: if True, only set the link in the second class passed to the __call__ method
        """
        self.class_object_first = class_object_first
        self.unidirectional_link = unidirectional_link

    @staticmethod
    def register_lazy(
        class_object_lazy: Type[IterableUtilsMixin],
        class_object_cached: Type[IterableUtilsMixin],
    ):
        """
        Link the lazy and cached versions.

        :param class_object_lazy: the lazy iterable class object
        :param class_object_cached: the chached iterable class object
        """
        class_object_lazy.cached_type = class_object_cached
        class_object_lazy.lazy_type = class_object_lazy

    @staticmethod
    def register_cached(
        class_object_lazy: Type[IterableUtilsMixin],
        class_object_cached: Type[IterableUtilsMixin],
    ):
        """
        Link the lazy and cached versions.

        :param class_object_lazy: the lazy iterable class object
        :param class_object_cached: the chached iterable class object
        """
        class_object_cached.cached_type = class_object_cached
        class_object_cached.lazy_type = class_object_lazy

    def __call__(
        self, class_object_second: Type[IterableUtilsMixinType]
    ) -> Type[IterableUtilsMixinType]:
        """
        Link the lazy and cached versions.

        :param class_object_second: the second iterable class object (the lazy or chached version)
        :raises TypeError: if the types are not correct
        :return: the modified class
        """
        if issubclass(self.class_object_first, LazyIterable) and issubclass(
            class_object_second, BaseIterable
        ):
            self.register_cached(self.class_object_first, class_object_second)
            if not self.unidirectional_link:
                self.register_lazy(self.class_object_first, class_object_second)
        elif issubclass(self.class_object_first, CachedIterable) and issubclass(
            class_object_second, BaseIterable
        ):
            self.register_lazy(class_object_second, self.class_object_first)
            if not self.unidirectional_link:
                self.register_cached(class_object_second, self.class_object_first)
        else:
            raise TypeError(
                "The lazy-cached pairing must be done between two classes, one extending CachedIterable, "
                "the other extending LazyIterable and both extending IterableUtilsMixin."
            )
        return class_object_second


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
