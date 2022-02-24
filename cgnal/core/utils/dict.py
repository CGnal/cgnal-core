"""Module for basic dictionary functionalities."""

import sys

if sys.version_info[0] < 3:
    from itertools import izip as zip
else:
    from functools import reduce

from itertools import islice, tee, groupby
from typing import Iterator, Iterable, List, Tuple, Dict, Any, Callable, Optional
from copy import deepcopy as copy
from collections.abc import Mapping
from operator import add
from cgnal.core.typing import SupportsLessThan, T


def groupIterable(iterable: Iterable[T], batch_size: int = 10000) -> Iterator[List[T]]:
    """
    Split a given iterable into batches of given `batch_size`.

    :param iterable: iterable
    :param batch_size: int
    :return: Iterator
    """
    iterable = iter(iterable)
    return iter(lambda: list(islice(iterable, batch_size)), [])


def pairwise(iterable: Iterable[T]) -> zip:
    """
    Return a pairing of elements of an iterable.

    Example: s -> (s0,s1), (s1,s2), (s2, s3), ...

    :param iterable: iterable whose elements ought to be paired
    :return: zipped iterable
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def union(*dicts: dict) -> dict:
    """
    Return a dictionary that results from the recursive merge of the input dictionaries.

    :param dicts: list of dicts
    :return: merged dict
    """

    def __dict_merge(dct: dict, merge_dct: dict):
        """
        Recursive dict merge.

        Inspired by :meth:``dict.update()``, instead of updating only top-level keys, dict_merge recurses down into
        dicts nested to an arbitrary depth, updating keys. The ``merge_dct`` is merged into ``dct``.

        :param dct: dict onto which the merge is executed
        :param merge_dct: dct merged into dct
        :return: None
        """
        merged = copy(dct)
        for k, v in merge_dct.items():
            if (
                k in dct
                and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], Mapping)
            ):
                merged[k] = __dict_merge(dct[k], merge_dct[k])
            else:
                merged[k] = merge_dct[k]
        return merged

    return reduce(__dict_merge, dicts)


def flattenKeys(input_dict: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """
    Flatten dictionary keys.

    Return a dictionary whose nested keys have been flattened into the root level recursively flattening a multilevel
    nested dict into a single level dict with flattened keys and corresponding values.
    The keys are joined by the given separator.

    :param input_dict: a multilevel dict ex `{"a": {"b": {"c": 2}}, "d": 2, "e": 3}`
    :param sep: str delimiter to join keys
    :return: dict with flattened keys
    """

    def _flatten_(key: str, value: Any) -> List[Tuple[str, Any]]:
        if isinstance(value, dict) and (len(value) > 0):
            return reduce(
                add,
                [
                    _flatten_(sep.join([key, name]), item)
                    for name, item in value.items()
                ],
            )
        else:
            return [(key, value)]

    return union(*[dict(_flatten_(key, value)) for key, value in input_dict.items()])


def unflattenKeys(input_dict: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """
    Transform a dict into a nested dict by splitting keys on the given separator.

    A dict with key as `{'a.b':2}` will be transformed into a dict of dicts as `{'a':{'b':2}}`

    :param input_dict: dict
    :param sep: str delimiter to split keys
    :return: dict
    """

    def __translate(key: str, value: Any) -> Dict[str, Any]:
        levels = list(reversed(key.split(sep)))
        return reduce(lambda agg, level: {level: agg}, levels[1:], {levels[0]: value})

    return union(*[__translate(key, value) for key, value in input_dict.items()])


def __check(value: Optional[T]) -> bool:
    return False if value is None else True


def filterNones(_dict: Dict[T, Any]) -> Dict[T, Any]:
    """
    Return a dictionary where the key,value pairs are filtered where the value is None.

    :param _dict: dict with Nones
    :return: dict without Nones
    """
    agg = {}
    for k, v in _dict.items():
        if isinstance(v, dict):
            agg[k] = filterNones(v)
        elif __check(v):
            agg[k] = v
    return agg


def groupBy(
    lst: Iterable[T], key: Callable[[T], SupportsLessThan]
) -> Iterator[Tuple[SupportsLessThan, List[T]]]:
    """
    Perform groupBy operation on a list according to the given key.

    The function uses itertools groupby but on a sorted list.

    :param lst: List
    :param key: function to be used as a key to perform groupby operation
    :return: Iterator
    """
    for k, it in groupby(sorted(lst, key=key), key=key):
        yield k, list(it)
