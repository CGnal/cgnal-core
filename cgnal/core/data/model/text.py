"""Module for providing abstraction and classes for handling NLP data."""

import uuid
from abc import ABC
from typing import Dict, Any, Optional, Iterator, Tuple, List, Hashable, Generic

import numpy as np
import pandas as pd

from cgnal.core.typing import K
from cgnal.core.data.model.core import BaseIterable, LazyIterable, CachedIterable
from cgnal.core.utils.dict import union, unflattenKeys


def generate_random_uuid() -> bytes:
    """Create a random number with 12 digits."""
    return uuid.uuid1().bytes[:12]


class Document(Generic[K]):
    """Document representation as couple of uuid and dictionary of information."""

    def __init__(self, uuid: K, data: Dict[str, Any]):
        """
        Return instance of a document.

        :param uuid: document id
        :param data: document data as a dictionary
        """
        self.uuid = uuid
        self.data = data

    def __str__(self) -> str:
        """Return string description of the class."""
        return f"Id: {self.uuid}"

    def getOrThrow(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Retrieve value associated to given key or return default value.

        :param key: key to retrieve
        :param default: default value to return
        :return: retrieve element
        """
        try:
            return self.data[key]
        except KeyError as e:
            if default is not None:
                return default
            else:
                raise e

    def removeProperty(self, key: str) -> "Document":
        """
        Generate new Document instance without given data element.

        :param key: key of data element to remove
        :return: Document without given data element
        """
        return Document(self.uuid, {k: v for k, v in self.data.items() if k != key})

    def addProperty(self, key: str, value: Any) -> "Document":
        """
        Generate new Document instance with given new data element.

        :param key: key of the data element to add
        :param value: value of the data element to add
        :return: Document with new given data element
        """
        return Document(self.uuid, union(self.data, unflattenKeys({key: value})))

    def setRandomUUID(self) -> "Document":
        """
        Generate new document instance with the same data as the current one but with random uuid.

        :return: Document instance with the same data as the current one but with random uuid
        """
        return Document(generate_random_uuid(), self.data)

    @property
    def author(self) -> Optional[str]:
        """
        Retrieve 'author' field.

        :return: author data field value
        """
        return self.getOrThrow("author")

    @property
    def text(self) -> Optional[str]:
        """
        Retrieve 'text' field.

        :return: text data field value
        """
        return self.getOrThrow("text")

    @property
    def language(self) -> Optional[str]:
        """
        Retrieve 'language' field.

        :return: language data field value
        """
        return self.getOrThrow("language")

    def __getitem__(self, item: str) -> Any:
        """
        Get given item from data.

        :param item: key of the data value to return
        :return: data value associated to item key
        """
        return self.data[item]

    @property
    def properties(self) -> Iterator[str]:
        """
        Yield data properties names.

        :return: iterator with data properties names
        """
        for prop in self.data.keys():
            yield prop

    def items(self) -> Iterator[Tuple[str, Any]]:
        """
        Yield data items.

        :return: iterator with tuples of data properties names and values
        """
        for prop in self.properties:
            yield prop, self[prop]


class Documents(BaseIterable[Document], ABC):
    """Base class representing a collection of documents, that is a corpus."""

    @property
    def __lazyType__(self):
        """Specify the type of LazyObject associated to this class."""
        return LazyDocuments

    @property
    def __cachedType__(self):
        """Specify the type of CachedObject associated to this class."""
        return CachedDocuments


class CachedDocuments(CachedIterable[Document], Documents):
    """Class representing a collection of documents cached in memory."""

    @staticmethod
    def __get_key__(key: str, dict: Dict[Hashable, Any]) -> Any:
        """Return the property of the dictionary or nan."""
        try:
            out = dict
            for level in key.split("."):
                out = out[level]
            return out
        except (KeyError, AttributeError):
            return np.nan

    def to_df(self, fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Represent the corpus of documents as a table by unpacking provided fields as columns.

        :param fields: Name of the document property to be unpacked as columns
        :return: dataframe representing the corpus with the given fields
        """
        _fields = fields if fields is not None else []
        return pd.DataFrame.from_dict(
            {
                doc.uuid: {
                    field: self.__get_key__(field, doc.data) for field in _fields
                }
                for doc in self
            },
            orient="index",
        )


class LazyDocuments(LazyIterable[Document], Documents):
    """Class representing a collection of documents provided by a generator."""

    ...
