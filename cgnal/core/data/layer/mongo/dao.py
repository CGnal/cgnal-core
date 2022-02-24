"""Module with the implementation and abstraction for serializing/deserializing objects to/from MongoDB."""

from bson.objectid import ObjectId
import pandas as pd

from typing import Dict, Generic, TypeVar, Hashable, Any
from cgnal.core.typing import T
from cgnal.core.data.layer import DAO
from cgnal.core.utils.dict import union
from cgnal.core.data.model.text import Document

K = TypeVar("K", Hashable, Hashable)


class MongoDAO(DAO[T, dict]):
    """Base Serialized/Deserializer for objects to/from Mongo."""

    ...


class DocumentDAO(MongoDAO[Document]):
    """Base class for serializing/deserializing documents from/to MongoDB."""

    mapping: Dict[str, str] = {}

    @property
    def inverse_mapping(self) -> Dict[str, Any]:
        """
        Apply inverse of self.mapping.

        :return: dictionary with self.mapping.keys as values and self.mapping.values as keys
        """
        return {v: k for k, v in self.mapping.items()}

    def translate(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map dictionary keys according to self.mapping.

        :param d: dict to map
        :return: dictionary with mapped keys
        """
        return {self.mapping.get(k, k): v for k, v in d.items()}

    def conversion(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map dictionary keys according to the inverse of self.mapping.

        :param d: dict to map
        :return: dictionary with mapped keys
        """
        return {self.inverse_mapping.get(k, k): v for k, v in d.items()}

    def __init__(self, uuid: str = "_id") -> None:
        """
        Serialize a Domain object into an instance of :class:`cgnal.data.model.text.Document`.

        :param uuid: sets the name of the uuid field as str
        """
        self.uuid = uuid

    # TODO the output value of this method is a bit inconsistent with the one of following class (SeriesDao) and
    #  the one of pandas.dao.DocumentDAO. Wouldn't it be better for computeKey method to have the same signature for all
    #  DAOs (in all modules), defined in DAO ABC, as something on the line of
    #  def computeKey(self, obj: DataVal) -> Hashable
    #  where DataVal = TypeVar('DataVal', Document, pd.DataFrame, pd.Series)?
    def computeKey(self, obj: Document) -> Dict[str, ObjectId]:
        """
        Get document id as dictionary.

        :param obj: document whose id is to retrieve
        :return: dictionary with '_id' key and ObjectId as value
        """
        return {"_id": ObjectId(obj.uuid)}

    def get(self, obj: Document) -> Dict[str, Any]:
        """
        Get a document as a dictionary.

        :param obj: a Document object to be transformed into a dictionary
        :return: document object as a dictionary
        """
        return self.conversion(union(obj.data, self.computeKey(obj)))

    def parse(self, json: Dict[str, Any]) -> Document:
        """
        Get a dictionary as a Document object.

        :param json: a json/dictionary to be parsed
        :return: an object of :class:`cgnal.data.model.text.Document`
        """
        translated = self.translate(json)
        return Document(str(translated[self.uuid]), self.translate(json))


class SeriesDAO(DAO, Generic[K, T]):
    """Base class for serializing/deserializing pandas Series from/to MongoDB."""

    def __init__(self, key_field: str = "_id") -> None:
        """
        Serialize a Domain object into an instance of :class: pd.Series.

        :param key_field: sets the name of the key_field field as str
        """
        self.key_field = key_field

    def computeKey(self, serie: pd.Series) -> Dict[str, ObjectId]:
        """
        Get series id as a dictionary. The id is key_field.

        :param serie: the series whose id is to be retrieved
        :return: dictionary with self.key_field as key and name of the series as value
        """
        return {self.key_field: ObjectId(serie.name)}

    def get(self, serie: pd.Series) -> Dict[K, T]:
        """
        Get a series as a dictionary.

        :param serie: pd.Series
        :return: dictionary
        """
        data: Dict[K, T] = serie.to_dict()
        return data

    def parse(self, json: Dict[K, T]) -> pd.Series:
        """
        Get a json as a pd.Series.

        :param json: dictionary
        :return: pd.Series
        """
        s = pd.Series(json)
        s.name = s.pop(self.key_field)
        return s
