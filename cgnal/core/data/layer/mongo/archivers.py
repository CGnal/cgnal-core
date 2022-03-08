"""Module with abstraction for accessing to MongoDB persistent layers."""

from collections.abc import Iterable
from bson.objectid import ObjectId
from pymongo.collection import Collection, UpdateResult
from typing import (
    Union,
    Optional,
    List,
    Dict,
    Any,
    Iterator,
    Iterable as IterableType,
    TYPE_CHECKING,
)
from cgnal.core.data.layer import Archiver
from cgnal.core.data.layer.mongo.dao import MongoDAO
from cgnal.core.typing import T

if TYPE_CHECKING:
    from mongomock.collection import Collection as MockCollection


class MongoArchiver(Archiver[T]):
    """Archiver based on MongoDB persistent layers."""

    def __init__(
        self, collection: Union[Collection, "MockCollection"], dao: MongoDAO
    ) -> None:
        """
        Return an instance of the archiver to access and modify Mongodb collections via a DAO object.

        :param collection: A Mongodb collection
        :param dao: An instance of :class:`cgnal.data.layer.mongo.dao.DocumentDao` or
            :class:`cgnal.data.layer.mongo.dao.SeriesDAO`  that helps to retrieve/archive a document.
        """
        self.collection = collection
        self.dao = dao

    def retrieveById(self, uuid: str) -> T:
        """
        Retrive document from collection by id.

        :param uuid: document id
        :return: retrieved document parsed according to self.dao
        """
        json = self.collection.find_one({"_id": ObjectId(uuid)})
        return self.dao.parse(json)

    def retrieve(
        self,
        condition: Dict[str, Dict[str, Any]] = {},
        sort_by: Optional[Union[str, List[str]]] = None,
    ) -> Iterator[T]:
        """
        Retrieve documents satisfying condition, sorted according to given ordering.

        :param condition: condition to satisfy. If {}, return all documents.
        :param sort_by: ordering to respect. If None, no ordering is given.
        :return: iterator of (ordered) documents satisfying given condition
        """
        jsons = self.collection.find(condition, no_cursor_timeout=True)
        if sort_by is not None:
            jsons = jsons.sort(sort_by)
        for json in jsons:
            yield self.dao.parse(json)
        jsons.close()

    def archiveOne(self, obj: T) -> UpdateResult:
        """
        Archive one document in collection.

        :param obj: document to archive
        :return: an instance of :class:`pymongo.results.UpdateResult` with update operation's results
        """
        return self.__insert__(obj)

    def __insert__(self, obj: T) -> UpdateResult:
        """
        Insert one document in collection.

        :param obj: document to archive
        :return: an instance of :class:`pymongo.results.UpdateResult` with update operation's results
        """
        return self.collection.update_one(
            self.dao.computeKey(obj), {"$set": self.dao.get(obj)}, upsert=True
        )

    def archiveMany(self, objs: IterableType[T]) -> List[UpdateResult]:
        """
        Insert many documents in collection.

        :param objs: documents to archive
        :return: list of instances of :class:`pymongo.results.UpdateResult` with update operations' results
        """
        return [self.__insert__(obj) for obj in objs]

    # TODO this method's output type is not consistent with its' ancestor's return type (that should be 'MongoArchiver')
    def archive(
        self, objs: Union[T, IterableType[T]]
    ) -> Union[UpdateResult, List[UpdateResult]]:
        """
        Archive one or more documents in collection.

        :param objs: documents to archive
        :return: list of instances of :class:`pymongo.results.UpdateResult` with update operations' results
        """
        if isinstance(objs, Iterable):
            return self.archiveMany(objs)
        else:
            return self.archiveOne(objs)

    def first(self) -> T:
        """
        Retrieve first element in collection.

        :return: parsed document
        """
        json = self.collection.find_one()
        return self.dao.parse(json)

    def aggregate(
        self, pipeline: List[Dict[str, Dict[str, Any]]], allowDiskUse: bool = True
    ) -> Iterator[T]:
        """
        Aggregate collection's documents using given aggregation steps.

        :param pipeline: a list of aggregation pipeline stages
        :param allowDiskUse: Enables writing to temporary files. When set to `True`, aggregation stages can write data to
            the _tmp subdirectory of the --dbpath directory. The default is False.
        :return: iterator with parsed aggregated documents
        """
        jsons = self.collection.aggregate(pipeline, allowDiskUse=allowDiskUse)
        for json in jsons:
            yield self.dao.parse(json)
        jsons.close()
