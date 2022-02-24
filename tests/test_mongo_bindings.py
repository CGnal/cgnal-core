import unittest

from cgnal.core.data.layer.mongo.archivers import MongoArchiver
from cgnal.core.data.layer.mongo.dao import DocumentDAO
from cgnal.core.data.model.text import Document, generate_random_uuid
from tests import db
from cgnal.core.tests.core import TestCase, logTest


class TestMongoConnection(TestCase):
    @logTest
    def test_basic_operation(self):
        self.assertTrue(len(db.list_collection_names()) == 0)

        collection = db["test"]

        collection.insert_one({"test": "This is a test"})

        self.assertEqual(len(db.list_collection_names()), 1)
        self.assertEqual(collection.count_documents({}), 1)

        db.drop_collection(collection.name)

        self.assertEqual(len(db.list_collection_names()), 0)

    @logTest
    def test_mongo_archiver(self):
        doc = Document(
            generate_random_uuid(),
            {"title": "This is a title", "text": "This is a text"},
        )

        archiver = MongoArchiver(db["documents"], DocumentDAO())

        archiver.archiveOne(doc)

        doc2 = next(archiver.retrieve())

        self.assertEqual(doc["text"], doc2["text"])
        self.assertEqual(doc["title"], doc2["title"])


if __name__ == "__main__":
    unittest.main()
