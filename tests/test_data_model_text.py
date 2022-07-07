import unittest
from typing import Iterator

import pandas as pd
from cgnal.core.tests.core import TestCase, logTest

from tests import TMP_FOLDER
from cgnal.core.data.model.text import (
    generate_random_uuid,
    Document,
    CachedDocuments,
    LazyDocuments,
)
from cgnal.core.data.model.core import IterGenerator

dict_doc1 = {"name": "Bob", "language": ["English", "French"]}
key_doc1 = "123"
dict_doc2 = {"name": "Alice", "language": ["Spanish", "German"]}
key_doc2 = "456"
doc1 = Document(key_doc1, dict_doc1)
doc2 = Document(key_doc2, dict_doc2)


def samples_gen():
    for i in [doc1, doc2]:
        yield i


cached_doc = CachedDocuments([doc1, doc2])
lazy_doc = LazyDocuments(IterGenerator(samples_gen))


class TestGenerate_random_uuid(TestCase):
    @logTest
    def test_Generate_random_uuid(self):
        self.assertTrue(len(generate_random_uuid()), 12)
        self.assertIsInstance(generate_random_uuid(), bytes)


class TestDocument(TestCase):

    dict_doc = {"name": "Bob", "language": ["English", "French"]}
    key_doc = "123"
    doc = Document(key_doc, dict_doc)

    @logTest
    def test__str__(self):
        self.assertEqual(self.doc.__str__(), "Id: 123")

    @logTest
    def test_getOrThrow(self):
        self.assertEqual(self.doc.getOrThrow("name", "empty"), "Bob")
        self.assertEqual(self.doc.getOrThrow("age", "empty"), "empty")

    @logTest
    def test_removeProperty(self):
        doc_new = self.doc.removeProperty("name")
        self.assertEqual(doc_new.getOrThrow("name", "empty"), "empty")

    @logTest
    def test_addProperty(self):
        doc_new = self.doc.addProperty("age", "25")
        self.assertEqual(doc_new.getOrThrow("age", "empty"), "25")

    @logTest
    def test_setRandomUUID(self):
        doc_new = self.doc.setRandomUUID()
        self.assertNotEqual(doc_new.uuid, self.doc.uuid)
        self.assertEqual(doc_new.data, self.doc.data)

    @logTest
    def author(self):
        doc_new = self.doc.addProperty("author", "Gioia")
        self.assertEqual(doc_new.author, "Gioia")

    @logTest
    def test_text(self):
        doc_new = self.doc.addProperty("text", "hello")
        self.assertEqual(doc_new.text, "hello")

    @logTest
    def test_language(self):
        self.assertEqual(self.doc.language, ["English", "French"])

    @logTest
    def test__getitem__(self):
        self.assertEqual(self.doc.__getitem__("name"), "Bob")

    @logTest
    def test_properties(self):
        self.assertEqual(list(self.doc.properties), ["name", "language"])

    @logTest
    def test_items(self):
        self.assertEqual(
            list(self.doc.items()),
            [("name", "Bob"), ("language", ["English", "French"])],
        )


class TestCachedDocuments(TestCase):
    @logTest
    def test_lazyType(self):
        self.assertIsInstance(
            cached_doc.lazy_type(IterGenerator(samples_gen)), LazyDocuments
        )

    @logTest
    def test_cachedType(self):
        self.assertIsInstance(cached_doc.cached_type([doc1, doc2]), CachedDocuments)

    @logTest
    def test_toLazy(self):
        self.assertIsInstance(cached_doc.to_lazy(), LazyDocuments)
        self.assertEqual(next(cached_doc.to_lazy().items).data, dict_doc1)

    @logTest
    def test_asCached(self):
        self.assertIsInstance(cached_doc.to_cached(), CachedDocuments)
        self.assertEqual(cached_doc.to_cached().items[1].data, dict_doc2)

    @logTest
    def test_take(self):
        self.assertEqual(len(cached_doc.take(1).items), 1)
        self.assertEqual(cached_doc.take(1).items[0].data, dict_doc1)

    @logTest
    def test_filter(self):
        def func(doc):
            if doc.language == ["English", "French"]:
                return True

        self.assertEqual(len(list(cached_doc.filter(func))), 1)
        self.assertEqual(list(cached_doc.filter(func))[0].data, dict_doc1)

    @logTest
    def test__iter__(self):
        self.assertIsInstance(cached_doc.__iter__(), Iterator)
        self.assertEqual(list(cached_doc.__iter__())[0].data, dict_doc1)
        self.assertEqual(list(cached_doc.__iter__())[1].data, dict_doc2)

    @logTest
    def test_batch(self):
        self.assertEqual(len(next(cached_doc.batch(1)).items), 1)
        self.assertEqual(len(next(cached_doc.batch(2)).items), 2)
        self.assertEqual(next(cached_doc.batch(1)).items[0].data, dict_doc1)

    @logTest
    def test_map(self):
        def func(doc: Document):
            if doc.language == ["English", "French"]:
                return doc.addProperty("language", ["Italian"])

        self.assertEqual(list(cached_doc.map(func).items)[0].language, ["Italian"])

    @logTest
    def test_foreach(self):

        lst = []

        def func(doc):
            lst.append(doc.data["name"])

        cached_doc.foreach(func)
        self.assertEqual(lst, ["Bob", "Alice"])

    @logTest
    def test_write_load(self):
        cached_doc.write(TMP_FOLDER + "/test_file.pkl")
        dict_doc3 = {"name": "Bob"}
        new_doc = CachedDocuments(dict_doc3).load(TMP_FOLDER + "/test_file.pkl")
        self.assertEqual(new_doc.items[0].data, dict_doc1)
        self.assertEqual(new_doc.items[1].data, dict_doc2)

    @logTest
    def test_get_key(self):
        self.assertEqual(cached_doc._get_key("name", dict_doc1), "Bob")

    @logTest
    def test_to_df(self):
        self.assertEqual(
            cached_doc.to_df(["name", "language"]),
            pd.DataFrame([dict_doc1, dict_doc2], index=[key_doc1, key_doc2]),
        )

    def test__len__(self):
        self.assertEqual(cached_doc.__len__(), 2)

    def test_items(self):
        self.assertEqual(cached_doc.items[0].data, dict_doc1)
        self.assertEqual(cached_doc.items[1].data, dict_doc2)

    def test_cached(self):
        self.assertTrue(cached_doc.cached)

    def test__getitem__(self):
        self.assertEqual(cached_doc.__getitem__(0).data, dict_doc1)
        self.assertEqual(cached_doc.__getitem__(1).data, dict_doc2)


class TestLazyDocuments(TestCase):
    @logTest
    def test_lazyType(self):
        self.assertIsInstance(
            lazy_doc.lazy_type(IterGenerator(samples_gen)), LazyDocuments
        )

    @logTest
    def test_cachedType(self):
        self.assertIsInstance(
            lazy_doc.cached_type([dict_doc1, dict_doc2]), CachedDocuments
        )

    @logTest
    def test_toLazy(self):
        self.assertIsInstance(lazy_doc.to_lazy(), LazyDocuments)
        self.assertEqual(next(lazy_doc.to_lazy().items).data, dict_doc1)

    @logTest
    def test_toCached(self):
        self.assertIsInstance(lazy_doc.to_cached(), CachedDocuments)
        self.assertEqual(lazy_doc.to_cached().items[1].data, dict_doc2)

    @logTest
    def test_take(self):
        self.assertEqual(len(lazy_doc.take(1).items), 1)
        self.assertEqual(lazy_doc.take(1).items[0].data, dict_doc1)

    @logTest
    def test_filter(self):
        def func(doc):
            if doc.language == ["English", "French"]:
                return True

        self.assertEqual(len(list(lazy_doc.filter(func))), 1)
        self.assertEqual(list(lazy_doc.filter(func))[0].data, dict_doc1)

    @logTest
    def test__iter__(self):
        self.assertIsInstance(lazy_doc.__iter__(), Iterator)
        self.assertEqual(list(lazy_doc.__iter__())[0].data, dict_doc1)
        self.assertEqual(list(lazy_doc.__iter__())[1].data, dict_doc2)

    @logTest
    def test_batch(self):
        self.assertEqual(len(next(lazy_doc.batch(1)).items), 1)
        self.assertEqual(len(next(lazy_doc.batch(2)).items), 2)
        self.assertEqual(next(lazy_doc.batch(1)).items[0].data, dict_doc1)

    @logTest
    def test_map(self):
        def func(doc: Document):
            if doc.language == ["English", "French"]:
                return doc.addProperty("language", ["Italian"])

        self.assertEqual(list(lazy_doc.map(func).items)[0].language, ["Italian"])

    @logTest
    def test_foreach(self):

        lst = []

        def func(doc):
            lst.append(doc.data["name"])

        lazy_doc.foreach(func)
        self.assertEqual(lst, ["Bob", "Alice"])

    def test_items(self):
        generator = lazy_doc.items
        self.assertEqual(next(generator).data, dict_doc1)
        self.assertEqual(next(generator).data, dict_doc2)

    def test_cached(self):
        self.assertFalse(lazy_doc.cached)


if __name__ == "__main__":
    unittest.main()
