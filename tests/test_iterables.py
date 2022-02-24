import unittest

from cgnal.core.data.model.text import Document, CachedDocuments, LazyDocuments
from cgnal.core.logging.defaults import getDefaultLogger
from cgnal.core.tests.core import logTest, TestCase

logger = getDefaultLogger()

n = 10


def createCorpus(n):
    for i in range(n):
        yield Document(str(i), {"text": "my text 1"})


class TestDocuments(TestCase):
    docs = (
        CachedDocuments(createCorpus(n))
        .map(lambda x: x.addProperty("tags", {"1": "1"}))
        .map(lambda x: x.addProperty("tags", {"2": "2"}))
    )

    @logTest
    def test_documents_parsing(self):
        filteredDocs = self.docs.filter(lambda x: int(x.uuid) % 2)
        self.assertTrue(isinstance(filteredDocs, LazyDocuments))
        self.assertEqual(len(filteredDocs.asCached), n / 2)

    @logTest
    def test_documents_cached(self):
        filteredDocs = self.docs.filter(lambda x: int(x.uuid) % 2).asCached
        self.assertTrue(isinstance(filteredDocs, CachedDocuments))


if __name__ == "__main__":
    unittest.main()
