import json
import unittest

import numpy as np
import pandas as pd

from cgnal.core.data.layer.pandas.dao import DataFrameDAO, SeriesDAO, DocumentDAO
from cgnal.core.data.model.text import Document
from cgnal.core.logging.defaults import getDefaultLogger
from cgnal.core.tests.core import logTest, TestCase

logger = getDefaultLogger()


class TestDocumentDAO(TestCase):
    dict_doc = {"name": "Bob", "languages": ["English", "Fench"]}
    key_doc = "123"
    doc = Document(key_doc, dict_doc)
    series_doc = pd.Series(dict_doc, name=key_doc)

    @logTest
    def test_computeKey(self):
        self.assertEqual(DocumentDAO().computeKey(self.doc), self.key_doc)

    @logTest
    def test_get(self):
        self.assertTrue((DocumentDAO().get(self.doc) == self.series_doc).all())

    @logTest
    def test_parse(self):
        self.assertTrue(isinstance(DocumentDAO().parse(self.series_doc), Document))
        self.assertTrue(DocumentDAO().parse(self.series_doc).data == self.doc.data)


class TestDataFrameDAO(TestCase):
    df1 = pd.DataFrame([[1, 2, 3], [6, 5, 4]], columns=["a", "b", "c"])
    key_df1 = hash(json.dumps({str(k): str(v) for k, v in df1.to_dict().items()}))
    index = [
        np.array(["a", "a", "b", "b", "c", "c"]),
        np.array([0, 1, 0, 1, 0, 1]),
    ]
    series_df1 = pd.Series([1, 6, 2, 5, 3, 4], index=index)
    series_df1.name = key_df1

    @logTest
    def test_computeKey(self):
        self.assertEqual(DataFrameDAO().computeKey(self.df1), self.key_df1)
        df2 = self.df1.copy()
        df2.name = "df2_name"
        self.assertEqual(DataFrameDAO().computeKey(df2), df2.name)

    @logTest
    def test_get(self):
        self.assertTrue(isinstance(DataFrameDAO().get(self.df1), pd.Series))
        self.assertEqual(DataFrameDAO().get(self.df1), self.series_df1)

    @logTest
    def test_parse(self):
        self.assertTrue(isinstance(DataFrameDAO().parse(self.series_df1), pd.DataFrame))
        self.assertEqual(DataFrameDAO().parse(self.series_df1), self.df1)

    @logTest
    def test_addName(self):
        df3 = self.df1.copy()
        DataFrameDAO().addName(df3, "df1_name")
        self.assertEqual(df3.name, "df1_name")


class TestSeriesDAO(TestCase):
    name_s1 = "test_series"
    s1 = pd.Series(np.ones(5), index=range(5), name=name_s1)
    s2 = pd.Series(np.zeros(5), index=range(5))

    @logTest
    def test_computeKey(self):
        self.assertEqual(SeriesDAO().computeKey(self.s1), self.name_s1)
        self.assertRaises(ValueError, lambda: SeriesDAO().computeKey(self.s2))

    @logTest
    def test_get(self):
        self.assertTrue(isinstance(SeriesDAO().get(self.s1), pd.Series))
        self.assertEqual(SeriesDAO().get(self.s1), self.s1)

    @logTest
    def test_parse(self):
        self.assertTrue(isinstance(SeriesDAO().get(self.s1), pd.Series))
        self.assertEqual(SeriesDAO().parse(self.s1), self.s1)

    @logTest
    def test_mapping(self):
        mappings = {k: k + 1 for k in self.s1.index}

        dao = SeriesDAO(mapping=mappings)

        parsed = dao.parse(self.s1)

        self.assertEqual(parsed, self.s1.rename(mappings))
        self.assertEqual(dao.get(parsed), self.s1)

    @logTest
    def test_keys(self):
        keys = [0, 1]

        dao = SeriesDAO(keys=keys)

        parsed = dao.parse(self.s1)

        self.assertEqual(parsed.to_dict(), self.s1.drop(keys).to_dict())
        self.assertEqual(dao.get(parsed).to_dict(), self.s1.to_dict())


if __name__ == "__main__":
    unittest.main()
