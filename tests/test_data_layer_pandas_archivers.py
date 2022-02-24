import os
import shutil
import unittest

import numpy as np
import pandas as pd

from cgnal.core.data.layer.pandas.archivers import (
    TableArchiver,
    PickleArchiver,
    CsvArchiver,
)
from cgnal.core.data.layer.pandas.dao import DataFrameDAO, SeriesDAO, DocumentDAO
from cgnal.core.data.layer.pandas.databases import Database
from cgnal.core.data.model.core import IterGenerator
from cgnal.core.logging.defaults import getDefaultLogger
from cgnal.core.tests.core import TestCase, logTest
from tests import TMP_FOLDER, DATA_FOLDER

TEST_DATA_PATH = DATA_FOLDER
logger = getDefaultLogger()


class TestTableArchiver(TestCase):
    db = Database(TMP_FOLDER)
    table1 = db.table("my_table_1")
    table2 = db.table("my_table_2")

    df1 = pd.DataFrame({"a": np.ones(5), "b": 2 * np.ones(5)})
    df1.name = "row1"
    df2 = df1 * 2
    df2.name = "row2"
    s1 = pd.Series(np.ones(5), index=np.arange(0, 10, 2))
    s1.name = "s1"
    s2 = s1 * 2
    s2.name = "s2"

    dao_df = DataFrameDAO()
    dao_series = SeriesDAO()

    @logTest
    def test_write_read(self):
        table3 = self.db.table("my_table_3")
        a = TableArchiver(table3, self.dao_df)
        a.archive([self.df1]).__write__()
        self.assertEqual(
            a.__read__()["a"], pd.DataFrame({"row1": [1.0, 1.0, 1.0, 1.0, 1.0]}).T
        )
        self.assertEqual(
            a.__read__()["b"], pd.DataFrame({"row1": [2.0, 2.0, 2.0, 2.0, 2.0]}).T
        )

        table4 = self.db.table("my_table_4")
        a = TableArchiver(table4, self.dao_series)
        a.archive([self.s1]).__write__()
        self.assertEqual(a.__read__(), pd.DataFrame(self.s1).T)

    @logTest
    def test_retrieve(self):
        a = TableArchiver(self.table1, self.dao_df)
        a.archiveOne(self.df1)
        self.assertEqual(self.df1, next(a.retrieve()))

        def multiply(df):
            return df * 2

        self.assertEqual(self.df1 * 2, next(a.retrieve(multiply)))

    @logTest
    def test_data(self):
        table3 = self.db.table("my_table_3")
        a = TableArchiver(table3, self.dao_df)
        a.archive([self.df1]).__write__()
        self.assertEqual(a.data, a.__read__())

        table4 = self.db.table("my_table_4")
        a = TableArchiver(table4, self.dao_series)
        a.archive([self.s1]).__write__()
        self.assertEqual(a.data, a.__read__())

        a.data = self.df2
        self.assertEqual(a.data, self.df2)

    @logTest
    def test_archiveOne(self):
        a = TableArchiver(self.table1, self.dao_df)
        a.archiveOne(self.df1)
        retrieved = a.retrieveById(self.df1.name)
        self.assertEqual(self.df1, retrieved)

        b = TableArchiver(self.table2, self.dao_series)
        b.archiveOne(self.s1)
        retrieved = b.retrieveById(self.s1.name)
        self.assertEqual(self.s1, retrieved)

    @logTest
    def test_archiveMany(self):
        a = TableArchiver(self.table1, self.dao_df)
        a.archiveMany([self.df1, self.df2])
        retrieved = a.retrieve()
        self.assertEqual(self.df1, next(retrieved))
        self.assertEqual(self.df2, next(retrieved))

        b = TableArchiver(self.table2, self.dao_series)
        b.archiveMany([self.s1, self.s2])
        retrieved = b.retrieve()
        self.assertEqual(self.s1, next(retrieved))
        self.assertEqual(self.s2, next(retrieved))

    @logTest
    def test_archive(self):
        a = TableArchiver(self.table1, self.dao_df)
        a.archive([self.df1, self.df2])
        retrieved = a.retrieve()
        self.assertEqual(self.df1, next(retrieved))
        self.assertEqual(self.df2, next(retrieved))

        b = TableArchiver(self.table2, self.dao_series)
        b.archive([self.s1])
        retrieved = b.retrieve()
        self.assertEqual(self.s1, next(retrieved))

    @logTest
    def test_retrieveGenerator(self):
        a = TableArchiver(self.table1, self.dao_df)
        a.archiveOne(self.df1)

        self.assertIsInstance(a.retrieveGenerator(), IterGenerator)
        self.assertEqual(self.df1, next(a.retrieveGenerator().iterator))

        def multiply(df):
            return df * 2

        self.assertEqual(self.df1 * 2, next(a.retrieveGenerator(multiply).iterator))

    @logTest
    def test_commit(self):
        table3 = self.db.table("my_table_3")
        a = TableArchiver(table3, self.dao_df)
        a.archive([self.df1]).commit()
        self.assertEqual(
            a.__read__()["a"], pd.DataFrame({"row1": [1.0, 1.0, 1.0, 1.0, 1.0]}).T
        )
        self.assertEqual(
            a.__read__()["b"], pd.DataFrame({"row1": [2.0, 2.0, 2.0, 2.0, 2.0]}).T
        )

        table4 = self.db.table("my_table_4")
        a = TableArchiver(table4, self.dao_series)
        a.archive([self.s1]).commit()
        self.assertEqual(a.__read__(), pd.DataFrame(self.s1).T)

    @logTest
    def test_retrieveById(self):
        a = TableArchiver(self.table1, self.dao_df)
        a.archiveOne(self.df1)
        df3 = a.retrieveById(self.df1.name)
        self.assertEqual(self.df1, df3)

        b = TableArchiver(self.table2, self.dao_series)
        b.archiveMany([self.s1, self.s2])
        retrieved1 = b.retrieveById(self.s1.name)
        retrieved2 = b.retrieveById(self.s2.name)
        self.assertEqual(self.s1, retrieved1)
        self.assertEqual(self.s2, retrieved2)


class TestPickleArchiver(TestCase):
    df1 = pd.DataFrame({"row1": np.ones(5), "row2": 2 * np.ones(5)}).T
    df2 = pd.DataFrame({"a": 5 * np.ones(5), "b": 6 * np.ones(5)})
    df2.name = "row1"
    s1 = pd.Series(3 * np.ones(5), index=np.arange(0, 5), name="row3")
    s2 = pd.Series(4 * np.ones(5), index=np.arange(0, 5), name="row4")
    dao_series = SeriesDAO()
    dao_df = DataFrameDAO()

    @logTest
    def test_data(self):
        self.df1.to_pickle(TMP_FOLDER + "/df1.pkl")
        a = PickleArchiver(TMP_FOLDER + "/df1.pkl", SeriesDAO())
        self.assertEqual(a.data, self.df1)

    @logTest
    def test_archive(self):
        self.df1.to_pickle(TMP_FOLDER + "/df1.pkl")
        a = PickleArchiver(TMP_FOLDER + "/df1.pkl", SeriesDAO())

        a.archive([self.s1])
        self.assertEqual(a.retrieveById("row3"), self.s1)

    @logTest
    def test__write__read__(self):
        self.df1.to_pickle(os.path.join(TMP_FOLDER, "df1.pkl"))
        a = PickleArchiver(os.path.join(TMP_FOLDER, "df1.pkl"), SeriesDAO())
        a.archive([self.s1]).__write__()
        self.assertEqual(a.__read__(), a.data)

    @logTest
    def test_commit(self):
        self.df1.to_pickle(TMP_FOLDER + "/df1.pkl")
        a = PickleArchiver(TMP_FOLDER + "/df1.pkl", SeriesDAO())
        a.archiveOne(self.s1).commit()
        self.assertEqual(a.data, a.__read__())

        b = PickleArchiver(os.path.join(TMP_FOLDER, "df2.pkl"), SeriesDAO())
        b.archiveOne(self.s1).commit()
        self.assertTrue(os.path.exists(os.path.join(TMP_FOLDER, "df2.pkl")))

    @logTest
    def test_retrieve(self):
        self.df1.to_pickle(TMP_FOLDER + "/df1.pkl")
        a = PickleArchiver(TMP_FOLDER + "/df1.pkl", SeriesDAO())
        retrieved = a.retrieve()
        row1 = pd.Series(np.ones(5), index=np.arange(0, 5), name="row1")
        self.assertEqual(next(retrieved), row1)
        self.assertEqual(
            next(retrieved),
            pd.Series(2 * np.ones(5), index=np.arange(0, 5), name="row2"),
        )

        def multiply(series):
            return series * 2

        self.assertEqual(row1 * 2, next(a.retrieve(multiply)))

    @logTest
    def test_archiveOne(self):
        self.df1.to_pickle(TMP_FOLDER + "/df1.pkl")
        a = PickleArchiver(TMP_FOLDER + "/df1.pkl", SeriesDAO())
        a.archiveOne(self.s1)
        self.assertEqual(a.retrieveById("row3"), self.s1)

    @logTest
    def test_archiveMany(self):
        self.df1.to_pickle(TMP_FOLDER + "/df1.pkl")
        a = PickleArchiver(TMP_FOLDER + "/df1.pkl", SeriesDAO())
        _ = a.data
        a.archiveMany([self.s1, self.s2])
        self.assertEqual(a.retrieveById("row3"), self.s1)
        self.assertEqual(a.retrieveById("row4"), self.s2)

    @logTest
    def test_retrieveGenerator(self):
        self.df1.to_pickle(TMP_FOLDER + "/df1.pkl")
        a = PickleArchiver(TMP_FOLDER + "/df1.pkl", SeriesDAO())

        self.assertIsInstance(a.retrieveGenerator(), IterGenerator)

        retrieved = a.retrieveGenerator().iterator
        self.assertEqual(next(retrieved), a.retrieveById("row1"))
        self.assertEqual(next(retrieved), a.retrieveById("row2"))

        def multiply(series):
            return series * 2

        retrieved = a.retrieveGenerator(multiply).iterator
        self.assertEqual(next(retrieved), a.retrieveById("row1") * 2)
        self.assertEqual(next(retrieved), a.retrieveById("row2") * 2)

    @logTest
    def test_retrieveById(self):
        self.df1.to_pickle(TMP_FOLDER + "/df1.pkl")
        a = PickleArchiver(TMP_FOLDER + "/df1.pkl", SeriesDAO())
        self.assertEqual(
            a.retrieveById("row1"),
            pd.Series(np.ones(5), index=np.arange(0, 5), name="row1"),
        )
        self.assertEqual(
            a.retrieveById("row2"),
            pd.Series(2 * np.ones(5), index=np.arange(0, 5), name="row2"),
        )

    @logTest
    def test_archive_not_existing_file(self):
        filename = os.path.join(TMP_FOLDER, "test.p")
        self.assertFalse(os.path.exists(filename))

        serie = pd.Series({"b": 1, "c": 2})
        archiver = PickleArchiver(filename, SeriesDAO(mapping={"a": "b"}))

        archiver.archiveOne(serie)
        archiver.commit()

        self.assertTrue(os.path.exists(filename))
        self.assertTrue("a" in pd.read_pickle(filename).columns)

        os.remove(filename)
        self.assertFalse(os.path.exists(filename))

    @logTest
    def test_multikey_index(self):
        filename = os.path.join(TMP_FOLDER, "test.p")
        self.assertFalse(os.path.exists(filename))

        serie = pd.Series({"b": 1, "c": 2}, name=(2,))
        archiver = PickleArchiver(filename, SeriesDAO(mapping={"a": "b"}, keys=["f"]))

        archiver.archiveOne(serie)
        archiver.commit()

        self.assertTrue(os.path.exists(filename))

        data: pd.DataFrame = pd.read_pickle(filename)
        self.assertEqual(data.iloc[0]["f"], 2)

        retrieved = next(archiver.retrieve())

        self.assertEqual(serie, retrieved)

        os.remove(filename)
        self.assertFalse(os.path.exists(filename))


class TestCsvArchiver(TestCase):
    dao = DocumentDAO()
    shutil.copy(DATA_FOLDER + "/test.csv", TMP_FOLDER + "/test_copy.csv")
    a = CsvArchiver(os.path.join(TMP_FOLDER, "test_copy.csv"), dao)

    @logTest
    def test_data(self):
        self.assertTrue(
            (
                self.a.data.columns
                == [
                    "date",
                    "hashtags",
                    "reply_to",
                    "symbols",
                    "text",
                    "user",
                    "user_mentions",
                ]
            ).all()
        )
        self.assertEqual(len(self.a.data), 20)
        self.assertEqual(self.a.data.loc[974782389837844480]["user"], "MasteredMedia")

    @logTest
    def test_archive(self):
        doc = next(self.a.retrieve())
        doc.data.update({"symbols": ["test_symbol"]})
        self.a.archive(doc)
        self.assertEqual(self.a.retrieveById(doc.uuid).data["symbols"], ["test_symbol"])

    @logTest
    def test__write__read__(self):
        doc = next(self.a.retrieve())
        doc.data.update({"symbols": ["test_symbol"]})
        self.a.archive(doc).__write__()
        self.assertEqual(self.a.__read__().loc[doc.uuid]["symbols"], "['test_symbol']")

    @logTest
    def test_commit(self):
        doc = next(self.a.retrieve())
        doc.data.update({"symbols": ["test_symbol_2"]})
        self.a.archive(doc).commit()
        self.assertEqual(
            self.a.__read__().loc[doc.uuid]["symbols"], "['test_symbol_2']"
        )

    @logTest
    def test_retrieve(self):
        doc = [i for i in self.a.retrieve() if i.uuid == 974782424998703105][0]
        self.assertEqual(doc.data["user"], "LinkatalogEn")

        def update_data(data):
            df = data.copy()
            df.iloc[0]["symbols"] = ["test_symbol_6"]
            df.iloc[0]["user"] = "007BondJames"
            return df

        self.assertEqual(
            next(self.a.retrieve(update_data)).data["symbols"], ["test_symbol_6"]
        )
        self.assertEqual(
            next(self.a.retrieve(condition=update_data, sort_by=["user"])).data["user"],
            "007BondJames",
        )

    @logTest
    def test_archiveOne(self):
        doc = next(self.a.retrieve())
        doc.data.update({"symbols": ["test_symbol_3"]})
        self.a.archiveOne(doc)
        self.assertEqual(self.a.retrieveById(doc.uuid)["symbols"], ["test_symbol_3"])

    @logTest
    def test_archiveMany(self):
        doc_generator = self.a.retrieve()
        doc1 = next(doc_generator)
        doc1.data.update({"symbols": ["test_symbol_4"]})
        doc2 = next(doc_generator)
        doc2.data.update({"symbols": ["test_symbol_5"]})
        self.a.archiveMany([doc1, doc2])
        self.assertEqual(self.a.retrieveById(doc1.uuid)["symbols"], ["test_symbol_4"])
        self.assertEqual(self.a.retrieveById(doc2.uuid)["symbols"], ["test_symbol_5"])

    @logTest
    def test_retrieveGenerator(self):
        self.assertIsInstance(self.a.retrieveGenerator(), IterGenerator)
        doc = [
            i
            for i in self.a.retrieveGenerator().iterator
            if i.uuid == 974782424998703105
        ][0]
        self.assertEqual(doc.data["user"], "LinkatalogEn")

        def update_data(data):
            df = data.copy()
            df.iloc[0]["symbols"] = ["test_symbol_6"]
            return df

        self.assertEqual(
            next(self.a.retrieveGenerator(update_data).iterator).data["symbols"],
            ["test_symbol_6"],
        )

    @logTest
    def test_retrieveById(self):
        self.assertEqual(
            self.a.retrieveById(974782473274970112).data["user"], "lovegawria"
        )


if __name__ == "__main__":
    unittest.main()
