import unittest
import os
import pandas as pd

from cgnal.core.tests.core import TestCase, logTest
from cgnal.core.logging.defaults import getDefaultLogger
from cgnal.core.data.layer.pandas.databases import Database, Table
from tests import TMP_FOLDER

logger = getDefaultLogger()

db = Database(TMP_FOLDER + "/db")
df1 = pd.DataFrame([[1, 2, 3], [6, 5, 4]], columns=["a", "b", "c"])
df2 = pd.Series({"a": 1, "b": 2, "c": 3}, name="df2").to_frame()
df3 = pd.DataFrame([[1, 1, 1], [0, 0, 0]], columns=["a", "b", "c"])
table1 = db.table("df1")
table1.write(df1)
table2 = db.table("df2")
table2.write(df2)


class DatabaseTests(TestCase):
    @logTest
    def test_table(self):
        self.assertIsInstance(db.table("df1"), Table)
        self.assertIsInstance(db.table("df2"), Table)
        self.assertEqual(db.table("df1").data, df1)
        self.assertEqual(db.table("df2").data, df2)

    @logTest
    def test__getitem__(self):
        self.assertIsInstance(db.__getitem__("df1"), Table)
        self.assertEqual(db.__getitem__("df1").data, df1)
        self.assertIsInstance(db.__getitem__("df2"), Table)
        self.assertEqual(db.__getitem__("df2").data, df2)

    @logTest
    def test_tables(self):
        self.assertEqual(set(db.tables), set(["df2", "df1"]))


class TableTests(TestCase):
    @logTest
    def test_filename(self):
        self.assertEqual(table1.filename, TMP_FOLDER + "/db/df1.p")
        self.assertEqual(table2.filename, TMP_FOLDER + "/db/df2.p")

    @logTest
    def test_to_df(self):
        self.assertIsInstance(db.table("df1").to_df(), pd.DataFrame)
        self.assertIsInstance(db.table("df2").to_df(), pd.DataFrame)
        self.assertEqual(db.table("df1").to_df(), df1)
        self.assertEqual(db.table("df2").to_df(), df2)
        self.assertEqual(
            db.table("df1").to_df("b > a"),
            pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"]),
        )

    @logTest
    def test_data(self):
        self.assertIsInstance(db.table("df1").to_df(), pd.DataFrame)
        self.assertIsInstance(db.table("df2").to_df(), pd.DataFrame)
        self.assertEqual(db.table("df1").to_df(), df1)
        self.assertEqual(db.table("df2").to_df(), df2)

    @logTest
    def test_write(self):
        self.assertTrue(os.path.exists(TMP_FOLDER + "/db/df1.p"))
        self.assertIsInstance(db.table("df1"), Table)
        self.assertIsInstance(db.table("df2"), Table)
        self.assertEqual(db.table("df1").data, df1)
        self.assertEqual(db.table("df2").data, df2)

        data_tmp = table1.data
        df3 = pd.DataFrame([[1, 1, 1], [0, 0, 0]], columns=["a", "b", "c"])
        table1.write(df3)

        self.assertEqual(table1.data, pd.concat([data_tmp, df3]))


if __name__ == "__main__":
    unittest.main()
