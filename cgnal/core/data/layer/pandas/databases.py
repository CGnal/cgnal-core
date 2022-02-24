"""Module with abstraction for databases and tables."""

import os
import pandas as pd
from glob import glob

from typing import List, Optional
from cgnal.core.typing import PathLike
from cgnal.core.utils.fs import create_dir_if_not_exists
from cgnal.core.data.layer import DatabaseABC, TableABC
from cgnal.core.logging.defaults import WithLogging


class Database(WithLogging, DatabaseABC):
    """Class representing a Database object."""

    def __init__(self, name: PathLike, extension: str = ".p") -> None:
        """
        Return an instance of a class implementing standard read and write methods to pickle data sources.

        :param name: path to pickles
        :param extension: standard pickle extension
        """
        if not os.path.exists(name):
            self.logger.info(f"Creating new database {name}")
        self.name = create_dir_if_not_exists(name)
        self.extension = extension

    @property
    def tables(self) -> List[str]:
        """
        Complete pickle names with appropriate extension.

        :return: pickle names with appropriate extensions
        """
        return list(
            map(
                lambda x: os.path.basename(x)[: -len(self.extension)],
                glob(os.path.join(self.name, "*%s" % self.extension)),
            )
        )

    def __getitem__(self, table_name: str) -> "Table":
        """
        Return table from the database.

        :param table_name: Name of the table
        :return: object of class PickleTable
        """
        return self.table(table_name)

    def table(self, table_name: str) -> "Table":
        """
        Select table.

        :param table_name: name of the table
        :return: object of class PickleTable
        """
        if table_name in self.tables:
            return Table(self, table_name)
        else:
            self.logger.warning(f"Table {table_name} not found in database {self.name}")
            return Table(self, table_name)


class Table(WithLogging, TableABC):
    """Class representing a Table in a Database."""

    def __init__(self, db: Database, table_name: str) -> None:
        """
        Implement a constructor for tables using pickle file format.

        :param db: database to which the table belongs
        :param table_name: name of the table
        """
        if not isinstance(db, Database):
            raise ValueError(
                f"The db should an instance of {'.'.join([Database.__module__, Database.__name__])}"
            )

        self.db = db
        self.name = table_name

    @property
    def filename(self) -> PathLike:
        """
        Return path to pickle.

        :return: path to pickle file
        """
        return os.path.join(self.db.name, "%s.p" % self.name)

    def to_df(self, query: Optional[str] = None) -> pd.DataFrame:
        """
        Read pickle.

        :return: pd.DataFrame or pd.Series read from pickle
        """
        df = pd.read_pickle(self.filename)
        return df if query is None else df.query(query)

    @property
    def data(self) -> pd.DataFrame:
        """
        Read pickle.

        :return: pd.DataFrame or pd.Series read from pickle
        """
        return pd.read_pickle(self.filename)

    def write(self, df: pd.DataFrame, overwrite: bool = False) -> None:
        """
        Write pickle of data, eventually outer joined with an input DataFrame.

        :param df: input data
        :param overwrite: whether or not to overwrite existing file
        :return: None
        """
        # self.data can fail with:
        #   - KeyError if it tries to read a non-pickle file
        #   - IOError if the file does not exist
        try:
            _in = [self.to_df()] if not overwrite else []
        except (KeyError, IOError):
            _in = []

        # pd.concat can fail with a TypeError if df is not an NDFrame object
        try:
            _df = pd.concat(_in + [df])
        except TypeError:
            _df = df

        _df.to_pickle(self.filename)
