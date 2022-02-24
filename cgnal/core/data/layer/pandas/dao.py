"""Module with the implementation and abstraction for serializing/deserializing objects into DataFrames."""

import json
from typing import Hashable, Optional, Dict, Union, Sequence

import pandas as pd

from cgnal.core.data.layer import DAO
from cgnal.core.data.model.text import Document
from cgnal.core.utils.decorators import lazyproperty as lazy


class DocumentDAO(DAO[Document, pd.Series]):
    """Data access object for documents."""

    def computeKey(self, doc: Document) -> Hashable:
        """
        Get document id.

        :param doc: an instance of :class:`cgnal.data.model.text.Document`
        :return: uuid i.e. id of the given document
        """
        return doc.uuid

    def get(self, doc: Document) -> pd.Series:
        """
        Get doc as pd.Series with uuid as name.

        :param doc: an instance of :class:`cgnal.data.model.text.Document`
        :return: pd.Series
        """
        return pd.Series(doc.data, name=self.computeKey(doc))

    def parse(self, row: pd.Series) -> Document:
        """
        Get a row i.e. pd.Series as a Document.

        :param row: pd.Series, row of a pd.DataFrame
        :return: :class:`cgnal.data.model.text.Document`, a Document object
        """
        from typing import Dict

        data: Dict = row.to_dict()
        return Document(row.name, data)


class DataFrameDAO(DAO[pd.DataFrame, pd.Series]):
    """Data Access Object for pd.DataFrames."""

    def computeKey(self, df: pd.DataFrame) -> Hashable:
        """
        Get dataframe name.

        :param df: pd.DataFrame. A pandas dataframe
        :return: str, name of the dataframe
        """
        try:
            return df.name
        except AttributeError:
            return hash(json.dumps({str(k): str(v) for k, v in df.to_dict().items()}))

    def get(self, df: pd.DataFrame) -> pd.Series:
        """
        Get dataframe as pd.Series.

        :param df: pd.DataFrame. A pandas dataframe
        :return: pd.Series
        """
        serie = pd.concat({k: df[k] for k in df})
        serie.name = self.computeKey(df)
        return serie

    def parse(self, row: pd.Series) -> pd.DataFrame:
        """
        Get a row i.e. pd.Series as a pandas DataFrame.

        :param row: pd.Series, row of a pd.DataFrame
        :return: pd.DataFrame, a pandas dataframe object
        """
        if isinstance(row.index, pd.MultiIndex):
            return pd.concat({c: row[c] for c in row.index.levels[0]}, axis=1)
        else:
            return row.to_frame()

    @staticmethod
    def addName(df: pd.DataFrame, name: Optional[Hashable]) -> pd.DataFrame:
        """
        Add name to the input dataframe.

        :param df: pd.DataFrame
        :param name: str
        :return: pd.DataFrame
        """
        df.name = name
        return df


class SeriesDAO(DAO[pd.Series, pd.Series]):
    """Data Access Object for pd.Series."""

    def __init__(
        self,
        mapping: Optional[Dict[Hashable, Hashable]] = None,
        keys: Optional[Sequence[Hashable]] = None,
    ):
        """
        Create a Data Access Object for pd.Series.

        :param mapping: mapping of names between the file and the Series. The value of the key represent
            the name in the file, the value represent the name we want to have in the Series
        :param keys: which of the fields in the Series should be used as keys
        """
        self.mapping = mapping if mapping is not None else {}
        self.keys = keys

    @lazy
    def inverseMapping(self) -> Dict[Hashable, Hashable]:
        """
        Return the inverse mapping between field names.

        :return: dict with inverse mapping
        """
        if len(set(self.mapping.values())) != len(self.mapping):
            raise ValueError("Mapping is not invertible as there are duplicated values")
        return {v: k for k, v in self.mapping.items()}

    def computeKey(self, serie) -> Union[Hashable, Dict]:
        """
        Get series name.

        :param serie: pd.Series
        :return: dict representing the key
        """
        if serie.name is None:
            raise ValueError("Cannot persist Series without name (to be used as index)")

        if self.keys is None:
            return serie.name

        if isinstance(serie.name, tuple):
            if len(self.keys) != len(serie.name):
                raise ValueError("Keys and Series name have different dimensions")
            return {key: value for key, value in zip(self.keys, serie.name)}
        else:
            if len(self.keys) > 1:
                raise ValueError("Keys and Series name have different dimensions")
            else:
                return {self.keys[0]: serie.name}

    def get(self, serie: pd.Series) -> pd.Series:
        """
        Get a series as series object.

        :param s: pd.Series
        :return: pd.Series
        """
        row = serie.rename(self.inverseMapping)
        return (
            row.append(pd.Series(self.computeKey(serie)))
            if self.keys is not None
            else row
        )

    def parse(self, row: pd.Series) -> pd.Series:
        """
        Get a row as a pd.Series object.

        :param row: pd.Series
        :return: pd.Series
        """
        serie = row.rename(self.mapping)
        if self.keys is not None:
            return serie.rename(tuple(serie[self.keys])).drop(self.keys)
        else:
            return serie
