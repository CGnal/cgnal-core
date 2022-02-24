"""Basic utilities extending pandas functionalities."""

from typing import Union, List, Any
from pandas import DataFrame, Index
from pandas.core.arrays.sparse.dtype import SparseDtype


def is_sparse(df: DataFrame) -> bool:
    """
    Return `True` if a dataframe is sparse.

    :param df: pandas dataframe
    :return: bool
    """
    return all([isinstance(v, SparseDtype) for k, v in df.dtypes.items()])


def loc(df: DataFrame, idx: Union[Index, List[Any]]):
    """
    Return a filtered dataframe based on an index list and it is designed to work propertly with sparse dataframe as well.

    :param df: DataFrame
    :param idx: index list
    :return: DataFrame which is filtered
    """
    if is_sparse(df):
        csr = df.sparse.to_coo().tocsr()
        pos = [pos for pos, elem in enumerate(df.index) if elem in idx]
        return DataFrame.sparse.from_spmatrix(
            csr[pos, :],
            index=idx[pos]
            if isinstance(idx, Index)
            else [elem for elem in df.index if elem in idx],
            columns=df.columns,
        )
    else:
        return df.loc[idx]
