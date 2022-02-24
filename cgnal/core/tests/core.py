"""Basic testing functionalities to be used in unittests."""

import pandas as pd
import numpy as np
from time import time
from typing import Callable, TypeVar
from unittest import TestCase as CoreTestCase
from cgnal.core.logging.defaults import WithLogging


class TestCase(CoreTestCase, WithLogging):
    """Tests class with basis functionalities."""

    def compareLists(self, first: list, second: list, strict: bool = False) -> None:
        """
        Compare lists.

        :param first: first list
        :param second: second list
        :param strict: whether to check element by element (strict) or only dimensions (non-strict)
        :return: None
        """
        if strict:
            for item1, item2 in zip(first, second):
                self.assertEqual(item1, item2)
        else:
            self.assertTrue(len(set(first).intersection(second)) == len(set(first)))

    def compareDicts(self, first: dict, second: dict, strict: bool = False) -> None:
        """
        Compare dictionaries.

        :param first: first dictionary
        :param second: second dictionary
        :param strict: whether to check element by element (strict) or only dimensions (non-strict)
        :return: None
        """
        for key in set(first.keys()).union(second.keys()):
            if isinstance(first[key], dict):
                self.compareDicts(first[key], second[key])
            elif isinstance(first[key], list):
                self.compareLists(first[key], second[key], strict)
            else:
                self.assertEqual(first[key], second[key])

    def compareDataFrames(
        self, first: pd.DataFrame, second: pd.DataFrame, msg: str
    ) -> None:
        """
        Compare dataframes.

        :param first: first dataframe
        :param second: second dataframe
        :param msg: message
        :return: None
        """
        try:
            pd.testing.assert_frame_equal(first, second)
        except AssertionError as e:
            raise self.failureException("Input series are different") from e

    def compareSeries(self, first: pd.Series, second: pd.Series, msg: str) -> None:
        """
        Compare series.

        :param first: first series
        :param second: second series
        :param msg: message
        :return: None
        """
        try:
            pd.testing.assert_series_equal(first, second)
        except AssertionError as e:
            raise self.failureException("Input series are different") from e

    def compareArrays(self, first: np.ndarray, second: np.ndarray, msg: str) -> None:
        """
        Compare arrays.

        :param first: first array
        :param second: second array
        :param msg: message
        :return: None
        """
        try:
            np.testing.assert_almost_equal(first, second, decimal=7)
        except AssertionError as e:
            raise self.failureException("Input arrays are different") from e

    def setUp(self) -> None:
        """
        Set up the class to add custom comparing/assertion functionalities.

        :return: None
        """
        self.addTypeEqualityFunc(pd.DataFrame, self.compareDataFrames)
        self.addTypeEqualityFunc(pd.Series, self.compareSeries)
        self.addTypeEqualityFunc(np.ndarray, self.compareArrays)


T = TypeVar("T", bound=WithLogging)


def logTest(test: Callable[[T], None]) -> Callable[[T], None]:
    """
    Return a function to be used to decorate every test for adding logs on name of the tests and timings.

    :param test: test method to wrap
    :return: wrapped test method
    """

    def wrap(obj: T) -> None:
        t0 = time()
        obj.logger.info(f"Executing Test {str(test.__name__)}")
        test(obj)
        obj.logger.info(f"Execution Time: {time() - t0} secs")

    return wrap
