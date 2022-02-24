import os
import unittest
from typeguard import typechecked
from typing import List, Dict

import pandas as pd
from cgnal.core.logging.defaults import getDefaultLogger
from cgnal.core.tests.core import logTest, TestCase
from cgnal.core.utils.decorators import lazyproperty as lazy, param_check
from cgnal.core.utils.dict import (
    groupIterable,
    pairwise,
    union,
    flattenKeys,
    unflattenKeys,
    filterNones,
    groupBy,
)
from cgnal.core.utils.fs import (
    mkdir,
    create_dir_if_not_exists,
    get_lexicographic_dirname,
)
from cgnal.core.utils.pandas import is_sparse, loc

from tests import TMP_FOLDER

logger = getDefaultLogger()


class TestUtilsDict(TestCase):
    @logTest
    def test_groupIterable(self):
        self.assertEqual(
            [
                el
                for el in groupIterable(
                    {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}, batch_size=3
                )
            ],
            [["a", "b", "c"], ["d", "e", "f"]],
        )
        self.assertEqual(
            [
                el
                for el in groupIterable(
                    {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}, batch_size=3
                )
            ],
            [["a", "b", "c"], ["d", "e"]],
        )
        self.assertEqual(
            [
                el
                for el in groupIterable(
                    {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7},
                    batch_size=3,
                )
            ],
            [["a", "b", "c"], ["d", "e", "f"], ["g"]],
        )

    @logTest
    def test_pairwise(self):
        self.assertEqual(
            [el for el in pairwise({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6})],
            [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")],
        )
        self.assertEqual([el for el in pairwise({"a": 1})], [])

    @logTest
    def test_union(self):
        self.assertEqual(
            union({"1": {"a": 1}}, filterNones({"1": {"a": None}, "b": 1})),
            {"1": {"a": 1}, "b": 1},
        )

        self.assertEqual(
            union({"1": {"a": 1}}, filterNones({"1": {"a": 2}, "b": None})),
            {"1": {"a": 2}},
        )

        self.assertEqual(
            union({"1": None}, {"1": 1, "2": 3}, {"1": {"1a": 1, "1b": 2}, "3": 4}),
            {"1": {"1a": 1, "1b": 2}, "2": 3, "3": 4},
        )

    @logTest
    def test_flattenKeys(self):
        self.assertEqual(
            flattenKeys({"a": {"b": {"c": 2}}, "d": 2, "e": 3}, sep="."),
            {"a.b.c": 2, "d": 2, "e": 3},
        )

        self.assertEqual(
            flattenKeys({"a": {"b": {"c": 2}}, "a": 2, "e": 3}), {"a": 2, "e": 3}
        )

    @logTest
    def test_unflattenKeys(self):
        self.assertEqual(
            unflattenKeys({"a.b.c": 2, "d": 2, "e": 3}, sep="."),
            {"a": {"b": {"c": 2}}, "d": 2, "e": 3},
        )

        self.assertEqual(
            unflattenKeys({"a.b.c": 2, "d": 2, "e": 3}, sep="_"),
            {"a.b.c": 2, "d": 2, "e": 3},
        )

    @logTest
    def test_filterNones(self):
        self.assertEqual(filterNones({"a": 1, "b": None}), {"a": 1})

    @logTest
    def test_groupBy(self):
        self.assertEqual(
            [(k, v) for k, v in groupBy(["abc", "ab", "bcd", "c"], key=len)],
            [(1, ["c"]), (2, ["ab"]), (3, ["abc", "bcd"])],
        )


class TestUtilsFs(TestCase):
    @logTest
    def test_mkdir(self):
        directory = os.path.join("/tmp", "test_utils_fs")
        mkdir(directory)

        self.assertTrue(os.path.exists(directory))
        os.rmdir(directory)

    @logTest
    def test_create_dir_if_not_exists(self):
        directory = os.path.join("/tmp", "test_utils_fs")
        create_dir_if_not_exists(directory)

        self.assertTrue(os.path.exists(directory))
        os.rmdir(directory)

    @logTest
    def test_get_lexicographic_dirname(self):
        create_dir_if_not_exists(os.path.join("/tmp", "zzz"))

        self.assertEqual(get_lexicographic_dirname("/tmp", first=False), "zzz")
        os.rmdir(os.path.join("/tmp", "zzz"))


class TestPandas(TestCase):
    @logTest
    def test_is_sparse(self):
        self.assertTrue(
            is_sparse(
                pd.DataFrame(
                    {
                        "v1": pd.arrays.SparseArray([0, 0, 0, 0, 1]),
                        "v2": pd.arrays.SparseArray([1, 0, 0, 0, 1]),
                        "v3": pd.arrays.SparseArray([1, 0, 0, 0, 0]),
                    }
                )
            )
        )

        self.assertFalse(
            is_sparse(
                pd.DataFrame(
                    {
                        "v1": [0, 0, 0, 0, 1],
                        "v2": pd.arrays.SparseArray([1, 0, 0, 0, 1]),
                        "v3": pd.arrays.SparseArray([1, 0, 0, 0, 0]),
                    }
                )
            )
        )

    @logTest
    def test_loc(self):
        res = pd.DataFrame({"v1": [0], "v2": [1], "v3": [1]})
        self.assertTrue(
            (
                loc(
                    pd.DataFrame(
                        {
                            "v1": [0, 0, 0, 0, 1],
                            "v2": pd.arrays.SparseArray([1, 0, 0, 0, 1]),
                            "v3": pd.arrays.SparseArray([1, 0, 0, 0, 0]),
                        }
                    ),
                    [0],
                )
                == res
            )
            .all()
            .all()
        )

        # TODO: manca scipy nei requirements ?
        self.assertTrue(
            (
                loc(
                    pd.DataFrame(
                        {
                            "v1": pd.arrays.SparseArray([0, 0, 0, 0, 1]),
                            "v2": pd.arrays.SparseArray([1, 0, 0, 0, 1]),
                            "v3": pd.arrays.SparseArray([1, 0, 0, 0, 0]),
                        }
                    ),
                    [0],
                )
                == res
            )
            .all()
            .all()
        )
        self.assertTrue(
            is_sparse(
                loc(
                    pd.DataFrame(
                        {
                            "v1": pd.arrays.SparseArray([0, 0, 0, 0, 1]),
                            "v2": pd.arrays.SparseArray([1, 0, 0, 0, 1]),
                            "v3": pd.arrays.SparseArray([1, 0, 0, 0, 0]),
                        }
                    ),
                    [0],
                )
            )
        )


@typechecked
class MyTestClass:
    def __init__(self, param: str = "test"):
        self.param = param

    @lazy
    def list_param(self) -> List:
        return [1, 2, 3]

    def dict_constructor(self, k_vals: List[str], v_vals: List[List[int]]) -> Dict:
        return {k: v for k, v in zip(k_vals, v_vals)}


class MyClass:
    def __init__(self, param: str = "test"):
        self.param = param

    @lazy
    def list_param(self) -> List:
        return [1, 2, 3]

    # TODO: param_check decorator breakes when specification of types contained within collections is present.
    #  e.g. dict_constructor(self, k_vals: List[str], v_vals: List[List[int]])
    #  generates "...TypeError: Parameterized generics cannot be used with class or instance checks"
    @param_check(with_none=False)
    def dict_constructor(self, k_vals: List, v_vals: List) -> Dict:
        return {k: v for k, v in zip(k_vals, v_vals)}


class TestDecorators(TestCase):
    @logTest
    def test_lazyproperty(self):
        ex = MyClass()

        self.assertEqual(ex.__dict__, {"param": "test"})

        info = f"Testing lazyproperty decorator. Let's try to call the list_param={ex.list_param} attribute."
        logger.info(info)

        self.assertEqual(ex.__dict__, {"param": "test", "list_param": [1, 2, 3]})

    @logTest
    def test_param_check(self):
        ex = MyClass()

        self.assertEqual(
            ex.dict_constructor(
                k_vals=["a", "b", "c"], v_vals=[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            ),
            {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
        )
        self.assertRaises(TypeError, ex.dict_constructor, k_vals="a", v_vals=[1, 2, 3])
        self.assertRaises(
            ValueError, ex.dict_constructor, k_vals=None, v_vals=[1, 2, 3]
        )

    @logTest
    def test_param_check_with_typeguard(self):
        ex = MyTestClass()

        self.assertEqual(
            ex.dict_constructor(
                k_vals=["a", "b", "c"], v_vals=[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            ),
            {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
        )
        self.assertRaises(TypeError, ex.dict_constructor, k_vals="a", v_vals=[1, 2, 3])
        self.assertRaises(TypeError, ex.dict_constructor, k_vals=None, v_vals=[1, 2, 3])

    def test_cache_io(self):
        from cgnal.core.utils.decorators import Cached

        # from time import sleep

        class A(Cached):
            def __init__(self, cnt):
                self.cnt = cnt

            @Cached.cache
            def my_long_computation(self):
                # _ = sleep(1)
                self.cnt = self.cnt + 1
                return self.cnt

        a = A(0)

        # This should compute the value
        self.assertEqual(a.my_long_computation, 1)
        # This should get the retrieve data
        self.assertEqual(a.my_long_computation, 1)

        filename = os.path.join(TMP_FOLDER, "save_pickles_test")

        a.save_pickles(filename)

        b = A(1)

        b.load(filename)
        # This should get the retrieve data
        self.assertEqual(b.my_long_computation, 1)
        b.clear_cache()
        # This should compute the value
        self.assertEqual(b.my_long_computation, 2)


class TestDocumentArchivers(TestCase):

    url = "http://192.168.2.110:8686"

    test_file = "tests/test.txt"

    # @logTest
    # def test_base_function(self):
    #     sync = CloudSync(self.url, TMP_FOLDER)
    #
    #     namefile = sync.get_if_not_exists(self.test_file)
    #
    #     self.assertTrue(os.path.exists( namefile ))
    #
    #     os.remove( namefile )
    #
    #
    # @logTest
    # def test_decorator(self):
    #
    #     sync = CloudSync(self.url, TMP_FOLDER)
    #
    #     @sync.get_if_not_exists_decorator
    #     def decorated_function(filename):
    #         return filename
    #
    #     namefile = decorated_function(self.test_file)
    #
    #     self.assertTrue(os.path.exists( namefile ))
    #
    #     os.remove( namefile )
    #
    # @logTest
    # def test_multiple(self):
    #     sync = CloudSync(self.url, TMP_FOLDER)
    #
    #     namefile = sync.get_if_not_exists(self.test_file)
    #
    #     self.assertTrue(os.path.exists(namefile))
    #
    #     sleep(3)
    #
    #     time = os.path.getmtime(namefile)
    #
    #     namefile = sync.get_if_not_exists(self.test_file)
    #
    #     time2 = os.path.getmtime(namefile)
    #
    #     self.assertTrue(time==time2)
    #
    #     os.remove(namefile)
    #
    #
    # @logTest
    # def test_upload(self):
    #
    #     sync = CloudSync(self.url, TMP_FOLDER)
    #
    #     namefile = sync.get_if_not_exists(self.test_file)
    #
    #     upload = f"{self.test_file}.upload"
    #
    #     os.rename(namefile, sync.pathTo(upload))
    #
    #     sync.upload(upload)
    #
    #     os.remove(sync.pathTo(upload))
    #
    #     namefile_new = sync.get_if_not_exists(upload)
    #
    #     self.assertTrue(os.path.exists(namefile_new))


if __name__ == "__main__":
    unittest.main()
