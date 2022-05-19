import unittest
import os
import pandas as pd

from cgnal.core.data.model.core import (
    DillSerialization,
    IterableUtilsMixin,
    Range,
    CompositeRange,
    CachedIterable,
    LazyIterable,
    IterGenerator,
    RegisterLazyCachedIterables,
)
from cgnal.core.logging.defaults import getDefaultLogger
from cgnal.core.tests.core import logTest, TestCase
from tests import TMP_FOLDER
from typing import Iterator, Sequence

logger = getDefaultLogger()

n = 10


def generator() -> Iterator[int]:
    for i in range(n):
        yield i


class IntLazyIterable(
    LazyIterable[int], IterableUtilsMixin[int, "IntLazyIterable", "IntCachedIterable"]
):
    pass


@RegisterLazyCachedIterables(IntLazyIterable)
class IntCachedIterable(
    CachedIterable[int],
    IterableUtilsMixin[int, "IntLazyIterable", "IntCachedIterable"],
    DillSerialization,
):
    pass


lazy = IntLazyIterable(IterGenerator(generator))
cached = IntCachedIterable([i for i in range(n)])


class TestIterGenerator(TestCase):
    @logTest
    def test_iterator(self):
        def generator():
            for i in range(10):
                yield i

        self.assertTrue(isinstance(IterGenerator(generator).iterator, Iterator))
        self.assertEqual(next(IterGenerator(generator).iterator), 0)


class TestLazyIterable(TestCase):
    @logTest
    def test_map(self):
        plusOne = lazy.map(lambda x: x + 1)
        self.assertIsInstance(plusOne, LazyIterable)
        self.assertEqual([i for i in plusOne][0], 1)
        self.assertEqual(len(plusOne.to_cached()), n)

    @logTest
    def test__iter__(self):
        self.assertIsInstance(lazy.__iter__(), Iterator)
        self.assertEqual(list(lazy.__iter__()), list(range(n)))

    @logTest
    def test_take(self):
        self.assertIsInstance(lazy.take(1), CachedIterable)
        self.assertEqual(list(lazy.take(1))[0], 0)

    @logTest
    def test_filter(self):

        half = lazy.map(lambda x: x + 1).filter(lambda x: x % 2 == 0)
        self.assertEqual([i for i in half.to_cached()], [2, 4, 6, 8, 10])

    @logTest
    def test_toCached(self):
        cached = lazy.to_cached()
        self.assertIsInstance(cached, CachedIterable)
        self.assertEqual(list(cached), list(lazy))

    def test_toLazy(self):
        new_lazy = lazy.to_lazy()

        self.assertIsInstance(new_lazy, LazyIterable)
        self.assertEqual(list(new_lazy), list(lazy))

    @logTest
    def test_batch(self):

        batch_size = 2
        batches = lazy.map(lambda x: x + 1).batch(batch_size)

        for batch in batches:
            self.assertIsInstance(batch, CachedIterable)

        batches = lazy.map(lambda x: x + 1).batch(batch_size)
        self.assertEqual([i for i in next(batches)], [1, 2])

    @logTest
    def test_foreach(self):
        lst = []

        def f(x):
            lst.append(x + 2)

        lazy.foreach(f)
        self.assertEqual(lst, [i + 2 for i in lazy])

    def test_cached(self):
        self.assertTrue(not lazy.cached)

    def test_items(self):
        self.assertIsInstance(lazy.items, Iterator)
        self.assertEqual(list(lazy), list(range(n)))


class TestCachedIterables(TestCase):
    @staticmethod
    def functionWithSideEffect(lst):
        def function(x):
            lst.append(1)
            return x

        return function

    @logTest
    def test__len__(self):
        self.assertEqual(cached.__len__(), n)

    @logTest
    def test_map(self):

        plusOne = cached.map(lambda x: x + 1)

        self.assertIsInstance(plusOne, LazyIterable)
        self.assertEqual(len(plusOne.to_cached()), n)
        self.assertEqual(list(plusOne.to_cached())[0], 1)

    @logTest
    def test_take(self):
        self.assertTrue(isinstance(cached.take(1), CachedIterable))
        self.assertEqual(list(cached.take(1))[0], 0)

    @logTest
    def test_filter(self):

        half = cached.map(lambda x: x + 1).filter(lambda x: x % 2 == 0)
        self.assertEqual([i for i in half.to_cached()], [2, 4, 6, 8, 10])

    @logTest
    def test_batch(self):

        batch_size = 2
        batches = cached.map(lambda x: x + 1).batch(batch_size)

        for batch in batches:
            self.assertIsInstance(batch, CachedIterable)

        batches = cached.map(lambda x: x + 1).batch(batch_size)
        self.assertEqual([i for i in next(batches)], [1, 2])

    @logTest
    def test_toLazy(self):

        lazy = cached.to_lazy()
        self.assertIsInstance(lazy, LazyIterable)
        self.assertEqual(list(lazy), list(cached))

    @logTest
    def test_toCached(self):
        new_cached = cached.to_cached()
        self.assertIsInstance(new_cached, CachedIterable)
        self.assertEqual(new_cached.items, cached.items)

    @logTest
    def test_cached(self):
        self.assertTrue(cached.cached)

    @logTest
    def test_items(self):
        self.assertIsInstance(cached.items, Sequence)
        self.assertEqual(cached.items, list(range(10)))

    @logTest
    def test__iter__(self):
        self.assertIsInstance(cached.__iter__(), Iterator)
        self.assertEqual(list(cached.__iter__()), list(range(n)))

    def test_write_load(self):
        filename = os.path.join(TMP_FOLDER, "my_cached.p")
        cached.write(filename)
        new_cached = cached.load(filename)
        self.assertEqual(new_cached.items, cached.items)

    @logTest
    def test_foreach(self):
        lst = []

        def f(x):
            lst.append(x + 2)

        cached.foreach(f)
        self.assertEqual(lst, [i + 2 for i in cached])


class TestRange(TestCase):

    firstRange = Range("2021-01-01", "2021-01-10")
    secondRange = Range("2021-01-08", "2021-01-15")
    thirdRange = Range("2021-01-13", "2021-01-15")

    @logTest
    def test_start_end(self):

        self.assertEqual(self.firstRange.start, pd.to_datetime("2021-01-01"))
        self.assertEqual(self.firstRange.end, pd.to_datetime("2021-01-10"))

    @logTest
    def test_range(self):

        self.assertEqual(len(self.firstRange.range("D")), 10)
        self.assertEqual(len(self.firstRange.range("H")), 9 * 24 + 1)
        self.assertEqual(len(self.firstRange.range("B")), 6)

    @logTest
    def test_overlaps_range(self):
        self.assertTrue(self.firstRange._overlaps_range(self.secondRange))
        self.assertFalse(self.firstRange._overlaps_range(self.thirdRange))

    @logTest
    def test_overlaps(self):
        self.assertTrue(self.firstRange.overlaps(self.secondRange))
        self.assertFalse(self.firstRange.overlaps(self.thirdRange))

    def test__add__(self):
        self.assertTrue(isinstance(self.firstRange.__add__(self.secondRange), Range))
        self.assertTrue(
            isinstance(self.firstRange.__add__(self.thirdRange), CompositeRange)
        )
        self.assertEqual(
            self.firstRange.__add__(self.secondRange).start,
            Range("2021-01-01", "2021-01-15").start,
        )
        self.assertEqual(
            self.firstRange.__add__(self.secondRange).end,
            Range("2021-01-01", "2021-01-15").end,
        )
        self.assertEqual(
            self.firstRange.__add__(self.thirdRange).start,
            CompositeRange([self.firstRange, self.thirdRange]).start,
        )
        self.assertEqual(
            self.firstRange.__add__(self.thirdRange).end,
            CompositeRange([self.firstRange, self.thirdRange]).end,
        )

    def test__iter__(self):
        self.assertTrue(isinstance(self.firstRange.__iter__(), Iterator))
        self.assertEqual(next(self.firstRange.__iter__()).start, self.firstRange.start)
        self.assertEqual(next(self.firstRange.__iter__()).end, self.firstRange.end)

    def test_days(self):
        self.assertTrue(isinstance(self.firstRange.days, list))
        self.assertTrue(isinstance(self.firstRange.days[0], pd.Timestamp))
        self.assertEqual(len(self.firstRange.days), 10)

    def test_business_days(self):
        self.assertTrue(isinstance(self.firstRange.business_days, list))
        self.assertTrue(isinstance(self.firstRange.business_days[0], pd.Timestamp))
        self.assertEqual(len(self.firstRange.business_days), 6)

    def test_minutes_15(self):
        self.assertTrue(isinstance(self.firstRange.minutes_15, list))
        self.assertTrue(isinstance(self.firstRange.minutes_15[0], pd.Timestamp))
        self.assertEqual(len(self.firstRange.minutes_15), 865)

    @logTest
    def test__str__(self):
        self.assertEqual(
            self.firstRange.__str__(), "2021-01-01 00:00:00-2021-01-10 00:00:00"
        )


class TestCompositeRange(TestCase):

    firstRange = Range("2021-01-01", "2021-01-10")
    secondRange = Range("2021-01-08", "2021-01-15")
    thirdRange = Range("2021-01-14", "2021-01-20")
    fourthRange = Range("2021-01-16", "2021-01-18")
    fifthRange = Range("2021-01-22", "2021-01-25")
    compositeRange = CompositeRange([firstRange, thirdRange])

    @logTest
    def test_start_end(self):
        self.assertEqual(self.compositeRange.start, pd.to_datetime("2021-01-01"))
        self.assertEqual(self.compositeRange.end, pd.to_datetime("2021-01-20"))

    @logTest
    def test__iter__(self):
        self.assertTrue(isinstance(self.compositeRange.__iter__(), Iterator))
        self.assertEqual(
            [i for i in self.compositeRange.__iter__()][0].range(),
            self.firstRange.range(),
        )
        self.assertEqual(
            [i for i in self.compositeRange.__iter__()][1].range(),
            self.thirdRange.range(),
        )

    @logTest
    def test__add__(self):
        self.assertTrue(
            isinstance(self.compositeRange.__add__(self.firstRange), CompositeRange)
        )
        self.assertEqual(
            CompositeRange(
                [Range("2021-01-01", "2021-01-15"), Range("2021-01-22", "2021-01-25")]
            ).range(),
            CompositeRange([self.firstRange, self.secondRange])
            .__add__(self.fifthRange)
            .range(),
        )
        self.assertEqual(
            self.compositeRange.__add__(self.secondRange).range(),
            Range("2021-01-01", "2021-01-20").range(),
        )

    @logTest
    def test_range(self):
        self.assertTrue(isinstance(self.compositeRange.range(), list))
        pd_daterange = list(
            pd.date_range(self.firstRange.start, self.firstRange.end, freq="D")
        ) + list(pd.date_range(self.thirdRange.start, self.thirdRange.end, freq="D"))
        self.assertEqual(self.compositeRange.range("D"), pd_daterange)

    @logTest
    def test_overlaps(self):

        compositeRange = CompositeRange([self.firstRange, self.thirdRange])

        self.assertTrue(compositeRange.overlaps(self.secondRange))

        compositeRange = CompositeRange([self.firstRange, self.fourthRange])

        self.assertFalse(compositeRange.overlaps(self.fifthRange))

    @logTest
    def test_sum(self):

        # This should result in a simple Range, since the two ranges are not disjoint
        self.assertTrue(isinstance(self.firstRange + self.secondRange, Range))

        # This should result in a CompositeRange, since the two ranges are disjoint
        self.assertTrue(isinstance(self.firstRange + self.thirdRange, CompositeRange))

        # This should result in a simple Range, since the three ranges are not disjoint when taken all together
        self.assertTrue(
            isinstance(self.firstRange + self.secondRange + self.thirdRange, Range)
        )

    @logTest
    def test_simplify(self):

        composite = CompositeRange([self.firstRange, self.secondRange, self.thirdRange])

        simplified = composite.simplify()

        self.assertTrue(isinstance(simplified, Range))
        self.assertEqual(set(composite.range("D")), set(simplified.range("D")))

    @logTest
    def test_days(self):
        self.assertTrue(isinstance(self.compositeRange.days, list))
        self.assertTrue(isinstance(self.compositeRange.days[0], pd.Timestamp))
        self.assertEqual(len(self.compositeRange.days), 17)

    @logTest
    def test_business_days(self):
        self.assertTrue(isinstance(self.compositeRange.business_days, list))
        self.assertTrue(isinstance(self.compositeRange.business_days[0], pd.Timestamp))
        self.assertEqual(len(self.compositeRange.business_days), 11)

    @logTest
    def test_minutes_15(self):
        self.assertTrue(isinstance(self.compositeRange.minutes_15, list))
        self.assertTrue(isinstance(self.compositeRange.minutes_15[0], pd.Timestamp))
        self.assertEqual(len(self.compositeRange.minutes_15), 1442)

    @logTest
    def test__str__(self):
        self.assertEqual(
            self.compositeRange.__str__(),
            "2021-01-01 00:00:00-2021-01-10 00:00:00 // 2021-01-14 00:00:00-2021-01-20 00:00:00",
        )


if __name__ == "__main__":
    unittest.main()
