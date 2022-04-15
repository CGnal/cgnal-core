import os
import unittest

import numpy as np
import pandas as pd

from cgnal.core.data.model.ml import (
    LazyDataset,
    IterGenerator,
    MultiFeatureSample,
    Sample,
    PandasDataset,
    PandasTimeIndexedDataset,
    CachedDataset,
    features_and_labels_to_dataset,
)
from typing import Iterator, Generator
from cgnal.core.tests.core import TestCase, logTest
from tests import TMP_FOLDER

samples = [
    Sample(features=[100, 101], label=1),
    Sample(features=[102, 103], label=2),
    Sample(features=[104, 105], label=3),
    Sample(features=[106, 107], label=4),
    Sample(features=[108, 109], label=5),
    Sample(features=[110, 111], label=6),
    Sample(features=[112, 113], label=7),
    Sample(features=[114, 115], label=8),
    Sample(features=[116, 117], label=9),
]


def samples_gen():
    for sample in samples:
        if not any([np.isnan(x).any() for x in sample.features]):
            yield sample


lazyDat = LazyDataset(IterGenerator(samples_gen))


class features_and_labels_to_datasetTests(TestCase):
    def test_features_and_labels_to_dataset(self):

        dataset = features_and_labels_to_dataset(
            pd.concat(
                [
                    pd.Series([1, 0, 2, 3], name="feat1"),
                    pd.Series([1, 2, 3, 4], name="feat2"),
                ],
                axis=1,
            ),
            pd.Series([0, 0, 0, 1], name="Label"),
        )

        dataset_no_labels = features_and_labels_to_dataset(
            pd.concat(
                [
                    pd.Series([1, 0, 2, 3], name="feat1"),
                    pd.Series([1, 2, 3, 4], name="feat2"),
                ],
                axis=1,
            ),
            None,
        )

        self.assertTrue(isinstance(dataset_no_labels, CachedDataset))
        self.assertTrue(isinstance(dataset, CachedDataset))
        self.assertTrue(
            (
                dataset.getFeaturesAs("pandas")
                == pd.concat(
                    [
                        pd.Series([1, 0, 2, 3], name="feat1"),
                        pd.Series([1, 2, 3, 4], name="feat2"),
                    ],
                    axis=1,
                )
            )
            .all()
            .all()
        )
        self.assertTrue(
            (
                dataset.getLabelsAs("pandas")
                == pd.DataFrame(pd.Series([0, 0, 0, 1], name="Label"))
            )
            .all()
            .all()
        )


class LazyDatasetTests(TestCase):
    @logTest
    def test_withLookback_MultiFeatureSample(self):
        samples = [
            MultiFeatureSample(
                features=[np.array([100.0, 101.0]), np.array([np.NaN])], label=1.0
            ),
            MultiFeatureSample(
                features=[np.array([102.0, 103.0]), np.array([1.0])], label=2.0
            ),
            MultiFeatureSample(
                features=[np.array([104.0, 105.0]), np.array([2.0])], label=3.0
            ),
            MultiFeatureSample(
                features=[np.array([106.0, 107.0]), np.array([3.0])], label=4.0
            ),
            MultiFeatureSample(
                features=[np.array([108.0, 109.0]), np.array([4.0])], label=5.0
            ),
            MultiFeatureSample(
                features=[np.array([110.0, 111.0]), np.array([5.0])], label=6.0
            ),
            MultiFeatureSample(
                features=[np.array([112.0, 113.0]), np.array([6.0])], label=7.0
            ),
            MultiFeatureSample(
                features=[np.array([114.0, 115.0]), np.array([7.0])], label=8.0
            ),
            MultiFeatureSample(
                features=[np.array([116.0, 117.0]), np.array([8.0])], label=9.0
            ),
        ]

        def samples_gen():
            for sample in samples:
                if not any([np.isnan(x).any() for x in sample.features]):
                    yield sample

        X1 = np.array(
            [
                [[102.0, 103.0], [104.0, 105.0], [106.0, 107.0]],
                [[104.0, 105.0], [106.0, 107.0], [108.0, 109.0]],
                [[106.0, 107.0], [108.0, 109.0], [110.0, 111.0]],
                [[108.0, 109.0], [110.0, 111.0], [112.0, 113.0]],
            ]
        )
        y1 = np.array(
            [
                [[1.0], [2.0], [3.0]],
                [[2.0], [3.0], [4.0]],
                [[3.0], [4.0], [5.0]],
                [[4.0], [5.0], [6.0]],
            ]
        )
        lab1 = np.array([4.0, 5.0, 6.0, 7.0])
        X2 = np.array(
            [
                [[110.0, 111.0], [112.0, 113.0], [114.0, 115.0]],
                [[112.0, 113.0], [114.0, 115.0], [116.0, 117.0]],
            ]
        )
        y2 = np.array([[[5.0], [6.0], [7.0]], [[6.0], [7.0], [8.0]]])
        lab2 = np.array([8.0, 9.0])

        lookback = 3
        batch_size = 4

        lazyDat = LazyDataset(IterGenerator(samples_gen))
        lookbackDat = lazyDat.withLookback(lookback)
        batch_gen = lookbackDat.batch(batch_size)

        batch1: CachedDataset = next(batch_gen)
        batch2: CachedDataset = next(batch_gen)

        tmp1 = batch1.getFeaturesAs("array")
        temp1X = np.array(list(map(lambda x: np.stack(x), tmp1[:, :, 0])))
        temp1y = np.array(list(map(lambda x: np.stack(x), tmp1[:, :, 1])))
        tmp1lab = batch1.getLabelsAs("array")

        res = [
            np.array_equal(temp1X, X1),
            np.array_equal(temp1y, y1),
            np.array_equal(tmp1lab, lab1),
        ]

        tmp2 = batch2.getFeaturesAs("array")
        temp2X = np.array(list(map(lambda x: np.stack(x), tmp2[:, :, 0])))
        temp2y = np.array(list(map(lambda x: np.stack(x), tmp2[:, :, 1])))
        tmp2lab = batch2.getLabelsAs("array")

        res = res + [
            np.array_equal(temp2X, X2),
            np.array_equal(temp2y, y2),
            np.array_equal(tmp2lab, lab2),
        ]

        self.assertTrue(all(res))

    @logTest
    def test_withLookback_ArrayFeatureSample(self):

        samples = [
            Sample(features=np.array([100, 101]), label=1),
            Sample(features=np.array([102, 103]), label=2),
            Sample(features=np.array([104, 105]), label=3),
            Sample(features=np.array([106, 107]), label=4),
            Sample(features=np.array([108, 109]), label=5),
            Sample(features=np.array([110, 111]), label=6),
            Sample(features=np.array([112, 113]), label=7),
            Sample(features=np.array([114, 115]), label=8),
            Sample(features=np.array([116, 117]), label=9),
        ]

        def samples_gen():
            for sample in samples:
                if not any([np.isnan(x).any() for x in sample.features]):
                    yield sample

        X1 = np.array(
            [
                [[100, 101], [102, 103], [104, 105]],
                [[102, 103], [104, 105], [106, 107]],
                [[104, 105], [106, 107], [108, 109]],
                [[106, 107], [108, 109], [110, 111]],
            ]
        )
        lab1 = np.array([3, 4, 5, 6])
        X2 = np.array(
            [
                [[108, 109], [110, 111], [112, 113]],
                [[110, 111], [112, 113], [114, 115]],
                [[112, 113], [114, 115], [116, 117]],
            ]
        )
        lab2 = np.array([7, 8, 9])

        lookback = 3
        batch_size = 4

        lazyDat = LazyDataset(IterGenerator(samples_gen))
        lookbackDat = lazyDat.withLookback(lookback)
        batch_gen = lookbackDat.batch(batch_size)

        batch1: CachedDataset = next(batch_gen)
        batch2: CachedDataset = next(batch_gen)

        tmp1 = batch1.getFeaturesAs("array")
        tmp1lab = batch1.getLabelsAs("array")

        res = [np.array_equal(tmp1, X1), np.array_equal(tmp1lab, lab1)]

        tmp2 = batch2.getFeaturesAs("array")
        tmp2lab = batch2.getLabelsAs("array")

        res = res + [np.array_equal(tmp2, X2), np.array_equal(tmp2lab, lab2)]

        self.assertTrue(all(res))

    @logTest
    def test_withLookback_ListFeatureSample(self):

        samples = [
            Sample(features=[100, 101], label=1),
            Sample(features=[102, 103], label=2),
            Sample(features=[104, 105], label=3),
            Sample(features=[106, 107], label=4),
            Sample(features=[108, 109], label=5),
            Sample(features=[110, 111], label=6),
            Sample(features=[112, 113], label=7),
            Sample(features=[114, 115], label=8),
            Sample(features=[116, 117], label=9),
        ]

        def samples_gen():
            for sample in samples:
                if not any([np.isnan(x).any() for x in sample.features]):
                    yield sample

        X1 = np.array(
            [
                [[100, 101], [102, 103], [104, 105]],
                [[102, 103], [104, 105], [106, 107]],
                [[104, 105], [106, 107], [108, 109]],
                [[106, 107], [108, 109], [110, 111]],
            ]
        )
        lab1 = np.array([3, 4, 5, 6])
        X2 = np.array(
            [
                [[108, 109], [110, 111], [112, 113]],
                [[110, 111], [112, 113], [114, 115]],
                [[112, 113], [114, 115], [116, 117]],
            ]
        )
        lab2 = np.array([7, 8, 9])

        lookback = 3
        batch_size = 4

        lazyDat = LazyDataset(IterGenerator(samples_gen))
        lookbackDat = lazyDat.withLookback(lookback)
        batch_gen = lookbackDat.batch(batch_size)

        batch1: CachedDataset = next(batch_gen)
        batch2: CachedDataset = next(batch_gen)

        tmp1 = batch1.getFeaturesAs("array")
        tmp1lab = batch1.getLabelsAs("array")

        res = [np.array_equal(tmp1, X1), np.array_equal(tmp1lab, lab1)]

        tmp2 = batch2.getFeaturesAs("array")
        tmp2lab = batch2.getLabelsAs("array")

        res = res + [np.array_equal(tmp2, X2), np.array_equal(tmp2lab, lab2)]

        self.assertTrue(all(res))

    @logTest
    def test_features_labels(self):

        self.assertTrue(isinstance(lazyDat.features(), Generator))
        self.assertTrue(isinstance(lazyDat.labels(), Generator))
        self.assertTrue(isinstance(lazyDat.getFeaturesAs(), Generator))
        self.assertTrue(isinstance(lazyDat.getLabelsAs(), Generator))
        self.assertEqual(next(lazyDat.getFeaturesAs()), samples[0].features)
        self.assertEqual(next(lazyDat.getLabelsAs()), samples[0].label)
        self.assertEqual(next(lazyDat.features()), samples[0].features)
        self.assertEqual(next(lazyDat.labels()), samples[0].label)


class CachedDatasetTests(TestCase):
    @logTest
    def test_to_df(self):

        self.assertTrue(isinstance(CachedDataset(lazyDat).to_df(), pd.DataFrame))
        self.assertTrue(
            (
                CachedDataset(lazyDat).to_df()["features"][0].values
                == [100, 102, 104, 106, 108, 110, 112, 114, 116]
            ).all()
        )
        self.assertTrue(
            (
                CachedDataset(lazyDat).to_df()["labels"][0].values
                == [1, 2, 3, 4, 5, 6, 7, 8, 9]
            ).all()
        )

    @logTest
    def test_asPandasDataset(self):

        self.assertTrue(
            isinstance(CachedDataset(lazyDat).asPandasDataset, PandasDataset)
        )
        self.assertTrue(
            (
                CachedDataset(lazyDat).asPandasDataset.features[0].values
                == [100, 102, 104, 106, 108, 110, 112, 114, 116]
            ).all()
        )
        self.assertTrue(
            (
                CachedDataset(lazyDat).asPandasDataset.labels[0].values
                == [1, 2, 3, 4, 5, 6, 7, 8, 9]
            ).all()
        )


class PandasDatasetTests(TestCase):
    dataset: PandasDataset = PandasDataset(
        features=pd.concat(
            [
                pd.Series([1, np.nan, 2, 3], name="feat1"),
                pd.Series([1, 2, 3, 4], name="feat2"),
            ],
            axis=1,
        ),
        labels=pd.Series([0, 0, 0, 1], name="Label"),
    )

    dataset_no_label: PandasDataset = PandasDataset(
        features=pd.concat(
            [
                pd.Series([1, np.nan, 2, 3], name="feat1"),
                pd.Series([1, 2, 3, 4], name="feat2"),
            ],
            axis=1,
        )
    )

    @logTest
    def test_check_none(self):
        self.assertEqual(self.dataset._check_none(None), None)
        self.assertEqual(self.dataset._check_none("test"), "test")

    @logTest
    def test__len__(self):
        self.assertEqual(self.dataset.__len__(), 4)

    @logTest
    def test_items(self):

        self.assertTrue(isinstance(self.dataset.items, Iterator))
        self.assertEqual(next(self.dataset.items).features, {"feat1": 1.0, "feat2": 1})
        self.assertEqual(next(self.dataset.items).label["Label"], 0)
        self.assertEqual(
            next(self.dataset_no_label.items).features, {"feat1": 1.0, "feat2": 1}
        )
        self.assertEqual(next(self.dataset_no_label.items).label, None)

    @logTest
    def test_dropna_none_labels(self):
        res = pd.concat(
            [pd.Series([1, 2, 3], name="feat1"), pd.Series([1, 3, 4], name="feat2")],
            axis=1,
        )

        self.assertTrue(
            (
                self.dataset.dropna(subset=["feat1"]).features.reset_index(drop=True)
                == res
            )
            .all()
            .all()
        )
        self.assertTrue(
            (
                self.dataset.dropna(feat__subset=["feat1"]).features.reset_index(
                    drop=True
                )
                == res
            )
            .all()
            .all()
        )
        self.assertTrue(
            (
                self.dataset.dropna(labs__subset=["Label"]).features.reset_index(
                    drop=True
                )
                == res
            )
            .all()
            .all()
        )

    @logTest
    def test_cached(self):
        self.assertTrue(self.dataset.cached)

    @logTest
    def test_features_labels(self):
        self.assertEqual(
            self.dataset.features,
            pd.concat(
                [
                    pd.Series([1, np.nan, 2, 3], name="feat1"),
                    pd.Series([1, 2, 3, 4], name="feat2"),
                ],
                axis=1,
            ),
        )
        self.assertTrue((self.dataset.labels["Label"] == pd.Series([0, 0, 0, 1])).all())

    @logTest
    def test_index(self):
        self.assertTrue((self.dataset.index == range(4)).all())

    @logTest
    def test_createObject(self):

        self.assertTrue(
            isinstance(
                PandasDataset.createObject(
                    features=pd.concat(
                        [
                            pd.Series([1, np.nan, 2, 3], name="feat1"),
                            pd.Series([1, 2, 3, 4], name="feat2"),
                        ],
                        axis=1,
                    ),
                    labels=None,
                ),
                PandasDataset,
            )
        )
        self.assertEqual(
            PandasDataset.createObject(
                features=pd.concat(
                    [
                        pd.Series([1, np.nan, 2, 3], name="feat1"),
                        pd.Series([1, 2, 3, 4], name="feat2"),
                    ],
                    axis=1,
                ),
                labels=None,
            ).features,
            self.dataset_no_label.features,
        )
        self.assertEqual(
            PandasDataset.createObject(
                features=pd.concat(
                    [
                        pd.Series([1, np.nan, 2, 3], name="feat1"),
                        pd.Series([1, 2, 3, 4], name="feat2"),
                    ],
                    axis=1,
                ),
                labels=None,
            ).labels,
            self.dataset_no_label.labels,
        )

    @logTest
    def test_take(self):
        self.assertTrue(isinstance(self.dataset.takeAsPandas(1), PandasDataset))
        self.assertEqual(
            self.dataset.takeAsPandas(1).features.feat2, pd.Series([1], name="feat2")
        )
        self.assertEqual(
            self.dataset.takeAsPandas(1).labels["Label"], pd.Series([0], name="Label")
        )

    @logTest
    def test_loc(self):
        self.assertEqual(self.dataset.loc(2).features[2]["feat1"], 2)
        self.assertEqual(self.dataset.loc(2).features[2]["feat2"], 3)
        self.assertEqual(self.dataset.loc(2).labels[2]["Label"], 0)
        self.assertTrue(self.dataset_no_label.loc(2).labels is None)

    @logTest
    def test_from_sequence(self):
        features_1 = pd.DataFrame(
            {"feat1": [1, 2, 3, 4], "feat2": [100, 200, 300, 400]}, index=[1, 2, 3, 4]
        )
        features_2 = pd.DataFrame(
            {"feat1": [9, 11, 13, 14], "feat2": [90, 110, 130, 140]},
            index=[10, 11, 12, 13],
        )
        features_3 = pd.DataFrame(
            {"feat1": [90, 10, 10, 1400], "feat2": [0.9, 0.11, 0.13, 0.14]},
            index=[15, 16, 17, 18],
        )
        labels_1 = pd.DataFrame({"target": [1, 0, 1, 1]}, index=[1, 2, 3, 4])
        labels_2 = pd.DataFrame({"target": [1, 1, 1, 0]}, index=[10, 11, 12, 13])
        labels_3 = pd.DataFrame({"target": [0, 1, 1, 0]}, index=[15, 16, 17, 18])
        dataset_1 = PandasDataset(features_1, labels_1)
        dataset_2 = PandasDataset(features_2, labels_2)
        dataset_3 = PandasDataset(features_3, labels_3)
        dataset_merged = PandasDataset.from_sequence([dataset_1, dataset_2, dataset_3])
        self.assertEqual(
            pd.concat([features_1, features_2, features_3]), dataset_merged.features
        )
        self.assertEqual(
            pd.concat([labels_1, labels_2, labels_3]), dataset_merged.labels
        )

    @logTest
    def test_serialization(self):
        filename = os.path.join(TMP_FOLDER, "my_dataset.p")

        self.dataset.write(filename)

        newDataset: PandasDataset = PandasDataset.load(filename)

        self.assertTrue(isinstance(newDataset, PandasDataset))
        self.assertTrue(
            (self.dataset.features.fillna("NaN") == newDataset.features.fillna("NaN"))
            .all()
            .all()
        )

    @logTest
    def test_creation_from_samples(self):
        samples = [
            Sample(features=[100, 101], label=1, name=1),
            Sample(features=[102, 103], label=2, name=2),
            Sample(features=[104, 105], label=1, name=3),
            Sample(features=[106, 107], label=2, name=4),
            Sample(features=[108, 109], label=2, name=5),
            Sample(features=[110, 111], label=2, name=6),
            Sample(features=[112, 113], label=1, name=7),
            Sample(features=[114, 115], label=2, name=8),
            Sample(features=[116, 117], label=2, name=9),
        ]

        lazyDataset = CachedDataset(samples).filter(lambda x: x.label <= 5)

        assert isinstance(lazyDataset, LazyDataset)

        for format in ["pandas", "array", "dict"]:

            features1 = lazyDataset.getFeaturesAs(format)
            labels1 = lazyDataset.getLabelsAs(format)

            cached: CachedDataset = lazyDataset.asCached

            features2 = cached.getFeaturesAs(format)
            labels2 = cached.getLabelsAs(format)

            self.assertEqual(features1, features2)
            self.assertEqual(labels1, labels2)

            pandasDataset = cached.asPandasDataset

            features3 = pandasDataset.getFeaturesAs(format)
            labels3 = pandasDataset.getLabelsAs(format)

            self.assertEqual(features1, features3)
            self.assertEqual(labels1, labels3)

    @logTest
    def test_union(self):
        union = self.dataset.union(
            PandasDataset(
                features=pd.concat(
                    [
                        pd.Series([np.nan, 5, 6, 7], name="feat1"),
                        pd.Series([7, 8, 9, 10], name="feat2"),
                    ],
                    axis=1,
                ),
                labels=pd.Series([0, 0, 0, 1], name="Label"),
            )
        )

        self.assertTrue(isinstance(union, PandasDataset))
        self.assertEqual(
            union.features.reset_index(drop=True),
            pd.concat(
                [
                    pd.Series([1, np.nan, 2, 3, np.nan, 5, 6, 7], name="feat1"),
                    pd.Series([1, 2, 3, 4, 7, 8, 9, 10], name="feat2"),
                ],
                axis=1,
            ),
        )
        self.assertEqual(
            union.labels.Label.reset_index(drop=True),
            pd.Series([0, 0, 0, 1, 0, 0, 0, 1], name="Label"),
        )

    @logTest
    def test_intersection(self):
        other = PandasDataset(
            features=pd.concat(
                [
                    pd.Series([1, 2, 3, 4], name="feat1"),
                    pd.Series([5, 6, 7, 8], name="feat2"),
                ],
                axis=1,
            ),
            labels=pd.Series([1, 1, 0, 0], name="Label", index=[0, 1, 4, 5]),
        )

        self.assertEqual(other.intersection().labels.index.to_list(), [0, 1])
        self.assertEqual(other.intersection().features.index.to_list(), [0, 1])

    @logTest
    def test_getFeaturesAs(self):
        self.assertTrue(isinstance(self.dataset.getFeaturesAs("array"), np.ndarray))
        self.assertTrue(isinstance(self.dataset.getFeaturesAs("pandas"), pd.DataFrame))
        self.assertTrue(isinstance(self.dataset.getFeaturesAs("dict"), dict))

    @logTest
    def test_getLabelsAs(self):
        self.assertTrue(isinstance(self.dataset.getLabelsAs("array"), np.ndarray))
        self.assertTrue(isinstance(self.dataset.getLabelsAs("pandas"), pd.DataFrame))
        self.assertTrue(isinstance(self.dataset.getLabelsAs("dict"), dict))


class PandasTimeIndexedDatasetTests(TestCase):
    dates = pd.date_range("2010-01-01", "2010-01-04")

    dateStr = [str(x) for x in dates]

    dataset = PandasTimeIndexedDataset(
        features=pd.concat(
            [
                pd.Series([1, np.nan, 2, 3], index=dateStr, name="feat1"),
                pd.Series([1, 2, 3, 4], index=dateStr, name="feat2"),
            ],
            axis=1,
        )
    )

    @logTest
    def test_time_index(self):
        # duck-typing check
        days = [x.day for x in self.dataset.features.index]

        self.assertTrue(set(days), set(range(4)))

    @logTest
    def test_serialization(self):
        filename = os.path.join(TMP_FOLDER, "my_dataset.p")

        self.dataset.write(filename)

        newDataset = type(self.dataset).load(filename)

        self.assertTrue(isinstance(newDataset, PandasTimeIndexedDataset))
        self.assertTrue(
            (self.dataset.features.fillna("NaN") == newDataset.features.fillna("NaN"))
            .all()
            .all()
        )

    @logTest
    def test_createObject(self):

        NewDataset = self.dataset.createObject(
            features=pd.concat(
                [
                    pd.Series([1, 3], index=self.dateStr[0:2], name="feat1"),
                    pd.Series([1, 2], index=self.dateStr[0:2], name="feat2"),
                ],
                axis=1,
            ),
            labels=pd.Series([0, 0], index=self.dateStr[0:2], name="Label"),
        )

        self.assertTrue(isinstance(NewDataset, PandasTimeIndexedDataset))
        self.assertTrue(
            (
                NewDataset.features
                == pd.concat(
                    [
                        pd.Series(
                            [1, 3],
                            index=map(pd.to_datetime, self.dateStr[0:2]),
                            name="feat1",
                        ),
                        pd.Series(
                            [1, 2],
                            index=map(pd.to_datetime, self.dateStr[0:2]),
                            name="feat2",
                        ),
                    ],
                    axis=1,
                )
            )
            .all()
            .all()
        )
        self.assertTrue(
            (
                NewDataset.labels.values
                == pd.Series([0, 0], index=self.dateStr[0:2], name="Label").values
            ).all()
        )

    @logTest
    def test_loc(self):
        new_dataset = self.dataset.loc(
            [x for x in pd.date_range("2010-01-01", "2010-01-02")]
        )
        to_check = PandasTimeIndexedDataset(
            features=pd.DataFrame(self.dataset.features.iloc[:2])
        )
        self.assertIsInstance(new_dataset, PandasTimeIndexedDataset)
        self.assertEqual(new_dataset.features, to_check.features)


if __name__ == "__main__":
    unittest.main()
