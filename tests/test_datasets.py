from cgnal.core.data.model.ml import PandasDataset, PandasTimeIndexedDataset
from cgnal.core.datasets import weather_nyc, unbalanced
from cgnal.core.tests.core import logTest, TestCase


class TestLoadDatasets(TestCase):
    @logTest
    def test_unbalanced_dataset(self):
        self.assertIsInstance(unbalanced, PandasDataset)

    @logTest
    def test_time_indexed_dataset(self):
        self.assertIsInstance(weather_nyc, PandasTimeIndexedDataset)
