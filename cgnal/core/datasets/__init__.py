"""Module containing samples datasets to be used for testing."""

import os

import pandas as pd
from cgnal.core.data.model.ml import PandasDataset, PandasTimeIndexedDataset

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _create_nyc_dataset(filename: str) -> PandasTimeIndexedDataset:
    df = pd.read_csv(filename, index_col="Date")
    return PandasTimeIndexedDataset(features=df.drop("TempM", axis=1), labels=df.TempM)


weather_nyc = _create_nyc_dataset(os.path.join(DATA_FOLDER, "weather_nyc_short.csv"))

unbalanced: PandasDataset = PandasDataset.createObject(
    pd.read_pickle(os.path.join(DATA_FOLDER, "unbalanced_features.p")),
    pd.read_pickle(os.path.join(DATA_FOLDER, "unbalanced_labels.p")),
)
