from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.fat2019.data import split_labels


def dataframe_to_multihot(labels: pd.DataFrame, label_columns: list[str]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(label_columns)}
    y = np.zeros((len(labels), len(label_columns)), dtype=np.float32)
    for row_index, row_labels in enumerate(labels["labels"]):
        for label in split_labels(row_labels):
            if label in label_to_index:
                y[row_index, label_to_index[label]] = 1.0
    return y
