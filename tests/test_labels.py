import unittest

import numpy as np
import pandas as pd

from scripts.fat2019.labels import dataframe_to_multihot


class LabelTests(unittest.TestCase):
    def test_dataframe_to_multihot_preserves_label_order(self) -> None:
        labels = pd.DataFrame(
            {
                "fname": ["a.wav", "b.wav"],
                "labels": ["Bark,Siren", "Siren"],
            }
        )

        y = dataframe_to_multihot(labels, ["Bark", "Siren", "Applause"])

        expected = np.array(
            [
                [1, 1, 0],
                [0, 1, 0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(y, expected)


if __name__ == "__main__":
    unittest.main()
