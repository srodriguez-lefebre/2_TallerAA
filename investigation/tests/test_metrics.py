import unittest

import numpy as np

from scripts.fat2019.metrics import calculate_overall_lwlrap


class LwlrapTests(unittest.TestCase):
    def test_matches_hand_calculated_pdf_example(self) -> None:
        truth = np.array(
            [
                [False, True, True],
                [True, False, False],
            ]
        )
        scores = np.array(
            [
                [0.7, 0.2, 0.5],
                [0.4, 0.6, 0.3],
            ]
        )

        self.assertAlmostEqual(calculate_overall_lwlrap(truth, scores), 5 / 9)

    def test_ignores_samples_without_labels(self) -> None:
        truth = np.array(
            [
                [False, False, False],
                [True, False, True],
            ]
        )
        scores = np.array(
            [
                [0.9, 0.8, 0.7],
                [0.9, 0.1, 0.8],
            ]
        )

        self.assertAlmostEqual(calculate_overall_lwlrap(truth, scores), 1.0)


if __name__ == "__main__":
    unittest.main()
