import unittest

import numpy as np

from scripts.train_cv_ensemble import average_score_matrices


class CvEnsembleTests(unittest.TestCase):
    def test_average_score_matrices_averages_same_shape_arrays(self) -> None:
        averaged = average_score_matrices(
            [
                np.array([[0.2, 0.6], [0.8, 0.1]]),
                np.array([[0.4, 0.2], [0.6, 0.5]]),
            ]
        )

        np.testing.assert_allclose(averaged, np.array([[0.3, 0.4], [0.7, 0.3]]))

    def test_average_score_matrices_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            average_score_matrices([])

    def test_average_score_matrices_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            average_score_matrices(
                [
                    np.zeros((2, 2)),
                    np.zeros((2, 3)),
                ]
            )


if __name__ == "__main__":
    unittest.main()
