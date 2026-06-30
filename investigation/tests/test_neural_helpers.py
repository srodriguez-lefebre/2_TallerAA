import unittest

import numpy as np

from scripts.fat2019.neural_helpers import (
    compute_pos_weight,
    make_train_valid_indices,
    sigmoid_numpy,
)


class NeuralHelperTests(unittest.TestCase):
    def test_compute_pos_weight_uses_negative_positive_ratio(self) -> None:
        targets = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )

        weights = compute_pos_weight(targets)

        np.testing.assert_allclose(weights, np.array([1.0, 3.0], dtype=np.float32))

    def test_compute_pos_weight_is_finite_for_empty_class(self) -> None:
        targets = np.zeros((4, 2), dtype=np.float32)

        weights = compute_pos_weight(targets)

        self.assertTrue(np.all(np.isfinite(weights)))
        np.testing.assert_allclose(weights, np.ones(2, dtype=np.float32))

    def test_sigmoid_numpy_maps_logits_to_probabilities(self) -> None:
        probabilities = sigmoid_numpy(np.array([-1000.0, 0.0, 1000.0]))

        self.assertEqual(probabilities.dtype, np.float32)
        np.testing.assert_allclose(probabilities, np.array([0.0, 0.5, 1.0]), atol=1e-6)

    def test_make_train_valid_indices_full_train_uses_all_rows(self) -> None:
        train_indices, valid_indices = make_train_valid_indices(
            num_rows=5,
            test_size=0.2,
            seed=42,
            full_train=True,
        )

        np.testing.assert_array_equal(train_indices, np.arange(5))
        self.assertEqual(valid_indices.size, 0)

    def test_make_train_valid_indices_holdout_keeps_nonempty_splits(self) -> None:
        train_indices, valid_indices = make_train_valid_indices(
            num_rows=10,
            test_size=0.2,
            seed=42,
            full_train=False,
        )

        self.assertEqual(train_indices.size, 8)
        self.assertEqual(valid_indices.size, 2)
        self.assertEqual(set(train_indices).intersection(set(valid_indices)), set())


if __name__ == "__main__":
    unittest.main()
