import unittest

import numpy as np

from scripts.geron_ensemble_search import (
    average_transformed_score_matrices,
    blend_score_matrices,
    blend_score_matrices_with_transform,
    build_current_system_scores,
    build_fashion_system_scores,
    fit_predict_classwise_logistic_oof,
    generate_dirichlet_weight_candidates,
    rowwise_minmax_scores,
    rowwise_zscore_scores,
    weighted_average_score_matrices,
)


class GeronEnsembleSearchTests(unittest.TestCase):
    def test_weighted_average_normalizes_positive_weights(self) -> None:
        matrices = {
            "a": np.array([[0.2, 0.4], [0.6, 0.8]]),
            "b": np.array([[0.8, 0.6], [0.4, 0.2]]),
        }

        averaged = weighted_average_score_matrices(matrices, {"a": 2.0, "b": 1.0})

        np.testing.assert_allclose(
            averaged,
            np.array([[0.4, 0.46666667], [0.53333333, 0.6]]),
        )

    def test_weighted_average_rejects_invalid_weights(self) -> None:
        matrices = {"a": np.zeros((2, 2))}

        with self.assertRaisesRegex(ValueError, "non-negative"):
            weighted_average_score_matrices(matrices, {"a": -1.0})

        with self.assertRaisesRegex(ValueError, "positive weight"):
            weighted_average_score_matrices(matrices, {"a": 0.0})

        with self.assertRaisesRegex(ValueError, "unknown"):
            weighted_average_score_matrices(matrices, {"missing": 1.0})

    def test_weighted_average_rejects_shape_mismatch(self) -> None:
        matrices = {
            "a": np.zeros((2, 2)),
            "b": np.zeros((2, 3)),
        }

        with self.assertRaisesRegex(ValueError, "same shape"):
            weighted_average_score_matrices(matrices, {"a": 1.0, "b": 1.0})

    def test_average_transformed_score_matrices_uses_fit_statistics(self) -> None:
        fit_matrices = {
            "a": np.array([[0.0, 2.0], [2.0, 4.0]]),
            "b": np.array([[1.0, 3.0], [3.0, 5.0]]),
        }
        matrices = {
            "a": np.array([[2.0, 4.0]]),
            "b": np.array([[3.0, 5.0]]),
        }

        transformed = average_transformed_score_matrices(
            matrices,
            ["a", "b"],
            method="z_prob_avg",
            fit_matrices=fit_matrices,
        )

        np.testing.assert_allclose(transformed, np.array([[1.0, 1.0]]))

    def test_generate_dirichlet_weight_candidates_is_deterministic(self) -> None:
        first = generate_dirichlet_weight_candidates(
            model_count=3,
            seed=7,
            trials_per_alpha=2,
            alpha_scales=(0.5, 1.0),
        )
        second = generate_dirichlet_weight_candidates(
            model_count=3,
            seed=7,
            trials_per_alpha=2,
            alpha_scales=(0.5, 1.0),
        )

        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(first.sum(axis=1), np.ones(first.shape[0]))
        np.testing.assert_allclose(first[0], np.array([1 / 3, 1 / 3, 1 / 3]))

    def test_build_fashion_system_scores_matches_documented_weights(self) -> None:
        sklearn = np.full((1, 2), 1.0)
        head = np.full((1, 2), 2.0)
        relu = np.full((1, 2), 4.0)
        literal = np.full((1, 2), 8.0)

        scores = build_fashion_system_scores(sklearn, head, relu, literal)

        expected = 0.15 * sklearn + 0.85 * (0.575 * head + 0.30 * relu + 0.125 * literal)
        np.testing.assert_allclose(scores, expected)

    def test_build_current_system_scores_matches_documented_weights(self) -> None:
        fashion = np.full((1, 2), 1.0)
        sepres = np.full((1, 2), 2.0)
        resnet = np.full((1, 2), 4.0)
        headsep = np.full((1, 2), 8.0)

        scores = build_current_system_scores(fashion, sepres, resnet, headsep)

        expected = 0.575 * fashion + 0.10 * sepres + 0.175 * resnet + 0.15 * headsep
        np.testing.assert_allclose(scores, expected)

    def test_blend_score_matrices_uses_branch_weight(self) -> None:
        base = np.array([[0.2, 0.8]])
        branch = np.array([[1.0, 0.0]])

        blended = blend_score_matrices(base, branch, branch_weight=0.25)

        np.testing.assert_allclose(blended, np.array([[0.4, 0.6]]))

    def test_rowwise_zscore_scores_normalizes_each_row(self) -> None:
        scores = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])

        normalized = rowwise_zscore_scores(scores)

        np.testing.assert_allclose(normalized[0], np.array([-1.22474487, 0.0, 1.22474487]))
        np.testing.assert_allclose(normalized[1], np.zeros(3))

    def test_rowwise_minmax_scores_maps_rows_to_unit_interval(self) -> None:
        scores = np.array([[2.0, 4.0, 6.0], [3.0, 3.0, 3.0]])

        normalized = rowwise_minmax_scores(scores)

        np.testing.assert_allclose(normalized[0], np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(normalized[1], np.zeros(3))

    def test_blend_score_matrices_with_transform_supports_row_z(self) -> None:
        base = np.array([[0.1, 0.5, 0.9]])
        branch = np.array([[0.9, 0.5, 0.1]])

        blended = blend_score_matrices_with_transform(
            base,
            branch,
            branch_weight=0.25,
            method="row_z",
        )

        expected = 0.75 * rowwise_zscore_scores(base) + 0.25 * rowwise_zscore_scores(branch)
        np.testing.assert_allclose(blended, expected)

    def test_fit_predict_classwise_logistic_oof_returns_probabilities(self) -> None:
        truth = np.array(
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [1, 1],
                [0, 0],
            ],
            dtype=np.float64,
        )
        matrices = {
            "strong": np.array(
                [
                    [0.9, 0.1],
                    [0.8, 0.2],
                    [0.2, 0.9],
                    [0.1, 0.8],
                    [0.7, 0.7],
                    [0.2, 0.1],
                ],
                dtype=np.float64,
            ),
            "weak": np.full((6, 2), 0.5, dtype=np.float64),
        }

        oof = fit_predict_classwise_logistic_oof(
            matrices,
            ["strong", "weak"],
            truth,
            n_splits=3,
            seed=11,
            c=0.1,
        )

        self.assertEqual(oof.shape, truth.shape)
        self.assertTrue(np.all(oof >= 0.0))
        self.assertTrue(np.all(oof <= 1.0))


if __name__ == "__main__":
    unittest.main()
