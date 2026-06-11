import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.train_sklearn_variants import (
    load_cached_features,
    load_feature_matrices,
    make_model,
    parse_model_names,
)


class SklearnVariantTests(unittest.TestCase):
    def test_parse_model_names_splits_and_trims(self) -> None:
        self.assertEqual(
            parse_model_names(" logreg_c03,extra_trees , mlp_small "),
            ["logreg_c03", "extra_trees", "mlp_small"],
        )

    def test_parse_model_names_rejects_empty_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            parse_model_names(" , ")

    def test_make_model_supports_sgd_log_loss_variant(self) -> None:
        model = make_model("sgd_log_alpha1e4", seed=42)

        self.assertTrue(hasattr(model, "fit"))

    def test_make_model_supports_fine_logreg_variant(self) -> None:
        model = make_model("logreg_c0008", seed=42)

        self.assertTrue(hasattr(model, "fit"))

    def test_load_feature_matrices_can_include_noisy_cache(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            np.savez_compressed(data_dir / "curated_logmel_stats_all.npz", x=np.ones((2, 3)))
            np.savez_compressed(data_dir / "noisy_logmel_stats_all.npz", x=np.full((4, 3), 2.0))
            np.savez_compressed(data_dir / "test_logmel_stats.npz", x=np.zeros((1, 3)))

            matrices = load_feature_matrices(data_dir, "basic", include_noisy=True)

            self.assertEqual(matrices.curated.shape, (2, 3))
            self.assertEqual(matrices.noisy.shape, (4, 3))
            self.assertEqual(matrices.test.shape, (1, 3))

    def test_load_cached_features_preserves_legacy_tuple(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            np.savez_compressed(data_dir / "curated_logmel_stats_all.npz", x=np.ones((2, 3)))
            np.savez_compressed(data_dir / "test_logmel_stats.npz", x=np.zeros((1, 3)))

            curated, test = load_cached_features(data_dir, "basic")

            self.assertEqual(curated.shape, (2, 3))
            self.assertEqual(test.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
