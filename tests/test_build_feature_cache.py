import unittest
from pathlib import Path

from scripts.build_feature_cache import cache_path


class BuildFeatureCacheTests(unittest.TestCase):
    def test_cache_path_uses_existing_basic_names(self) -> None:
        data_dir = Path("data")

        self.assertEqual(
            cache_path(data_dir, split="curated", feature_set="basic"),
            data_dir / "curated_logmel_stats_all.npz",
        )
        self.assertEqual(
            cache_path(data_dir, split="noisy", feature_set="basic"),
            data_dir / "noisy_logmel_stats_all.npz",
        )
        self.assertEqual(
            cache_path(data_dir, split="test", feature_set="basic"),
            data_dir / "test_logmel_stats.npz",
        )

    def test_cache_path_uses_feature_set_suffix_for_extended(self) -> None:
        data_dir = Path("data")

        self.assertEqual(
            cache_path(data_dir, split="curated", feature_set="extended"),
            data_dir / "curated_logmel_stats_extended.npz",
        )
        self.assertEqual(
            cache_path(data_dir, split="noisy", feature_set="extended"),
            data_dir / "noisy_logmel_stats_extended.npz",
        )
        self.assertEqual(
            cache_path(data_dir, split="test", feature_set="extended"),
            data_dir / "test_logmel_stats_extended.npz",
        )


if __name__ == "__main__":
    unittest.main()
