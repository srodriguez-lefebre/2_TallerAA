import unittest
from pathlib import Path

import numpy as np

from scripts.fat2019.spectrogram_images import crop_or_pad_frames, normalize_logmel_image
from scripts.build_logmel_image_cache import logmel_image_cache_path, parse_splits


class SpectrogramImageTests(unittest.TestCase):
    def test_crop_or_pad_frames_pads_short_matrix(self) -> None:
        image = np.ones((3, 2), dtype=np.float32)

        fixed = crop_or_pad_frames(image, frames=5)

        self.assertEqual(fixed.shape, (3, 5))
        np.testing.assert_allclose(fixed[:, :2], 1.0)
        np.testing.assert_allclose(fixed[:, 2:], 0.0)

    def test_crop_or_pad_frames_center_crops_long_matrix(self) -> None:
        image = np.arange(18, dtype=np.float32).reshape(3, 6)

        fixed = crop_or_pad_frames(image, frames=4)

        np.testing.assert_array_equal(fixed, image[:, 1:5])

    def test_normalize_logmel_image_returns_finite_zero_mean_image(self) -> None:
        image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        normalized = normalize_logmel_image(image)

        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(normalized)))
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=6)

    def test_parse_splits_rejects_unknown_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            parse_splits("curated,noisy")

    def test_logmel_image_cache_path_includes_shape(self) -> None:
        self.assertEqual(
            str(logmel_image_cache_path(Path("data"), split="curated", n_mels=128, frames=512)),
            "data/curated_logmel_image_m128_f512.npz",
        )


if __name__ == "__main__":
    unittest.main()
