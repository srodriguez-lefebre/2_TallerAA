import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ModuleNotFoundError:
    np = None
    torch = None

from scripts.predict_logmel_cnn import resolve_submission_path


class PredictLogmelCnnTests(unittest.TestCase):
    def test_resolve_submission_path_uses_explicit_output(self) -> None:
        output_path = resolve_submission_path(
            submissions_dir=Path("submissions/cnn"),
            checkpoint_path=Path("models/model.pt"),
            output_path=Path("custom/submission.csv"),
        )

        self.assertEqual(output_path, Path("custom/submission.csv"))

    def test_resolve_submission_path_uses_checkpoint_stem(self) -> None:
        output_path = resolve_submission_path(
            submissions_dir=Path("submissions/cnn"),
            checkpoint_path=Path("models/small_logmel_cnn_best.pt"),
            output_path=None,
        )

        self.assertEqual(output_path, Path("submissions/cnn/small_logmel_cnn_best.csv"))

    def test_parse_tta_views_removes_whitespace_and_duplicates(self) -> None:
        from scripts.predict_logmel_cnn import parse_tta_views

        self.assertEqual(parse_tta_views("start, center,end,center"), ["start", "center", "end"])

    @unittest.skipUnless(np is not None, "numpy is required")
    def test_select_frame_view_crops_expected_window(self) -> None:
        from scripts.predict_logmel_cnn import select_frame_view

        images = np.arange(12, dtype=np.float32).reshape(1, 2, 6)

        np.testing.assert_array_equal(select_frame_view(images, frames=4, view="start"), images[:, :, :4])
        np.testing.assert_array_equal(select_frame_view(images, frames=4, view="center"), images[:, :, 1:5])
        np.testing.assert_array_equal(select_frame_view(images, frames=4, view="end"), images[:, :, 2:6])

    @unittest.skipUnless(torch is not None, "torch is required")
    def test_enable_dropout_modules_keeps_batchnorm_in_eval_mode(self) -> None:
        from scripts.predict_logmel_cnn import enable_dropout_modules

        model = torch.nn.Sequential(
            torch.nn.BatchNorm1d(3),
            torch.nn.Dropout(p=0.5),
        )
        model.eval()

        enable_dropout_modules(model)

        self.assertFalse(model[0].training)
        self.assertTrue(model[1].training)


if __name__ == "__main__":
    unittest.main()
