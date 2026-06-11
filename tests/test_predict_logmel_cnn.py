import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
