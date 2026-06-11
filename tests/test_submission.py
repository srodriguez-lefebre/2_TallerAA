import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.fat2019.submission import (
    build_model_submission,
    build_prior_submission,
    validate_submission,
)


class ConstantModel:
    def predict_proba(self, x):
        return [[0.2, 0.8] for _ in range(len(x))]


class SubmissionTests(unittest.TestCase):
    def test_build_prior_submission_uses_expected_columns_and_probabilities(self) -> None:
        sample = pd.DataFrame(
            {
                "fname": ["a.wav", "b.wav"],
                "Bark": [0.0, 0.0],
                "Siren": [0.0, 0.0],
            }
        )
        priors = {"Bark": 0.25, "Siren": 0.75}

        submission = build_prior_submission(sample, priors)

        self.assertEqual(list(submission.columns), ["fname", "Bark", "Siren"])
        self.assertEqual(submission["fname"].tolist(), ["a.wav", "b.wav"])
        self.assertEqual(submission["Bark"].tolist(), [0.25, 0.25])
        self.assertEqual(submission["Siren"].tolist(), [0.75, 0.75])
        validate_submission(submission, ["Bark", "Siren"], expected_rows=2)

    def test_validate_submission_rejects_wrong_label_order(self) -> None:
        submission = pd.DataFrame(
            {
                "fname": ["a.wav"],
                "Siren": [0.75],
                "Bark": [0.25],
            }
        )

        with self.assertRaisesRegex(ValueError, "columns"):
            validate_submission(submission, ["Bark", "Siren"], expected_rows=1)

    def test_validate_submission_rejects_probabilities_outside_unit_interval(self) -> None:
        submission = pd.DataFrame(
            {
                "fname": ["a.wav"],
                "Bark": [1.25],
                "Siren": [0.25],
            }
        )

        with self.assertRaisesRegex(ValueError, "probabilities"):
            validate_submission(submission, ["Bark", "Siren"], expected_rows=1)

    def test_prior_submission_can_be_written_and_read_back(self) -> None:
        sample = pd.DataFrame({"fname": ["a.wav"], "Bark": [0.0]})
        submission = build_prior_submission(sample, {"Bark": 0.4})

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "submission.csv"
            submission.to_csv(path, index=False)
            loaded = pd.read_csv(path)

        validate_submission(loaded, ["Bark"], expected_rows=1)

    def test_build_model_submission_uses_model_probabilities(self) -> None:
        sample = pd.DataFrame(
            {
                "fname": ["a.wav", "b.wav"],
                "Bark": [0.0, 0.0],
                "Siren": [0.0, 0.0],
            }
        )

        submission = build_model_submission(sample, ["Bark", "Siren"], ConstantModel(), [[1], [2]])

        self.assertEqual(submission["Bark"].tolist(), [0.2, 0.2])
        self.assertEqual(submission["Siren"].tolist(), [0.8, 0.8])
        validate_submission(submission, ["Bark", "Siren"], expected_rows=2)


if __name__ == "__main__":
    unittest.main()
