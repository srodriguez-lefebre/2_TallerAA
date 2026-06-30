import tempfile
import unittest
from pathlib import Path

import pandas as pd


class KaggleCodeSubmissionTests(unittest.TestCase):
    def test_slugify_for_kaggle_id(self) -> None:
        from scripts.kaggle_code_submission import slugify

        self.assertEqual(slugify("Current835 Sep Temporal TTA1024"), "current835-sep-temporal-tta1024")

    def test_prepare_copy_kernel_writes_expected_metadata_and_submission(self) -> None:
        from scripts.kaggle_code_submission import prepare_copy_kernel_submission

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "candidate.csv"
            pd.DataFrame({"fname": ["a.wav"], "Bark": [0.5]}).to_csv(csv_path, index=False)

            prepared = prepare_copy_kernel_submission(
                csv_path=csv_path,
                slug="fat2019-example",
                title="FAT2019 example",
                owner="owner",
                competition="freesound-audio-tagging-2019",
                work_dir=tmp_path,
            )

            self.assertEqual(prepared.dataset_id, "owner/fat2019-example")
            self.assertEqual(prepared.kernel_id, "owner/fat2019-example-copy")
            self.assertTrue((prepared.dataset_dir / "submission.csv").exists())
            self.assertTrue((prepared.kernel_dir / "kernel-metadata.json").exists())
            self.assertIn("competition_sources", (prepared.kernel_dir / "kernel-metadata.json").read_text())
            self.assertIn("shutil.copyfile", prepared.code_file.read_text())


if __name__ == "__main__":
    unittest.main()
