import unittest

import numpy as np

from scripts.fat2019.features import (
    extract_log_mel_stats,
    extract_log_mel_stats_extended,
    log_mel_spectrogram,
    mel_filterbank,
)


class FeatureTests(unittest.TestCase):
    def test_mel_filterbank_has_requested_shape(self) -> None:
        filters = mel_filterbank(
            sample_rate=16000,
            n_fft=512,
            n_mels=40,
            fmin=20.0,
            fmax=8000.0,
        )

        self.assertEqual(filters.shape, (257, 40))
        self.assertTrue(np.all(filters >= 0.0))

    def test_log_mel_spectrogram_is_finite_and_channel_first(self) -> None:
        sample_rate = 16000
        seconds = 1
        t = np.linspace(0.0, seconds, sample_rate * seconds, endpoint=False)
        waveform = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        features = log_mel_spectrogram(
            waveform,
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=160,
            n_mels=64,
        )

        self.assertEqual(features.shape[0], 64)
        self.assertGreater(features.shape[1], 1)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_extract_log_mel_stats_is_fixed_length(self) -> None:
        waveform = np.ones(4096, dtype=np.float32)

        features = extract_log_mel_stats(
            waveform,
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=32,
        )

        self.assertEqual(features.shape, (32 * 4 + 3,))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_extract_log_mel_stats_extended_is_fixed_length(self) -> None:
        waveform = np.ones(4096, dtype=np.float32)

        features = extract_log_mel_stats_extended(
            waveform,
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=32,
        )

        self.assertEqual(features.shape, (32 * 7 + 3,))
        self.assertTrue(np.all(np.isfinite(features)))


if __name__ == "__main__":
    unittest.main()
