"""Round-trip test for saving and loading recordings (no hardware, no board needed)."""

import numpy as np

from kestrel.acquire import list_recordings, load_recording, save_recording


def test_save_and_load_round_trip(tmp_path):
    eeg = np.random.default_rng(0).normal(size=(4, 512))
    path = save_recording(
        eeg, 256, "eyes_closed", note="test", requested_seconds=2.1, out_dir=tmp_path
    )
    loaded, meta = load_recording(path)
    assert np.array_equal(loaded, eeg)
    assert meta["label"] == "eyes_closed"
    assert meta["sampling_rate"] == 256
    assert meta["n_samples"] == 512
    assert meta["expected_samples"] == 537  # dropped samples stay visible
    assert meta["subject"] == "S01"
    assert list_recordings(tmp_path, labels=["eyes_closed"]) == [path]
    assert list_recordings(tmp_path, labels=["reading"]) == []
