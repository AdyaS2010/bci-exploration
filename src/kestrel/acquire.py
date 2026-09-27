"""Record EEG from the Muse 2 (or BrainFlow's synthetic board) and save it with metadata.

Each recording is two files in data/raw/:
    S01_20261004-153012_eyes_closed.npy   raw EEG, shape (4 channels, n samples), µV, unfiltered
    S01_20261004-153012_eyes_closed.json  sidecar: label, note, rate, channels, counts, time

Raw data is saved unfiltered so any preprocessing choice can be changed and rerun later.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams

from kestrel import __version__, config


def board_id(use_synthetic=None):
    use_synthetic = config.USE_SYNTHETIC if use_synthetic is None else use_synthetic
    return config.SYNTHETIC_BOARD_ID if use_synthetic else config.MUSE_BOARD_ID


def eeg_layout(board):
    """Which data rows hold EEG, and their names.

    Returns (row indices, names used by Kestrel, names the board itself reports).
    For the Muse these match. The synthetic board's first four channels are relabelled
    to the Muse montage so the rest of the pipeline runs unchanged.
    """
    rows = BoardShim.get_eeg_channels(board)[: len(config.CHANNELS)]
    source_names = BoardShim.get_eeg_names(board)[: len(config.CHANNELS)]
    return rows, list(config.CHANNELS), source_names


def connect(use_synthetic=None, serial_number="", timeout_s=15):
    """Open a BrainFlow session. serial_number can pin a specific Muse (e.g. "Muse-1A2B")."""
    params = BrainFlowInputParams()
    params.serial_number = serial_number
    params.timeout = timeout_s
    board = BoardShim(board_id(use_synthetic), params)
    board.prepare_session()
    return board


def record(board, seconds, settle_s=config.SETTLE_SECONDS):
    """Stream for `seconds` and return (eeg in µV with shape (4, n), sampling rate).

    The first `settle_s` seconds are discarded: right after the stream starts, the Muse
    often delivers a burst of dropped or offset samples that would contaminate the
    first epochs.
    """
    bid = board.get_board_id()
    fs = BoardShim.get_sampling_rate(bid)
    rows, _, _ = eeg_layout(bid)
    board.start_stream()
    try:
        time.sleep(settle_s)
        board.get_board_data()  # empties the buffer, throwing away the settling period
        time.sleep(seconds)
        data = board.get_board_data()
    finally:
        board.stop_stream()
    return data[rows].astype(float), fs


def recording_stem(label, subject=config.DEFAULT_SUBJECT, when=None):
    when = datetime.now() if when is None else when
    return f"{subject}_{when:%Y%m%d-%H%M%S}_{label}"


def save_recording(
    eeg,
    fs,
    label,
    note="",
    subject=config.DEFAULT_SUBJECT,
    board_name="",
    source_channel_names=None,
    requested_seconds=None,
    out_dir=None,
):
    """Write the raw array and its JSON sidecar. Returns the .npy path.

    `requested_seconds` lets the sidecar record expected vs actual samples. Bluetooth
    drops packets, and a recording that quietly lost 10% of its data should say so.
    """
    out_dir = Path(config.RAW_DIR if out_dir is None else out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    stem = recording_stem(label, subject, now)
    npy_path = out_dir / f"{stem}.npy"
    np.save(npy_path, eeg)

    meta = {
        "label": label,
        "note": note,
        "subject": subject,
        "board": board_name,
        "sampling_rate": fs,
        "channels": list(config.CHANNELS),
        "source_channel_names": source_channel_names or list(config.CHANNELS),
        "units": "microvolts",
        "filtering": "none (raw)",
        "n_samples": int(eeg.shape[1]),
        "duration_s": round(eeg.shape[1] / fs, 3),
        "expected_samples": int(requested_seconds * fs) if requested_seconds else None,
        "recorded_at": now.isoformat(timespec="seconds"),
        "kestrel_version": __version__,
    }
    npy_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return npy_path


def load_recording(npy_path):
    """Load (eeg, metadata) for one recording."""
    npy_path = Path(npy_path)
    meta = json.loads(npy_path.with_suffix(".json").read_text())
    return np.load(npy_path), meta


def list_recordings(raw_dir=None, labels=None, subject=None):
    """Recordings in time order, optionally filtered by label(s) and subject."""
    raw_dir = Path(config.RAW_DIR if raw_dir is None else raw_dir)
    paths = []
    for json_path in sorted(raw_dir.glob("*.json")):
        meta = json.loads(json_path.read_text())
        if labels is not None and meta["label"] not in labels:
            continue
        if subject is not None and meta["subject"] != subject:
            continue
        paths.append(json_path.with_suffix(".npy"))
    return paths


def record_block(label, seconds, note="", subject=config.DEFAULT_SUBJECT, use_synthetic=None):
    """Connect, record one labelled block, disconnect, save. Returns the .npy path.

    Synthetic-board recordings go to data/synthetic/, real ones to data/raw/.
    """
    board = connect(use_synthetic)
    try:
        bid = board.get_board_id()
        _, _, source_names = eeg_layout(bid)
        eeg, fs = record(board, seconds)
    finally:
        board.release_session()
    synthetic = bid == config.SYNTHETIC_BOARD_ID
    return save_recording(
        eeg,
        fs,
        label,
        note=note,
        subject=subject,
        board_name="synthetic" if synthetic else "muse_2",
        source_channel_names=source_names,
        requested_seconds=seconds,
        out_dir=config.SYNTHETIC_DIR if synthetic else config.RAW_DIR,
    )
