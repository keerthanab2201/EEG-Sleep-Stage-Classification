import os
import mne
import numpy as np
from typing import Tuple, Optional

DATASET_DIR = "dataset"
PSG_FILE = "SC4001E0-PSG.edf"
HYPNO_FILE = "SC4001EC-Hypnogram.edf"

STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def load_psg(filepath: str) -> Tuple[mne.io.Raw, np.ndarray, float]:
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=0)
    sfreq = raw.info["sfreq"]
    data = raw.get_data()
    return raw, data, sfreq


def pick_eeg_channel(raw: mne.io.Raw) -> str:
    ch_names = raw.ch_names
    print(f"Available channels: {ch_names}")
    preferred = "EEG Fpz-Cz"
    fallback = "EEG Pz-Oz"
    if preferred in ch_names:
        print(f"Using EEG channel: {preferred}")
        return preferred
    elif fallback in ch_names:
        print(f"Preferred {preferred} not found. Using fallback: {fallback}")
        return fallback
    else:
        raise ValueError(
            f"Neither {preferred} nor {fallback} found in channels: {ch_names}"
        )


def load_hypnogram(filepath: str) -> mne.Annotations:
    annot = mne.read_annotations(filepath)
    return annot


def annotations_to_labels(
    annot: mne.Annotations, n_epochs: int, epoch_duration: float = 30.0
) -> np.ndarray:
    labels = np.full(n_epochs, -1, dtype=int)
    epoch_times = np.arange(n_epochs) * epoch_duration
    for ann in annot:
        desc = ann["description"]
        if desc not in STAGE_MAP:
            continue
        onset = ann["onset"]
        duration = ann["duration"]
        end = onset + duration
        mask = (epoch_times >= onset) & (epoch_times < end)
        labels[mask] = STAGE_MAP[desc]
    return labels


def load_data() -> Tuple[np.ndarray, np.ndarray, float, int]:
    dataset_dir = DATASET_DIR
    psg_path = os.path.join(dataset_dir, PSG_FILE)
    hypno_path = os.path.join(dataset_dir, HYPNO_FILE)

    print("Loading PSG recording...")
    raw, data, sfreq = load_psg(psg_path)

    ch_name = pick_eeg_channel(raw)
    ch_idx = raw.ch_names.index(ch_name)
    eeg_data = data[ch_idx, :]

    print(f"EEG data shape: {eeg_data.shape}")
    print(f"Sampling frequency: {sfreq} Hz")
    print(
        f"Recording duration: {len(eeg_data) / sfreq:.1f}s ({len(eeg_data) / sfreq / 3600:.2f}h)"
    )

    print("\nLoading hypnogram annotations...")
    annot = load_hypnogram(hypno_path)
    print(f"Found {len(annot)} annotations")
    unique = set(a["description"] for a in annot)
    print(f"Unique labels: {unique}")

    epoch_duration = 30.0
    epoch_samples = int(sfreq * epoch_duration)
    n_epochs = len(eeg_data) // epoch_samples

    labels = annotations_to_labels(annot, n_epochs, epoch_duration)

    valid_mask = labels != -1
    labels = labels[valid_mask]

    n_valid = np.sum(valid_mask)
    X = np.zeros((n_valid, epoch_samples), dtype=np.float32)
    valid_indices = np.where(valid_mask)[0]
    for i, epoch_idx in enumerate(valid_indices):
        start = epoch_idx * epoch_samples
        end = start + epoch_samples
        X[i] = eeg_data[start:end]

    print(f"\nTotal 30s epochs: {n_epochs}")
    print(f"Valid epochs (after removing unknowns): {n_valid}")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {STAGE_NAMES[u]}: {c} epochs ({c / n_valid * 100:.1f}%)")

    return X, labels, sfreq, epoch_samples
