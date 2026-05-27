# normalize.py  —  compute and apply global (train-set) normalisation

import pickle
import numpy as np
import config


def fit(raw_sequences):
    """Compute mean and std from a list of raw (un-normalised) numpy arrays.
    Each array has shape (T, F).  Stats are computed across ALL frames.
    """
    all_frames = np.concatenate(raw_sequences, axis=0)   # (N_total_frames, F)
    mean = all_frames.mean(axis=0)
    std  = all_frames.std(axis=0) + 1e-5
    return mean, std


def transform(sequences, mean, std):
    """Apply pre-computed stats to a list of sequences."""
    return [(s - mean) / std for s in sequences]


def fit_transform(sequences):
    """Fit on sequences and immediately transform them (train set only)."""
    mean, std = fit(sequences)
    return transform(sequences, mean, std), mean, std


def save_stats(mean, std, path=config.STATS_PATH):
    with open(path, "wb") as f:
        pickle.dump({"mean": mean, "std": std}, f)
    print(f"Saved normalisation stats → {path}")


def load_stats(path=config.STATS_PATH):
    with open(path, "rb") as f:
        stats = pickle.load(f)
    return stats["mean"], stats["std"]