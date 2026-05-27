import os

import numpy as np
import torch

import normalize


EXPECTED_KEYPOINTS = 29
EXPECTED_DIMS = 2


def validate_sequence_shape(seq, expected_keypoints=EXPECTED_KEYPOINTS, expected_dims=EXPECTED_DIMS):
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected keypoint sequence with shape (T, {expected_keypoints}, {expected_dims}), "
            f"got {arr.shape}."
        )
    if arr.shape[0] == 0:
        raise ValueError("The extracted sequence is empty.")
    if arr.shape[1] != expected_keypoints or arr.shape[2] != expected_dims:
        raise ValueError(
            f"Expected keypoint sequence with shape (T, {expected_keypoints}, {expected_dims}), "
            f"got {arr.shape}."
        )
    return arr


def flatten_sequence(seq):
    arr = validate_sequence_shape(seq)
    return arr.reshape(arr.shape[0], -1)


def normalize_sequence(flat_seq, stats_path):
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")

    mean, std = normalize.load_stats(stats_path)
    arr = np.asarray(flat_seq, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Expected flattened sequence with shape (T, F), got {arr.shape}.")
    if arr.shape[1] != mean.shape[0]:
        raise ValueError(
            f"Feature size mismatch: sequence has {arr.shape[1]} features, "
            f"but stats expect {mean.shape[0]}."
        )

    return (arr - mean) / std


def sequence_to_model_input(seq, stats_path, device):
    flat = flatten_sequence(seq)
    norm = normalize_sequence(flat, stats_path)
    sample = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(device)
    return sample, [norm.shape[0]], norm


def summarize_flat_sequence(flat_seq):
    arr = np.asarray(flat_seq, dtype=np.float32)
    zero_ratio = float(np.mean(arr == 0.0))
    return {
        "frames": int(arr.shape[0]),
        "features": int(arr.shape[1]),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "zero_ratio": zero_ratio,
    }


def summarize_raw_sequence(seq):
    arr = validate_sequence_shape(seq)
    zero_mask = np.all(arr == 0.0, axis=2)
    all_zero_frames = np.all(zero_mask, axis=1)
    return {
        "frames": int(arr.shape[0]),
        "keypoints": int(arr.shape[1]),
        "dims": int(arr.shape[2]),
        "all_zero_frames": int(all_zero_frames.sum()),
        "all_zero_frame_ratio": float(all_zero_frames.mean()),
        "missing_point_ratio": float(zero_mask.mean()),
        "x_mean": float(arr[..., 0].mean()),
        "y_mean": float(arr[..., 1].mean()),
        "x_std": float(arr[..., 0].std()),
        "y_std": float(arr[..., 1].std()),
    }


def compare_sequences(reference_seq, candidate_seq):
    ref = flatten_sequence(reference_seq)
    cand = flatten_sequence(candidate_seq)

    common_len = min(ref.shape[0], cand.shape[0])
    if common_len <= 0:
        raise ValueError("Cannot compare empty sequences.")

    ref_trim = ref[:common_len]
    cand_trim = cand[:common_len]
    abs_diff = np.abs(ref_trim - cand_trim)

    return {
        "common_frames": int(common_len),
        "ref_frames": int(ref.shape[0]),
        "candidate_frames": int(cand.shape[0]),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
        "ref_mean": float(ref.mean()),
        "candidate_mean": float(cand.mean()),
        "ref_std": float(ref.std()),
        "candidate_std": float(cand.std()),
    }


def resample_sequence(seq, target_frames):
    arr = validate_sequence_shape(seq)
    if target_frames <= 1 or arr.shape[0] == target_frames:
        return arr.copy()

    source_idx = np.linspace(0, arr.shape[0] - 1, num=arr.shape[0], dtype=np.float32)
    target_idx = np.linspace(0, arr.shape[0] - 1, num=target_frames, dtype=np.float32)

    out = np.empty((target_frames, arr.shape[1], arr.shape[2]), dtype=np.float32)
    for keypoint_idx in range(arr.shape[1]):
        for dim_idx in range(arr.shape[2]):
            out[:, keypoint_idx, dim_idx] = np.interp(
                target_idx,
                source_idx,
                arr[:, keypoint_idx, dim_idx],
            )
    return out


def match_reference_distribution(seq, reference_mean, reference_std, eps=1e-6):
    arr = validate_sequence_shape(seq).copy()
    current_mean = arr.mean(axis=0, keepdims=True)
    current_std = arr.std(axis=0, keepdims=True)
    adjusted = ((arr - current_mean) / (current_std + eps)) * reference_std + reference_mean
    return adjusted.astype(np.float32)
