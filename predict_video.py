import argparse
import csv
import os
import numpy as np

import torch

import config
from hierarchical_inference import ensemble_predict, load_artifact_bundles, parse_artifact_dirs
from inference_utils import load_inference_settings
from video_keypoint_extractor import VideoKeypointExtractor
from video_preprocess_utils import (
    flatten_sequence,
    match_reference_distribution,
    resample_sequence,
    sequence_to_model_input,
    summarize_flat_sequence,
    summarize_raw_sequence,
)


def build_parser():
    saved_infer = load_inference_settings()

    parser = argparse.ArgumentParser(description="Predict a sign label from a raw video file.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--top-groups", type=int, default=saved_infer["top_k_groups"])
    parser.add_argument("--top-classes", type=int, default=saved_infer["top_k_classes"])
    parser.add_argument("--temperature", type=float, default=saved_infer["temperature"])
    parser.add_argument("--group-score-power", type=float, default=saved_infer["group_score_power"])
    parser.add_argument("--artifact-dirs", type=str, default="")
    parser.add_argument("--topk", type=int, default=5, help="Number of final predictions to print.")
    parser.add_argument("--min-confidence", type=float, default=0.25, help="Warn when top-1 score falls below this.")
    parser.add_argument("--unknown-label", type=str, default="Unknown", help="Label to show when confidence is very low.")
    parser.add_argument("--save-extracted", type=str, default="", help="Optional .npy path to save the extracted keypoint sequence.")
    parser.add_argument("--target-frames", type=int, default=150, help="Resample raw-video sequence to this many frames before inference.")
    parser.add_argument("--match-dataset-scale", action="store_true", help="Align extracted sequence distribution to dataset-wide raw feature stats.")
    return parser


def load_class_words(csv_path):
    mapping = {}
    if not os.path.exists(csv_path):
        return mapping

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = str(row.get("class", "")).strip()
            word = str(row.get("word", "")).strip()
            if class_id and word:
                mapping[class_id] = word
    return mapping


def display_label(global_id, inv_label_map, class_words):
    raw_label = str(inv_label_map.get(global_id, "?"))
    return class_words.get(raw_label, raw_label)


def print_predictions(output, inv_label_map, class_words, group_of_class, topk):
    print("\nTop predictions:")
    print(f"  {'Rank':<6} {'Label':<24} {'Group':<8} {'Score':>10}")
    print("  " + "-" * 54)

    for rank, (score, global_id) in enumerate(output["sorted_scores"][:topk], start=1):
        label = display_label(global_id, inv_label_map, class_words)
        gid = group_of_class.get(global_id, -1)
        print(f"  #{rank:<5} {label:<24} {gid:<8} {score:>10.4f}")


def print_quality_summary(raw_summary, flat_summary):
    print("\nVideo quality diagnostics:")
    print(f"  Frames               : {raw_summary['frames']}")
    print(f"  Missing point ratio  : {raw_summary['missing_point_ratio']:.1%}")
    print(f"  All-zero frame ratio : {raw_summary['all_zero_frame_ratio']:.1%}")
    print(f"  Flat feature std     : {flat_summary['std']:.4f}")
    print(f"  Flat feature zero %  : {flat_summary['zero_ratio']:.1%}")

    warnings = []
    if raw_summary["all_zero_frame_ratio"] > 0.20:
        warnings.append("many frames had no detected landmarks")
    if raw_summary["missing_point_ratio"] > 0.35:
        warnings.append("a large share of landmarks were missing")
    if flat_summary["std"] < 0.03:
        warnings.append("very low motion/variation was detected")

    if warnings:
        print("  Warning              : " + " | ".join(warnings))


def load_reference_raw_stats(data_map_path):
    import pickle

    with open(data_map_path, "rb") as f:
        data_map = pickle.load(f)

    sample = np.asarray(next(iter(data_map.values())), dtype=np.float32)
    if sample.ndim != 3:
        raise ValueError(f"Expected raw dataset samples with shape (T, 29, 2), got {sample.shape}.")

    total_sum = np.zeros((sample.shape[1], sample.shape[2]), dtype=np.float64)
    total_sq_sum = np.zeros((sample.shape[1], sample.shape[2]), dtype=np.float64)
    total_frames = 0

    for seq in data_map.values():
        arr = np.asarray(seq, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1:] != sample.shape[1:]:
            continue
        total_sum += arr.sum(axis=0)
        total_sq_sum += np.square(arr).sum(axis=0)
        total_frames += arr.shape[0]

    if total_frames == 0:
        raise ValueError("Could not compute dataset raw stats from data_map.")

    mean = total_sum / total_frames
    variance = np.maximum((total_sq_sum / total_frames) - np.square(mean), 1e-6)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    args = build_parser().parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dirs = parse_artifact_dirs(args.artifact_dirs)
    stats_path = os.path.join(artifact_dirs[0], "global_stats.pkl")

    print(f"Using device      : {device}")
    print(f"Video path        : {args.video}")
    print(f"Artifact dirs     : {artifact_dirs}")

    print("\nLoading trained artifacts...")
    bundles = load_artifact_bundles(artifact_dirs, device)
    label_map = bundles[0]["label_map"]
    inv_label_map = bundles[0]["inv_label_map"]
    group_info = bundles[0]["group_info"]
    group_of_class = group_info["group_of_class"]
    class_words = load_class_words(config.CLASS_MAP_PATH)

    print("Extracting keypoints from raw video...")
    extractor = VideoKeypointExtractor()
    seq = extractor.extract(args.video)
    print(f"Extracted sequence: {seq.shape}")
    if args.target_frames > 1 and seq.shape[0] != args.target_frames:
        seq = resample_sequence(seq, args.target_frames)
        print(f"Resampled sequence: {seq.shape}")
    if args.match_dataset_scale:
        ref_mean, ref_std = load_reference_raw_stats(config.DATA_MAP_PATH)
        seq = match_reference_distribution(seq, ref_mean, ref_std)
        print("Applied dataset-scale matching to extracted keypoints.")
    raw_summary = summarize_raw_sequence(seq)
    if args.save_extracted:
        np.save(args.save_extracted, seq)
        print(f"Saved extracted keypoints: {args.save_extracted}")

    print("Preparing model input...")
    flat_seq = flatten_sequence(seq)
    flat_summary = summarize_flat_sequence(flat_seq)
    sample, length, norm_seq = sequence_to_model_input(seq, stats_path, device)
    print(f"Flattened shape   : {norm_seq.shape}")
    print(f"Sequence length   : {length[0]}")
    print(f"Known classes     : {len(label_map)}")
    print_quality_summary(raw_summary, flat_summary)

    print("\nRunning hierarchical inference...")
    output = ensemble_predict(
        bundles,
        sample,
        length,
        args.top_groups,
        args.top_classes,
        args.temperature,
        args.group_score_power,
    )

    if not output["sorted_scores"]:
        print("No predictions were produced.")
        return

    best_class_id = output["best_class"]
    resolved_label = display_label(best_class_id, inv_label_map, class_words)
    best_score = output["sorted_scores"][0][0]
    best_group = group_of_class.get(best_class_id, -1)
    best_label = resolved_label if best_score >= args.min_confidence else args.unknown_label

    print("\nFinal prediction:")
    print(f"  Label      : {best_label}")
    print(f"  Group      : {best_group}")
    print(f"  Confidence : {best_score:.4f}")
    if best_score < args.min_confidence:
        print(f"  Reliability: LOW (best model guess was '{resolved_label}', below the warning threshold)")

    print_predictions(output, inv_label_map, class_words, group_of_class, args.topk)


if __name__ == "__main__":
    main()
