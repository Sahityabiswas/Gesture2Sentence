import argparse
import pickle

import config
from video_keypoint_extractor import VideoKeypointExtractor
from video_preprocess_utils import compare_sequences, summarize_flat_sequence, summarize_raw_sequence, flatten_sequence


def print_summary(title, raw_summary, flat_summary):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Frames               : {raw_summary['frames']}")
    print(f"Missing point ratio  : {raw_summary['missing_point_ratio']:.1%}")
    print(f"All-zero frame ratio : {raw_summary['all_zero_frame_ratio']:.1%}")
    print(f"X mean/std           : {raw_summary['x_mean']:.4f} / {raw_summary['x_std']:.4f}")
    print(f"Y mean/std           : {raw_summary['y_mean']:.4f} / {raw_summary['y_std']:.4f}")
    print(f"Flat mean/std        : {flat_summary['mean']:.4f} / {flat_summary['std']:.4f}")
    print(f"Flat min/max         : {flat_summary['min']:.4f} / {flat_summary['max']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare a raw video's extracted keypoints against a known dataset video_id."
    )
    parser.add_argument("--video", required=True, help="Path to the raw video file.")
    parser.add_argument("--vid", required=True, help="Reference video_id from data_map_FDMSE-ISL_keypoints.pkl")
    parser.add_argument("--save-extracted", default="", help="Optional .npy path to save extracted raw-video keypoints.")
    args = parser.parse_args()

    with open(config.DATA_MAP_PATH, "rb") as f:
        data_map = pickle.load(f)

    if args.vid not in data_map:
        raise KeyError(f"Reference video_id not found in data_map: {args.vid}")

    reference_seq = data_map[args.vid]
    extractor = VideoKeypointExtractor()
    candidate_seq = extractor.extract(args.video)

    if args.save_extracted:
        import numpy as np
        np.save(args.save_extracted, candidate_seq)
        print(f"Saved extracted sequence to: {args.save_extracted}")

    reference_raw = summarize_raw_sequence(reference_seq)
    reference_flat = summarize_flat_sequence(flatten_sequence(reference_seq))
    candidate_raw = summarize_raw_sequence(candidate_seq)
    candidate_flat = summarize_flat_sequence(flatten_sequence(candidate_seq))
    comparison = compare_sequences(reference_seq, candidate_seq)

    print_summary(f"Reference dataset sample: {args.vid}", reference_raw, reference_flat)
    print_summary(f"Extracted raw video: {args.video}", candidate_raw, candidate_flat)

    print("\nDirect comparison")
    print("-----------------")
    print(f"Common frames compared : {comparison['common_frames']}")
    print(f"Reference frame count  : {comparison['ref_frames']}")
    print(f"Candidate frame count  : {comparison['candidate_frames']}")
    print(f"Mean abs difference    : {comparison['mean_abs_diff']:.4f}")
    print(f"Max abs difference     : {comparison['max_abs_diff']:.4f}")
    print(f"Reference mean/std     : {comparison['ref_mean']:.4f} / {comparison['ref_std']:.4f}")
    print(f"Candidate mean/std     : {comparison['candidate_mean']:.4f} / {comparison['candidate_std']:.4f}")


if __name__ == "__main__":
    main()
