import argparse
import os
import pickle

import numpy as np
import torch

import config
from hierarchical_inference import ensemble_predict, load_artifact_bundles, parse_artifact_dirs
from inference_utils import load_inference_settings
import normalize


def get_vid_split(vid_id, vid_splits):
    for split_name, split_vids in vid_splits.items():
        if vid_id in split_vids:
            return split_name
    return "unknown"


def get_class_split_counts(raw_label, vid_class, vid_splits):
    counts = {}
    for split_name, split_vids in vid_splits.items():
        counts[split_name] = sum(1 for vid in split_vids if vid_class.get(vid) == raw_label)
    return counts


saved_infer = load_inference_settings()

parser = argparse.ArgumentParser()
parser.add_argument("--vid", type=str, default=None)
parser.add_argument("--top-groups", type=int, default=saved_infer["top_k_groups"])
parser.add_argument("--top-classes", type=int, default=saved_infer["top_k_classes"])
parser.add_argument("--temperature", type=float, default=saved_infer["temperature"])
parser.add_argument("--group-score-power", type=float, default=saved_infer["group_score_power"])
parser.add_argument("--artifact-dirs", type=str, default="")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
artifact_dirs = parse_artifact_dirs(args.artifact_dirs)

with open(config.DATA_MAP_PATH, "rb") as f:
    data_map = pickle.load(f)
with open(config.VID_CLASS_PATH, "rb") as f:
    vid_class = pickle.load(f)
with open(config.VID_SPLITS_PATH, "rb") as f:
    vid_splits = pickle.load(f)

bundles = load_artifact_bundles(artifact_dirs, device)
label_map = bundles[0]["label_map"]
inv_label_map = bundles[0]["inv_label_map"]
group_info = bundles[0]["group_info"]
group_of_class = group_info["group_of_class"]
classes_in_group = group_info["classes_in_group"]
num_groups = bundles[0]["num_groups"]
mean, std = normalize.load_stats(os.path.join(artifact_dirs[0], "global_stats.pkl"))

allowed = set(label_map.keys())
if args.vid:
    vid_id = args.vid
else:
    vid_id = next(v for v in vid_splits["test"] if v in data_map and vid_class.get(v) in allowed)

true_label = vid_class.get(vid_id, "unknown")
true_class_id = label_map.get(true_label, -1)
video_split = get_vid_split(vid_id, vid_splits)
class_counts = get_class_split_counts(true_label, vid_class, vid_splits) if true_label != "unknown" else {}

print(f"Artifact dirs: {artifact_dirs}")
print(f"Video      : {vid_id}")
print(f"Video split: {video_split}")
print(f"True label : {true_label}  (class id: {true_class_id})")
if class_counts:
    split_summary = " | ".join(
        f"{split_name}={class_counts.get(split_name, 0)}"
        for split_name in ("train", "val", "test")
        if split_name in vid_splits
    )
    print(f"Class split: {split_summary}")
if args.vid and video_split != "test":
    print(f"[Warning] Video {vid_id} comes from the '{video_split}' split, not 'test'.")

seq = np.array(data_map[vid_id], dtype=np.float32).reshape(-1, mean.shape[0])
seq = (seq - mean) / std
sample = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
length = [seq.shape[0]]

output = ensemble_predict(
    bundles,
    sample,
    length,
    args.top_groups,
    args.top_classes,
    args.temperature,
    args.group_score_power,
)

group_probs = output["group_probs"] if output["group_probs"] is not None else output["bundle_outputs"][0]["group_probs"]
k_g = min(args.top_groups, num_groups)
top_g = torch.topk(group_probs, k_g)

print(f"\nTop-{k_g} predicted groups (stage 1):")
for rank, (prob, gid) in enumerate(zip(top_g.values, top_g.indices), start=1):
    gid = gid.item()
    n_cls = len(classes_in_group.get(gid, []))
    true_tag = " <- TRUE" if group_of_class.get(true_class_id) == gid else ""
    print(f"  #{rank}  group {gid:3d}  ({n_cls} classes)  confidence {prob.item():.1%}{true_tag}")

print(f"\nRe-ranking across top-{args.top_groups} groups x top-{args.top_classes} classes each...")
print(f"Scoring rule: (group_prob ^ {args.group_score_power:.2f}) x class_prob")

print("\nTop-5 final predictions:")
print(f"  {'Rank':<6}  {'Class':<10}  {'Group':>6}  {'Combined':>10}")
print("  " + "-" * 40)
for rank, (combined, global_id) in enumerate(output["sorted_scores"][:5], start=1):
    class_name = str(inv_label_map.get(global_id, "?"))
    gid = group_of_class.get(global_id, -1)
    correct = " <- CORRECT" if global_id == true_class_id else ""
    print(f"  #{rank:<5}  {class_name:<10}  {gid:>6}  {combined:>10.4f}{correct}")

if output["sorted_scores"]:
    best_class_id = output["best_class"]
    best_class_name = str(inv_label_map.get(best_class_id, "?"))
    top5_class_ids = output["top5_ids"]
    top1_hit = best_class_id == true_class_id
    top5_hit = true_class_id in top5_class_ids

    print(f"\nFinal prediction : '{best_class_name}'")
    print(f"True label       : '{true_label}'")
    print(f"Top-1 correct    : {'YES' if top1_hit else 'NO'}")
    print(f"Top-5 correct    : {'YES' if top5_hit else 'NO'}")

    true_gid = group_of_class.get(true_class_id, -1)
    true_group_rank = None
    true_group_prob = None
    for rank, (prob, gid) in enumerate(zip(top_g.values.tolist(), top_g.indices.tolist()), start=1):
        if gid == true_gid:
            true_group_rank = rank
            true_group_prob = prob
            break

    print("\nTrue-group diagnostics:")
    if true_group_rank is None:
        print(f"  True group      : {true_gid} was not in top-{k_g} stage-1 groups")
    else:
        print(f"  True group      : {true_gid} at stage-1 rank #{true_group_rank} with prob {true_group_prob:.1%}")

    bundle0_candidates = output["bundle_outputs"][0]["candidates"]
    true_group_candidates = [item for item in bundle0_candidates if item[2] == true_gid]
    if true_group_candidates:
        top_local = true_group_candidates[:args.top_classes]
        in_topk = any(global_id == true_class_id for _, global_id, _, _, _ in top_local)
        print(f"  In true-group top-{args.top_classes}: {'YES' if in_topk else 'NO'}")
        print(f"  True-group top-{args.top_classes} classes (bundle #1 view):")
        for rank, (combined, global_id, _, _, _) in enumerate(top_local, start=1):
            class_name = str(inv_label_map.get(global_id, "?"))
            marker = " <- TRUE" if global_id == true_class_id else ""
            print(f"    #{rank}  {class_name:<10}  score={combined:.4f}{marker}")
