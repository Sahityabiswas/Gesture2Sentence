import argparse
import csv
import os
from collections import Counter

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

import config
from hierarchical_inference import ensemble_predict, load_artifact_bundles, parse_artifact_dirs
from inference_utils import load_inference_settings, save_inference_settings
import normalize
from dataset import SignDataset, build_label_map, collect_split, collate_fn, load_raw_data


SAVED_INFER = load_inference_settings()
DEFAULT_TOP_K_GROUPS = SAVED_INFER["top_k_groups"]
DEFAULT_TOP_K_CLASSES = SAVED_INFER["top_k_classes"]
DEFAULT_TEMPERATURE = SAVED_INFER["temperature"]
DEFAULT_GROUP_SCORE_POWER = SAVED_INFER["group_score_power"]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


parser = argparse.ArgumentParser(
    description="Evaluate hierarchical sign-language inference with optional hyperparameter sweep."
)
parser.add_argument("--top-groups", type=int, default=DEFAULT_TOP_K_GROUPS)
parser.add_argument("--top-classes", type=int, default=DEFAULT_TOP_K_CLASSES)
parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
parser.add_argument("--group-score-power", type=float, default=DEFAULT_GROUP_SCORE_POWER)
parser.add_argument(
    "--sweep",
    action="store_true",
    help="Run multiple inference settings and compare top-1/top-5 accuracy.",
)
parser.add_argument(
    "--group-grid",
    type=str,
    default="3,5",
    help="Comma-separated values for top-group sweep.",
)
parser.add_argument(
    "--class-grid",
    type=str,
    default="5,10",
    help="Comma-separated values for top-class sweep.",
)
parser.add_argument(
    "--temp-grid",
    type=str,
    default="1.0,1.2,1.5,2.0",
    help="Comma-separated values for temperature sweep.",
)
parser.add_argument(
    "--power-grid",
    type=str,
    default="1.0,0.7,0.5,0.3",
    help="Comma-separated values for group-score-power sweep.",
)
parser.add_argument(
    "--report-dir",
    type=str,
    default="evaluation_reports",
    help="Directory where per-video and per-class CSV reports are written.",
)
parser.add_argument(
    "--save-best-settings",
    action="store_true",
    help="When sweeping, save the best inference setting to inference_settings.json.",
)
parser.add_argument(
    "--artifact-dirs",
    type=str,
    default="",
    help="Comma-separated artifact directories for single-model or ensemble inference.",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

artifact_dirs = parse_artifact_dirs(args.artifact_dirs)
print("Artifact dirs:", artifact_dirs)
bundles = load_artifact_bundles(artifact_dirs, device)
label_map = bundles[0]["label_map"]
inv_label_map = bundles[0]["inv_label_map"]
group_of_class = bundles[0]["group_info"]["group_of_class"]
mean, std = normalize.load_stats(os.path.join(artifact_dirs[0], "global_stats.pkl"))

print("\nLoading test data...")
data_map, vid_class, vid_splits = load_raw_data()
_, allowed = build_label_map(vid_class)
te_seqs, te_labels, te_lengths, te_vids = collect_split(
    "test", data_map, vid_class, vid_splits, label_map, allowed, return_vids=True
)
te_seqs = normalize.transform(te_seqs, mean, std)
print(f"Test samples: {len(te_seqs)}")

test_loader = DataLoader(
    SignDataset(te_seqs, te_labels, te_lengths),
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
)


def evaluate_setting(top_k_groups, top_k_classes, temperature, group_score_power):
    all_preds = []
    all_targets = []
    all_top5 = []
    all_true_groups = []
    all_pred_groups = []
    group_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for batch_X, batch_y, batch_len in test_loader:
            batch_X = batch_X.to(device)

            for i in range(len(batch_y)):
                true_y = batch_y[i].item()
                sample_X = batch_X[i].unsqueeze(0)
                sample_len = [batch_len[i]]
                output = ensemble_predict(
                    bundles,
                    sample_X,
                    sample_len,
                    top_k_groups,
                    top_k_classes,
                    temperature,
                    group_score_power,
                )
                all_preds.append(output["best_class"])
                all_targets.append(true_y)

                top5_globals = output["top5_ids"]
                all_top5.append(top5_globals)
                if true_y in top5_globals:
                    top5_correct += 1

                true_group = group_of_class[true_y]
                if output["group_probs"] is not None and output["comparable_groups"]:
                    pred_group = int(torch.argmax(output["group_probs"]).item())
                    group_correct += int(pred_group == true_group)
                else:
                    pred_group = -1
                all_true_groups.append(true_group)
                all_pred_groups.append(pred_group)

    total = len(all_targets)
    top1_acc = sum(p == t for p, t in zip(all_preds, all_targets)) / total
    group_acc = group_correct / total
    top5_acc = top5_correct / total

    return {
        "top_groups": top_k_groups,
        "top_classes": top_k_classes,
        "temperature": temperature,
        "group_score_power": group_score_power,
        "group_acc": group_acc,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "preds": all_preds,
        "targets": all_targets,
        "top5": all_top5,
        "true_groups": all_true_groups,
        "pred_groups": all_pred_groups,
    }


def ensure_report_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def export_reports(result, report_dir):
    report_dir = ensure_report_dir(report_dir)
    per_video_path = os.path.join(report_dir, "per_video_predictions.csv")
    per_class_path = os.path.join(report_dir, "per_class_summary.csv")

    with open(per_video_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_id", "true_class_id", "true_label", "pred_class_id", "pred_label",
            "true_group", "pred_group", "top1_correct", "top5_correct", "top5_labels",
        ])
        for vid, true_id, pred_id, top5_ids, true_gid, pred_gid in zip(
            te_vids,
            result["targets"],
            result["preds"],
            result["top5"],
            result["true_groups"],
            result["pred_groups"],
        ):
            top5_labels = [str(inv_label_map.get(x, "?")) for x in top5_ids]
            writer.writerow([
                vid,
                true_id,
                str(inv_label_map.get(true_id, "?")),
                pred_id,
                str(inv_label_map.get(pred_id, "?")),
                true_gid,
                pred_gid,
                int(pred_id == true_id),
                int(true_id in top5_ids),
                "|".join(top5_labels),
            ])

    totals = Counter(result["targets"])
    top1_hits = Counter()
    top5_hits = Counter()
    for true_id, pred_id, top5_ids in zip(result["targets"], result["preds"], result["top5"]):
        if pred_id == true_id:
            top1_hits[true_id] += 1
        if true_id in top5_ids:
            top5_hits[true_id] += 1

    with open(per_class_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id", "label", "test_samples", "top1_correct", "top5_correct",
            "top1_acc", "top5_acc", "zero_top1",
        ])
        for class_id in sorted(totals):
            total = totals[class_id]
            writer.writerow([
                class_id,
                str(inv_label_map.get(class_id, "?")),
                total,
                top1_hits[class_id],
                top5_hits[class_id],
                round(top1_hits[class_id] / total, 4),
                round(top5_hits[class_id] / total, 4),
                int(top1_hits[class_id] == 0),
            ])

    zero_top1_classes = sum(1 for class_id in totals if top1_hits[class_id] == 0)
    print(f"\nSaved per-video report : {per_video_path}")
    print(f"Saved per-class report : {per_class_path}")
    print(f"Classes with zero top-1 correct on test: {zero_top1_classes}")


def print_single_result(result):
    print(
        f"\nRunning re-ranked inference "
        f"(top-{result['top_groups']} groups x top-{result['top_classes']} classes, "
        f"T={result['temperature']:.2f}, group_power={result['group_score_power']:.2f})..."
    )
    if any(gid < 0 for gid in result["pred_groups"]):
        print("Stage-1 group accuracy : n/a  (ensemble uses non-matching group maps)")
    else:
        print(f"Stage-1 group accuracy : {result['group_acc']:.4f}  (top-1 group correct)")
    print(f"Final  top-1  accuracy : {result['top1_acc']:.4f}  (exact label match)")
    print(f"Final  top-5  accuracy : {result['top5_acc']:.4f}  (true class in top-5 candidates)")

    show = list(range(min(20, len(label_map))))
    target_names = [str(inv_label_map[i]) for i in show]
    valid_preds = [p if p >= 0 else 0 for p in result["preds"]]
    print("\nPer-class report (first 20 classes):")
    print(
        classification_report(
            result["targets"],
            valid_preds,
            labels=show,
            target_names=target_names,
            zero_division=0,
        )
    )


def print_sweep_results(results):
    results = sorted(results, key=lambda r: (r["top1_acc"], r["top5_acc"]), reverse=True)
    print("\nSweep results:")
    print(
        f"  {'Rank':<6} {'TopG':>4} {'TopC':>4} {'Temp':>6} {'Power':>6} "
        f"{'GroupAcc':>9} {'Top1':>8} {'Top5':>8}"
    )
    print("  " + "-" * 61)

    for rank, result in enumerate(results, start=1):
        print(
            f"  #{rank:<5} {result['top_groups']:>4} {result['top_classes']:>4} "
            f"{result['temperature']:>6.2f} {result['group_score_power']:>6.2f} "
            f"{result['group_acc']:>9.4f} "
            f"{result['top1_acc']:>8.4f} {result['top5_acc']:>8.4f}"
        )

    best = results[0]
    print(
        f"\nBest setting: top-groups={best['top_groups']}, "
        f"top-classes={best['top_classes']}, temperature={best['temperature']:.2f}, "
        f"group-score-power={best['group_score_power']:.2f}"
    )
    print(f"Best top-1 accuracy: {best['top1_acc']:.4f}")
    print(f"Best top-5 accuracy: {best['top5_acc']:.4f}")
    return best


if args.sweep:
    group_grid = parse_int_list(args.group_grid)
    class_grid = parse_int_list(args.class_grid)
    temp_grid = parse_float_list(args.temp_grid)
    power_grid = parse_float_list(args.power_grid)

    print("\nRunning sweep over inference settings...")
    print(f"Top-group values : {group_grid}")
    print(f"Top-class values : {class_grid}")
    print(f"Temperature values: {temp_grid}")
    print(f"Group-power values: {power_grid}")

    sweep_results = []
    for top_k_groups in group_grid:
        for top_k_classes in class_grid:
            for temperature in temp_grid:
                for group_score_power in power_grid:
                    print(
                        f"  Evaluating top-groups={top_k_groups}, "
                        f"top-classes={top_k_classes}, temperature={temperature:.2f}, "
                        f"group-power={group_score_power:.2f}"
                    )
                    sweep_results.append(
                        evaluate_setting(
                            top_k_groups,
                            top_k_classes,
                            temperature,
                            group_score_power,
                        )
                    )

    best = print_sweep_results(sweep_results)
    export_reports(best, args.report_dir)
    if args.save_best_settings:
        saved = save_inference_settings({
            "top_k_groups": best["top_groups"],
            "top_k_classes": best["top_classes"],
            "temperature": best["temperature"],
            "group_score_power": best["group_score_power"],
        })
        print(f"Saved best inference settings to inference_settings.json: {saved}")
else:
    result = evaluate_setting(
        args.top_groups,
        args.top_classes,
        args.temperature,
        args.group_score_power,
    )
    print_single_result(result)
    export_reports(result, args.report_dir)

print("Next: use the best setting in predict.py or retrain if top-1 is still low.")
