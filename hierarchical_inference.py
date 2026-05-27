import os
import pickle

import torch
import torch.nn.functional as F

import config
from model import build_model


def _artifact_path(artifact_dir, filename):
    return os.path.join(artifact_dir, filename)


def parse_artifact_dirs(text=None):
    if not text:
        env_text = os.getenv("SLD_ENSEMBLE_DIRS", "")
        if env_text.strip():
            text = env_text
    if not text:
        return [config.ARTIFACT_DIR]
    dirs = [item.strip() for item in text.split(",") if item.strip()]
    return dirs or [config.ARTIFACT_DIR]


def load_artifact_bundle(artifact_dir, device):
    with open(_artifact_path(artifact_dir, "label_map.pkl"), "rb") as f:
        label_map = pickle.load(f)
    with open(_artifact_path(artifact_dir, "group_map.pkl"), "rb") as f:
        group_info = pickle.load(f)

    ckpt_g = torch.load(
        _artifact_path(artifact_dir, "group_model.pt"),
        map_location=device,
        weights_only=True,
    )
    group_model = build_model(
        ckpt_g["input_size"],
        ckpt_g["num_classes"],
        device,
        state_dict=ckpt_g["model_state"],
    )
    group_model.load_state_dict(ckpt_g["model_state"])
    group_model.eval()

    submodel_dir = _artifact_path(artifact_dir, "submodels")
    sub_models = {}
    for gid in group_info["classes_in_group"]:
        path = os.path.join(submodel_dir, f"group_{gid}.pt")
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = build_model(
            ckpt["input_size"],
            ckpt["num_classes"],
            device,
            state_dict=ckpt["model_state"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        sub_models[gid] = {
            "model": model,
            "inv_local_map": ckpt["inv_local_map"],
        }

    return {
        "artifact_dir": artifact_dir,
        "label_map": label_map,
        "inv_label_map": {v: k for k, v in label_map.items()},
        "group_info": group_info,
        "group_model": group_model,
        "sub_models": sub_models,
        "group_signature": group_info.get("group_signature"),
        "num_groups": group_info["num_groups"],
    }


def load_artifact_bundles(artifact_dirs, device):
    bundles = [load_artifact_bundle(path, device) for path in artifact_dirs]
    base_label_map = bundles[0]["label_map"]
    for bundle in bundles[1:]:
        if bundle["label_map"] != base_label_map:
            raise ValueError("All ensemble artifact dirs must share the same label_map.pkl")
    return bundles


def predict_bundle(bundle, sample_X, sample_len, top_k_groups, top_k_classes, temperature, group_score_power):
    with torch.no_grad():
        g_logits = bundle["group_model"](sample_X, sample_len)
        g_probs = F.softmax(g_logits, dim=1)[0]

    k_groups = min(top_k_groups, bundle["num_groups"])
    top_group_probs, top_group_ids = torch.topk(g_probs, k_groups)
    candidates = []
    global_scores = {}

    for g_prob, gid in zip(top_group_probs.tolist(), top_group_ids.tolist()):
        if gid not in bundle["sub_models"]:
            continue

        with torch.no_grad():
            sub_logits = bundle["sub_models"][gid]["model"](sample_X, sample_len)
            sub_probs = F.softmax(sub_logits / temperature, dim=1)[0]

        k_cls = min(top_k_classes, sub_probs.shape[0])
        top_cls_probs, top_cls_ids = torch.topk(sub_probs, k_cls)
        inv_local = bundle["sub_models"][gid]["inv_local_map"]

        for c_prob, local_idx in zip(top_cls_probs.tolist(), top_cls_ids.tolist()):
            global_id = inv_local[local_idx]
            combined_score = (g_prob ** group_score_power) * c_prob
            global_scores[global_id] = max(global_scores.get(global_id, 0.0), combined_score)
            candidates.append((combined_score, global_id, gid, g_prob, c_prob))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_class = candidates[0][1] if candidates else -1
    return {
        "best_class": best_class,
        "candidates": candidates,
        "global_scores": global_scores,
        "group_probs": g_probs,
    }


def ensemble_predict(bundles, sample_X, sample_len, top_k_groups, top_k_classes, temperature, group_score_power):
    bundle_outputs = [
        predict_bundle(bundle, sample_X, sample_len, top_k_groups, top_k_classes, temperature, group_score_power)
        for bundle in bundles
    ]

    aggregate_scores = {}
    for output in bundle_outputs:
        for global_id, score in output["global_scores"].items():
            aggregate_scores[global_id] = aggregate_scores.get(global_id, 0.0) + score

    if not aggregate_scores:
        return {
            "best_class": -1,
            "top5_ids": [],
            "sorted_scores": [],
            "group_probs": bundle_outputs[0]["group_probs"] if bundle_outputs else None,
            "comparable_groups": len(bundles) == 1,
            "bundle_outputs": bundle_outputs,
        }

    num_bundles = max(len(bundle_outputs), 1)
    sorted_scores = sorted(
        ((score / num_bundles, global_id) for global_id, score in aggregate_scores.items()),
        reverse=True,
    )
    top5_ids = [global_id for _, global_id in sorted_scores[:5]]

    comparable_groups = len({bundle["group_signature"] for bundle in bundles}) == 1
    group_probs = None
    if comparable_groups:
        stacked = torch.stack([output["group_probs"] for output in bundle_outputs], dim=0)
        group_probs = stacked.mean(dim=0)
    elif bundle_outputs:
        group_probs = bundle_outputs[0]["group_probs"]

    return {
        "best_class": sorted_scores[0][1],
        "top5_ids": top5_ids,
        "sorted_scores": sorted_scores,
        "group_probs": group_probs,
        "comparable_groups": comparable_groups,
        "bundle_outputs": bundle_outputs,
    }
