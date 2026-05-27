import argparse
import pickle

import numpy as np
import torch

import config
from dataset_ctrgcn import load_raw_data, load_stats
from model_ctrgcn import CTRGCNSignModel


def prepare_sample(video_id, data_map, mean, std):
    if video_id not in data_map:
        raise KeyError(f"Video id '{video_id}' not found in data_map.")

    seq = np.asarray(data_map[video_id], dtype=np.float32)
    if seq.ndim == 2:
        expected = config.NUM_KEYPOINTS * config.INPUT_CHANNELS
        if seq.shape[1] != expected:
            raise ValueError(
                f"Expected flattened feature dim {expected}, got {seq.shape[1]}."
            )
        seq = seq.reshape(seq.shape[0], config.NUM_KEYPOINTS, config.INPUT_CHANNELS)
    elif seq.ndim != 3:
        raise ValueError(f"Unexpected sequence shape: {seq.shape}")

    seq = (seq - mean) / std
    x = torch.tensor(seq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    lengths = [seq.shape[0]]
    return x, lengths


@torch.no_grad()
def predict_video_id(video_id, checkpoint_path, top_k):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CTRGCNSignModel(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(config.LABEL_MAP_PATH, "rb") as f:
        label_map = pickle.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}

    data_map, _, _ = load_raw_data()
    mean, std = load_stats()
    sample_x, lengths = prepare_sample(video_id, data_map, mean, std)
    sample_x = sample_x.to(device)

    logits = model(sample_x, lengths)
    probs = torch.softmax(logits, dim=1)[0]

    top_k = min(top_k, probs.shape[0])
    top_probs, top_ids = torch.topk(probs, k=top_k)

    results = []
    for rank, (class_idx, prob) in enumerate(zip(top_ids.tolist(), top_probs.tolist()), start=1):
        results.append(
            {
                "rank": rank,
                "class_id": class_idx,
                "label": inv_label_map.get(class_idx, "?"),
                "probability": prob,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict one video_id with the separate CTRGCN model.")
    parser.add_argument("--video-id", required=True, help="Video id from data_map_FDMSE-ISL_keypoints.pkl")
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH, help="Path to trained CTRGCN checkpoint")
    parser.add_argument("--top-k", type=int, default=5, help="How many predictions to show")
    args = parser.parse_args()

    results = predict_video_id(args.video_id, args.checkpoint, args.top_k)
    print(f"Predictions for video_id={args.video_id}")
    for item in results:
        print(
            f"#{item['rank']} | class_id={item['class_id']} | "
            f"label={item['label']} | prob={item['probability']:.4f}"
        )


if __name__ == "__main__":
    main()
