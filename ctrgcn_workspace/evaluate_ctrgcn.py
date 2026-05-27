import argparse
import pickle

import torch
from sklearn.metrics import classification_report

import config
from dataset_ctrgcn import build_dataloaders, set_seed
from model_ctrgcn import CTRGCNSignModel


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    preds = []
    targets = []

    for batch_x, batch_y, batch_len in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        logits = model(batch_x, batch_len)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(batch_y.tolist())

    return preds, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    args = parser.parse_args()

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, _, meta = build_dataloaders()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CTRGCNSignModel(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    with open(config.LABEL_MAP_PATH, "rb") as f:
        label_map = pickle.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}

    preds, targets = run_eval(model, test_loader, device)
    acc = sum(int(p == t) for p, t in zip(preds, targets)) / max(len(targets), 1)
    print(f"Test accuracy: {acc:.4f}")

    show = list(range(min(20, meta["num_classes"])))
    names = [str(inv_label_map[i]) for i in show]
    print(classification_report(targets, preds, labels=show, target_names=names, zero_division=0))


if __name__ == "__main__":
    main()
