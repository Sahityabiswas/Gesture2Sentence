import json
import os
import time

import torch
from torch import nn

import config
from dataset_ctrgcn import build_dataloaders, set_seed
from model_ctrgcn import CTRGCNSignModel


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y, batch_len in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x, batch_len)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() * batch_y.size(0)
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total_samples += batch_y.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_top5 = 0
    total_samples = 0

    for batch_x, batch_y, batch_len in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        logits = model(batch_x, batch_len)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_y.size(0)
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        k = min(5, logits.shape[1])
        topk = torch.topk(logits, k=k, dim=1).indices
        for i, true_label in enumerate(batch_y):
            total_top5 += int(true_label in topk[i])

        total_samples += batch_y.size(0)

    return {
        "loss": total_loss / total_samples,
        "top1": total_correct / total_samples,
        "top5": total_top5 / total_samples,
    }


def main():
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, label_map, meta = build_dataloaders()
    print(f"Classes: {meta['num_classes']}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = CTRGCNSignModel(num_classes=meta["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    history = []
    best_val_top1 = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_acc,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        improved = val_metrics["top1"] > best_val_top1
        if improved:
            best_val_top1 = val_metrics["top1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": meta["num_classes"],
                    "label_map": label_map,
                    "config": {
                        "num_keypoints": config.NUM_KEYPOINTS,
                        "input_channels": config.INPUT_CHANNELS,
                        "base_channels": config.BASE_CHANNELS,
                    },
                },
                config.BEST_MODEL_PATH,
            )
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss {train_loss:.4f} | train_top1 {train_acc:.4f} | "
            f"val_loss {val_metrics['loss']:.4f} | val_top1 {val_metrics['top1']:.4f} | "
            f"val_top5 {val_metrics['top5']:.4f}"
        )

        if patience_counter >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    torch.save(
        {
            "model_state": model.state_dict(),
            "num_classes": meta["num_classes"],
            "label_map": label_map,
        },
        config.LAST_MODEL_PATH,
    )

    test_metrics = evaluate(model, test_loader, criterion, device)
    summary = {
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "test_loss": test_metrics["loss"],
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "elapsed_seconds": time.time() - start_time,
        "history": history,
    }
    with open(config.HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining finished.")
    print(f"Best epoch   : {best_epoch}")
    print(f"Best val top1: {best_val_top1:.4f}")
    print(f"Test top1    : {test_metrics['top1']:.4f}")
    print(f"Test top5    : {test_metrics['top5']:.4f}")
    print(f"Best model   : {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
