# Train_hierarchical.py  —  improved for ~2000 classes
#
# Key changes vs previous version:
#   1. CosineAnnealingLR replaces ReduceLROnPlateau
#      → smooth LR decay avoids getting stuck in sharp minima
#   2. Linear warmup for first 5 epochs
#      → large LR (1e-3) needs warmup to avoid early instability
#   3. Top-3 validation accuracy tracked alongside top-1
#      → better signal for sub-models with many similar classes
#   4. Group model skip-check uses NUM_GROUPS from config
#      → safe to change NUM_GROUPS without stale checkpoints

import os
import time
import pickle
import hashlib
import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from collections import Counter

import config
import normalize
from dataset import load_raw_data, build_label_map, collect_split, SignDataset, collate_fn
from model import build_model, build_loss, compute_class_weights

torch.backends.cudnn.benchmark = True
torch.manual_seed(config.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
os.makedirs(config.SUBMODEL_DIR, exist_ok=True)
print("Artifact dir:", os.path.abspath(config.ARTIFACT_DIR))

WARMUP_EPOCHS = 5   # linear LR warmup period


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def make_loaders(seqs, labels, lengths):
    indices = list(range(len(seqs)))
    counts  = Counter(labels)
    can_stratify = all(v >= 2 for v in counts.values())

    try:
        tr_idx, va_idx = train_test_split(
            indices,
            test_size=config.VAL_SPLIT,
            stratify=labels if can_stratify else None,
            random_state=config.SEED,
        )
    except ValueError:
        tr_idx, va_idx = train_test_split(
            indices, test_size=config.VAL_SPLIT, random_state=config.SEED
        )

    def subset(idx):
        return (
            [seqs[i]    for i in idx],
            [labels[i]  for i in idx],
            [lengths[i] for i in idx],
        )

    tr_s, tr_l, tr_n = subset(tr_idx)
    va_s, va_l, va_n = subset(va_idx)

    mk = lambda s, l, n, sh: DataLoader(
        SignDataset(s, l, n, augment=sh),
        batch_size=config.BATCH_SIZE,
        shuffle=sh,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.NUM_WORKERS > 0,
    )
    return mk(tr_s, tr_l, tr_n, True), mk(va_s, va_l, va_n, False)


def train_one_model(train_loader, val_loader, input_size, num_classes,
                    train_labels=None, label="model"):
    model = build_model(input_size, num_classes, device)

    class_weights = None
    if train_labels is not None:
        class_weights = compute_class_weights(train_labels, num_classes, device)
    criterion = build_loss(class_weights=class_weights,
                           label_smoothing=config.LABEL_SMOOTH)

    optimizer = optim.Adam(model.parameters(), lr=config.LR,
                           weight_decay=config.WEIGHT_DECAY)

    # ── FIX: CosineAnnealingLR + linear warmup ────────────────────────────────
    # CosineAnnealingLR smoothly reduces LR from LR → ~0 over EPOCHS.
    # Warmup prevents instability in the first few epochs when LR is large.
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS - WARMUP_EPOCHS,
        eta_min=1e-6,
    )

    def get_warmup_factor(epoch):
        """Linear warmup: epoch 1 → LR/5, epoch 5 → LR, epoch 6+ → cosine."""
        if epoch <= WARMUP_EPOCHS:
            return epoch / WARMUP_EPOCHS
        return 1.0

    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_warmup_factor)

    best_acc         = 0.0
    best_state       = None
    patience_counter = 0
    t0               = time.time()
    scaler           = GradScaler(device="cuda")

    for epoch in range(1, config.EPOCHS + 1):

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        tr_correct = tr_total = 0
        for bX, by, bn in train_loader:
            bX = bX.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            with autocast(device_type="cuda"):
                logits = model(bX, bn)
                loss   = criterion(logits, by)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            _, pred    = torch.max(logits, 1)
            tr_correct += (pred == by).sum().item()
            tr_total   += by.size(0)

        # ── val (top-1 and top-3) ─────────────────────────────────────────────
        model.eval()
        va_top1 = va_top3 = va_total = 0
        with torch.no_grad():
            for bX, by, bn in val_loader:
                bX = bX.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                logits = model(bX, bn)

                # top-1
                _, pred = torch.max(logits, 1)
                va_top1 += (pred == by).sum().item()

                # top-3 (capped at num_classes if small group)
                k = min(3, logits.shape[1])
                top3_pred = torch.topk(logits, k, dim=1).indices
                for i, true_lbl in enumerate(by):
                    if true_lbl in top3_pred[i]:
                        va_top3 += 1

                va_total += by.size(0)

        tr_acc   = tr_correct / tr_total
        va_acc   = va_top1    / va_total
        va_top3a = va_top3    / va_total

        # Step LR schedulers
        if epoch <= WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        improved = va_acc > best_acc
        if improved:
            best_acc         = va_acc
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        elapsed = time.time() - t0
        per_ep  = elapsed / epoch
        eta     = per_ep * (config.EPOCHS - epoch)
        eta_str = f"{eta/60:.1f}m" if eta >= 60 else f"{eta:.0f}s"
        tag     = " ✓" if improved else ""

        print(f"  [{label}] ep {epoch:3d} | "
              f"train {tr_acc:.3f} | val {va_acc:.3f} (top3:{va_top3a:.3f}) | "
              f"lr {current_lr:.2e} | eta {eta_str}{tag}")

        if patience_counter >= config.PATIENCE:
            print(f"  Early stop at epoch {epoch}  ({elapsed/60:.1f} min total)")
            break

    print(f"  [{label}] best val acc: {best_acc:.4f}  "
          f"({(time.time()-t0)/60:.1f} min)\n")
    return best_state, best_acc


def build_feature_based_groups(train_seqs, train_labels, target_groups):
    """
    Group classes by feature similarity instead of raw class-id order.
    Each class prototype is the mean of its sequence-level pooled features.
    KMeans then clusters similar classes into the same group.
    """
    per_class_vectors = {}
    for seq, lbl in zip(train_seqs, train_labels):
        seq_vec = seq.mean(axis=0)
        per_class_vectors.setdefault(lbl, []).append(seq_vec)

    class_ids = sorted(per_class_vectors.keys())
    class_prototypes = np.stack(
        [np.mean(per_class_vectors[cid], axis=0) for cid in class_ids]
    ).astype(np.float32)

    num_groups = min(target_groups, len(class_ids))
    if num_groups <= 1:
        assignments = np.zeros(len(class_ids), dtype=np.int64)
    else:
        kmeans = KMeans(
            n_clusters=num_groups,
            n_init=20,
            random_state=config.SEED,
        )
        assignments = kmeans.fit_predict(class_prototypes)

    raw_groups = {}
    for class_id, cluster_id in zip(class_ids, assignments.tolist()):
        raw_groups.setdefault(int(cluster_id), []).append(int(class_id))

    classes_in_group = {}
    group_of_class = {}
    for new_gid, (_, grouped_class_ids) in enumerate(
        sorted(raw_groups.items(), key=lambda item: min(item[1]))
    ):
        sorted_ids = sorted(grouped_class_ids)
        classes_in_group[new_gid] = sorted_ids
        for class_id in sorted_ids:
            group_of_class[class_id] = new_gid

    group_sizes = [len(v) for v in classes_in_group.values()]
    return group_of_class, classes_in_group, group_sizes


def compute_group_signature(classes_in_group):
    normalized = [
        tuple(sorted(class_ids))
        for _, class_ids in sorted(classes_in_group.items())
    ]
    return hashlib.md5(repr(normalized).encode("utf-8")).hexdigest()


# ════════════════════════════════════════════════════════════════════════════
# LOAD & NORMALISE DATA
# ════════════════════════════════════════════════════════════════════════════
print("\n── Loading data ──")
data_map, vid_class, vid_splits = load_raw_data()
label_map, allowed = build_label_map(vid_class)
print(f"Total classes: {len(label_map)}")

raw_tr_seqs, raw_tr_labels, raw_tr_lengths = collect_split(
    "train", data_map, vid_class, vid_splits, label_map, allowed)
raw_te_seqs, raw_te_labels, raw_te_lengths = collect_split(
    "test",  data_map, vid_class, vid_splits, label_map, allowed)

tr_seqs, mean, std = normalize.fit_transform(raw_tr_seqs)
te_seqs            = normalize.transform(raw_te_seqs, mean, std)
normalize.save_stats(mean, std)

with open(config.LABEL_MAP_PATH, "wb") as f:
    pickle.dump(label_map, f)

input_size = tr_seqs[0].shape[1]
print(f"Input feature size: {input_size}")
print(f"Train samples: {len(tr_seqs)}  |  Test samples: {len(te_seqs)}")


# ════════════════════════════════════════════════════════════════════════════
# BUILD GROUP MAP
# ════════════════════════════════════════════════════════════════════════════
group_of_class, classes_in_group, group_sizes = build_feature_based_groups(
    tr_seqs, raw_tr_labels, config.NUM_GROUPS
)
num_groups = len(classes_in_group)
group_signature = compute_group_signature(classes_in_group)
avg_group_size = float(np.mean(group_sizes))
print(f"\nGroups: {num_groups}  |  avg {avg_group_size:.1f} classes per group")
print(f"Grouping strategy: feature-based class clustering")

with open(config.GROUP_MAP_PATH, "wb") as f:
    pickle.dump({
        "group_of_class":   group_of_class,
        "classes_in_group": classes_in_group,
        "group_sizes":      group_sizes,
        "group_strategy":   "feature_kmeans",
        "group_signature":  group_signature,
        "num_groups":       num_groups,
    }, f)
print(f"Saved group map → {config.GROUP_MAP_PATH}")


# ════════════════════════════════════════════════════════════════════════════
# STAGE 1 — TRAIN GROUP MODEL
# ════════════════════════════════════════════════════════════════════════════
# IMPORTANT: delete group_model.pt before retraining if grouping changed.
# The old checkpoint can have the same num_classes but still represent a stale
# class-to-group assignment, so we also validate the grouping signature.
if os.path.exists(config.GROUP_MODEL_PATH):
    ckpt_check = torch.load(config.GROUP_MODEL_PATH, map_location="cpu", weights_only=True)
    saved_num_classes = ckpt_check.get("num_classes", -1)
    saved_signature = ckpt_check.get("group_signature")
    if saved_num_classes != num_groups or saved_signature != group_signature:
        print(f"\n[WARNING] group_model.pt was saved with {saved_num_classes} groups "
              f"and signature {saved_signature}. Current setup uses {num_groups} groups "
              f"with signature {group_signature}. Deleting stale checkpoint and retraining.")
        os.remove(config.GROUP_MODEL_PATH)

if os.path.exists(config.GROUP_MODEL_PATH):
    print(f"\nGroup model already exists — skipping Stage 1.")
    print(f"Delete {config.GROUP_MODEL_PATH} to retrain it.")
else:
    print("\n══════════════════════════════════════")
    print("STAGE 1 — Group model (predicts group)")
    print("══════════════════════════════════════")

    group_labels_tr = [group_of_class[y] for y in raw_tr_labels]
    train_loader_g, val_loader_g = make_loaders(tr_seqs, group_labels_tr, raw_tr_lengths)

    best_state_g, _ = train_one_model(
        train_loader_g, val_loader_g,
        input_size, num_groups,
        train_labels=group_labels_tr,
        label="group-model",
    )

    torch.save({
        "model_state": best_state_g,
        "input_size":  input_size,
        "num_classes": num_groups,
        "group_signature": group_signature,
    }, config.GROUP_MODEL_PATH)
    print(f"Saved group model → {config.GROUP_MODEL_PATH}")


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 — TRAIN ONE SUB-MODEL PER GROUP
# ════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════")
print("STAGE 2 — Sub-models (one per group)")
print("══════════════════════════════════════\n")

sub_results  = {}
stage2_start = time.time()

for group_id, class_ids in sorted(classes_in_group.items()):

    save_path = os.path.join(config.SUBMODEL_DIR, f"group_{group_id}.pt")

    if os.path.exists(save_path):
        ckpt_check = torch.load(save_path, map_location="cpu", weights_only=True)
        saved_signature = ckpt_check.get("group_signature")
        saved_classes = sorted(ckpt_check.get("class_ids", []))
        if saved_signature == group_signature and saved_classes == sorted(class_ids):
            print(f"  Group {group_id:3d}: already trained → skipping")
            sub_results[group_id] = None
            continue
        print(f"  Group {group_id:3d}: stale checkpoint detected → retraining")
        os.remove(save_path)

    local_map = {cid: i for i, cid in enumerate(sorted(class_ids))}
    g_seqs, g_labels, g_lengths = [], [], []
    for seq, lbl, length in zip(tr_seqs, raw_tr_labels, raw_tr_lengths):
        if lbl in local_map:
            g_seqs.append(seq)
            g_labels.append(local_map[lbl])
            g_lengths.append(length)

    if len(g_seqs) < 4:
        print(f"  Group {group_id:3d}: skipping — only {len(g_seqs)} samples")
        continue

    done       = group_id
    elapsed_s2 = time.time() - stage2_start
    eta_s2     = (elapsed_s2 / max(done, 1)) * (num_groups - done)
    eta_str    = f"{eta_s2/3600:.1f}h" if eta_s2 >= 3600 else f"{eta_s2/60:.1f}m"

    print(f"  ── Group {group_id:3d}/{num_groups-1}  "
          f"{len(class_ids)} classes  "
          f"{len(g_seqs)} samples  "
          f"Stage-2 ETA: {eta_str}")

    train_loader_s, val_loader_s = make_loaders(g_seqs, g_labels, g_lengths)

    best_state_s, best_acc = train_one_model(
        train_loader_s, val_loader_s,
        input_size, len(local_map),
        train_labels=g_labels,
        label=f"group-{group_id}",
    )

    torch.save({
        "model_state":   best_state_s,
        "input_size":    input_size,
        "num_classes":   len(local_map),
        "local_map":     local_map,
        "inv_local_map": {i: cid for cid, i in local_map.items()},
        "class_ids":     sorted(class_ids),
        "group_signature": group_signature,
    }, save_path)

    sub_results[group_id] = best_acc
    print(f"  Saved → {save_path}\n")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n── Sub-model summary ──")
valid = {k: v for k, v in sub_results.items() if v is not None}
for gid in sorted(valid):
    print(f"  Group {gid:3d}: best val acc {valid[gid]:.4f}")

if valid:
    print(f"\nAverage sub-model val acc : {np.mean(list(valid.values())):.4f}")

print(f"Total Stage 2 time        : {(time.time()-stage2_start)/3600:.2f} hrs")
print("\nAll done!  Next step: python evaluate.py")
