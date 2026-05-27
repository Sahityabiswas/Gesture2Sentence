import pickle
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import config


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_raw_data():
    with open(config.DATA_MAP_PATH, "rb") as f:
        data_map = pickle.load(f)
    with open(config.VID_CLASS_PATH, "rb") as f:
        vid_class = pickle.load(f)
    with open(config.VID_SPLITS_PATH, "rb") as f:
        vid_splits = pickle.load(f)
    return data_map, vid_class, vid_splits


def build_label_map(vid_class, max_classes=config.MAX_CLASSES):
    unique = sorted(set(vid_class.values()))
    if max_classes is not None:
        unique = unique[:max_classes]
    allowed = set(unique)
    label_map = {label: i for i, label in enumerate(sorted(allowed))}
    return label_map, allowed


def _ensure_tvc(seq):
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        feature_dim = arr.shape[1]
        expected = config.NUM_KEYPOINTS * config.INPUT_CHANNELS
        if feature_dim != expected:
            raise ValueError(
                f"Expected flattened feature dim {expected}, got {feature_dim}."
            )
        return arr.reshape(arr.shape[0], config.NUM_KEYPOINTS, config.INPUT_CHANNELS)
    raise ValueError(f"Unexpected sequence shape: {arr.shape}")


def collect_split(split_key, data_map, vid_class, vid_splits, label_map, allowed):
    seqs, labels, lengths = [], [], []
    for vid in vid_splits[split_key]:
        if vid not in data_map or vid not in vid_class:
            continue
        label = vid_class[vid]
        if label not in allowed:
            continue
        seq = _ensure_tvc(data_map[vid])
        seqs.append(seq)
        labels.append(label_map[label])
        lengths.append(seq.shape[0])
    return seqs, labels, lengths


def fit_stats(sequences):
    all_frames = np.concatenate(sequences, axis=0)
    mean = all_frames.mean(axis=0)
    std = all_frames.std(axis=0) + 1e-5
    return mean, std


def apply_stats(sequences, mean, std):
    return [(seq - mean) / std for seq in sequences]


def save_stats(mean, std):
    with open(config.STATS_PATH, "wb") as f:
        pickle.dump({"mean": mean, "std": std}, f)


def load_stats():
    with open(config.STATS_PATH, "rb") as f:
        stats = pickle.load(f)
    return stats["mean"], stats["std"]


def augment_sequence(seq):
    x = seq.clone()

    if x.shape[0] > 4 and random.random() < config.FRAME_DROP_PROB:
        keep = torch.rand(x.shape[0]) > config.FRAME_DROP_PROB
        if int(keep.sum().item()) >= max(2, int(0.6 * x.shape[0])):
            x = x[keep]

    if config.COORD_NOISE_STD > 0:
        x = x + torch.randn_like(x) * config.COORD_NOISE_STD

    return x


class SkeletonGraphDataset(Dataset):
    def __init__(self, seqs, labels, augment=False):
        self.seqs = [torch.tensor(seq, dtype=torch.float32) for seq in seqs]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        if self.augment:
            seq = augment_sequence(seq)
        seq = seq.permute(2, 0, 1).contiguous()
        return seq, self.labels[idx], seq.shape[1]


def collate_graph(batch):
    seqs, labels, lengths = zip(*batch)
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    seqs = [seqs[i].permute(1, 0, 2) for i in order]
    seqs = pad_sequence(seqs, batch_first=True)
    seqs = seqs.permute(0, 2, 1, 3).contiguous()
    labels = torch.stack([labels[i] for i in order])
    lengths = [lengths[i] for i in order]
    return seqs, labels, lengths


def build_dataloaders():
    set_seed()
    data_map, vid_class, vid_splits = load_raw_data()
    label_map, allowed = build_label_map(vid_class)

    raw_train_seqs, raw_train_labels, _ = collect_split(
        "train", data_map, vid_class, vid_splits, label_map, allowed
    )
    raw_test_seqs, raw_test_labels, raw_test_lengths = collect_split(
        "test", data_map, vid_class, vid_splits, label_map, allowed
    )

    mean, std = fit_stats(raw_train_seqs)
    train_seqs = apply_stats(raw_train_seqs, mean, std)
    test_seqs = apply_stats(raw_test_seqs, mean, std)
    save_stats(mean, std)

    indices = list(range(len(train_seqs)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config.VAL_SPLIT,
        stratify=raw_train_labels,
        random_state=config.SEED,
    )

    def pick(items, idxs):
        return [items[i] for i in idxs]

    train_ds = SkeletonGraphDataset(
        pick(train_seqs, train_idx),
        pick(raw_train_labels, train_idx),
        augment=True,
    )
    val_ds = SkeletonGraphDataset(
        pick(train_seqs, val_idx),
        pick(raw_train_labels, val_idx),
        augment=False,
    )
    test_ds = SkeletonGraphDataset(
        test_seqs,
        raw_test_labels,
        augment=False,
    )

    loader_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=collate_graph,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    with open(config.LABEL_MAP_PATH, "wb") as f:
        pickle.dump(label_map, f)

    meta = {
        "num_classes": len(label_map),
        "num_keypoints": config.NUM_KEYPOINTS,
        "input_channels": config.INPUT_CHANNELS,
        "test_lengths": raw_test_lengths,
    }
    return train_loader, val_loader, test_loader, label_map, meta
