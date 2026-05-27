# dataset.py  —  PyTorch Dataset and DataLoader helpers

import numpy as np
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import config
import normalize


# ── Raw data loading ──────────────────────────────────────────────────────────

def load_raw_data():
    """Load all pkl/csv files and return the three core dicts."""
    with open(config.DATA_MAP_PATH,  "rb") as f:
        data_map  = pickle.load(f)
    with open(config.VID_CLASS_PATH, "rb") as f:
        vid_class = pickle.load(f)
    with open(config.VID_SPLITS_PATH,"rb") as f:
        vid_splits = pickle.load(f)
    return data_map, vid_class, vid_splits


def build_label_map(vid_class, max_classes=config.MAX_CLASSES):
    """Map raw string/int labels → contiguous 0-based indices."""
    unique = sorted(set(vid_class.values()))
    if max_classes is not None:
        unique = unique[:max_classes]
    allowed   = set(unique)
    label_map = {label: i for i, label in enumerate(sorted(allowed))}
    return label_map, allowed


def collect_split(split_key, data_map, vid_class, vid_splits, label_map, allowed, return_vids=False):
    """Extract sequences, integer labels, lengths, and optionally video ids for one split."""
    seqs, labels, lengths, vids = [], [], [], []
    for vid in vid_splits[split_key]:
        if vid not in data_map or vid not in vid_class:
            continue
        label = vid_class[vid]
        if label not in allowed:
            continue
        seq = np.array(data_map[vid], dtype=np.float32)
        seq = seq.reshape(seq.shape[0], -1)        # (T, F)
        seqs.append(seq)
        labels.append(label_map[label])
        lengths.append(len(seq))
        vids.append(vid)
    if return_vids:
        return seqs, labels, lengths, vids
    return seqs, labels, lengths


# ── Dataset ───────────────────────────────────────────────────────────────────

def augment_sequence(seq):
    """
    Lightweight augmentation for keypoint sequences.
    Applied only during training to improve robustness.
    """
    x = seq.clone()
    length = x.shape[0]

    if length > 4 and np.random.rand() < config.AUG_FRAME_DROP_PROB:
        keep_mask = torch.rand(length) > config.AUG_FRAME_DROP_PROB
        if keep_mask.sum().item() >= max(2, int(0.6 * length)):
            x = x[keep_mask]
            length = x.shape[0]

    if length > 4 and np.random.rand() < config.AUG_TIME_MASK_PROB:
        mask_len = max(1, int(length * config.AUG_TIME_MASK_RATIO))
        mask_len = min(mask_len, length - 1)
        start = np.random.randint(0, max(1, length - mask_len + 1))
        x[start:start + mask_len] = 0.0

    if config.AUG_SCALE_JITTER > 0:
        scale = 1.0 + np.random.uniform(-config.AUG_SCALE_JITTER, config.AUG_SCALE_JITTER)
        x = x * scale

    if config.AUG_SHIFT_STD > 0:
        shift = torch.randn(1, x.shape[1], dtype=x.dtype) * config.AUG_SHIFT_STD
        x = x + shift

    if config.AUG_NOISE_STD > 0:
        x = x + torch.randn_like(x) * config.AUG_NOISE_STD

    return x


class SignDataset(Dataset):
    def __init__(self, seqs, labels, lengths, augment=False):
        """
        seqs    : list of numpy arrays (T_i, F), already normalised
        labels  : list of int
        lengths : list of int
        """
        self.seqs    = [torch.tensor(s, dtype=torch.float32) for s in seqs]
        self.labels  = torch.tensor(labels, dtype=torch.long)
        self.lengths = lengths
        self.augment = augment

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        if self.augment and getattr(config, "AUGMENT_TRAIN", False):
            seq = augment_sequence(seq)
        return seq, self.labels[idx], seq.shape[0]


def collate_fn(batch):
    """Pad sequences and sort by descending length (required for pack_padded_sequence)."""
    seqs, labels, lengths = zip(*batch)
    order   = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    seqs    = pad_sequence([seqs[i]   for i in order], batch_first=True)
    labels  = torch.stack ([labels[i] for i in order])
    lengths = [lengths[i] for i in order]
    return seqs, labels, lengths


# ── Public factory ─────────────────────────────────────────────────────────────

def build_loaders():
    """
    Full pipeline: load → filter → normalise → split → return DataLoaders.
    Also saves global_stats.pkl and label_map.pkl as side effects.
    """
    print("Loading raw data...")
    data_map, vid_class, vid_splits = load_raw_data()

    label_map, allowed = build_label_map(vid_class)
    print(f"Using {len(label_map)} classes.")

    # Collect raw sequences per split
    raw_tr_seqs, raw_tr_labels, raw_tr_lengths = collect_split(
        "train", data_map, vid_class, vid_splits, label_map, allowed)
    raw_te_seqs, raw_te_labels, raw_te_lengths = collect_split(
        "test",  data_map, vid_class, vid_splits, label_map, allowed)

    # Global normalisation — fit on train only
    tr_seqs_norm, mean, std = normalize.fit_transform(raw_tr_seqs)
    te_seqs_norm             = normalize.transform(raw_te_seqs, mean, std)
    normalize.save_stats(mean, std)

    # Train / validation split (stratified)
    indices = list(range(len(tr_seqs_norm)))
    tr_idx, va_idx = train_test_split(
        indices,
        test_size=config.VAL_SPLIT,
        stratify=raw_tr_labels,
        random_state=config.SEED,
    )

    def subset(seqs, labels, lengths, idx):
        return ([seqs[i] for i in idx],
                [labels[i] for i in idx],
                [lengths[i] for i in idx])

    tr_s, tr_l, tr_n = subset(tr_seqs_norm, raw_tr_labels, raw_tr_lengths, tr_idx)
    va_s, va_l, va_n = subset(tr_seqs_norm, raw_tr_labels, raw_tr_lengths, va_idx)

    print(f"Train: {len(tr_s)}  Val: {len(va_s)}  Test: {len(te_seqs_norm)}")

    # Save label map for inference
    with open(config.LABEL_MAP_PATH, "wb") as f:
        pickle.dump(label_map, f)

    make_loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=config.BATCH_SIZE,
        shuffle=shuffle, collate_fn=collate_fn, num_workers=0
    )

    train_loader = make_loader(SignDataset(tr_s, tr_l, tr_n, augment=True), shuffle=True)
    val_loader   = make_loader(SignDataset(va_s, va_l, va_n, augment=False), shuffle=False)
    test_loader  = make_loader(
        SignDataset(te_seqs_norm, raw_te_labels, raw_te_lengths, augment=False),
        shuffle=False,
    )

    # Expose input feature size and num classes for model construction
    input_size  = tr_seqs_norm[0].shape[1]
    num_classes = len(label_map)

    return train_loader, val_loader, test_loader, input_size, num_classes, label_map
