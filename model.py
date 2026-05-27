# model.py  —  Upgraded for ~2000 classes on medium dataset
#
# Improvements over previous version:
#   1. Dual architecture: Transformer encoder OR BiLSTM+Residual (config.USE_TRANSFORMER)
#   2. Residual connections in BiLSTM layers
#   3. Label smoothing built into build_loss()
#   4. Class-weighted loss via compute_class_weights()
#   5. Knowledge distillation via distillation_loss()
#   6. Classifier head auto-detects BatchNorm vs LayerNorm from checkpoint
#      (handles mixed checkpoints saved under different versions)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config


# ── Loss Functions ─────────────────────────────────────────────────────────────

def build_loss(class_weights=None, label_smoothing=config.LABEL_SMOOTH):
    """
    CrossEntropyLoss with optional class weights and label smoothing.

    Args:
        class_weights   : FloatTensor (num_classes,) from compute_class_weights()
        label_smoothing : float, default from config.LABEL_SMOOTH

    Usage in train_hierarchical.py:
        criterion = build_loss(class_weights=weights)
        loss = criterion(logits, labels)
    """
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


def compute_class_weights(labels, num_classes, device):
    """
    Inverse-frequency class weights to handle class imbalance.

    Args:
        labels      : list of int class indices (training labels for this group/stage)
        num_classes : total number of classes in this model
        device      : torch device

    Returns:
        weights : FloatTensor of shape (num_classes,)

    Usage in train_hierarchical.py:
        weights   = compute_class_weights(g_labels, len(local_map), device)
        criterion = build_loss(class_weights=weights)
    """
    counts = torch.zeros(num_classes)
    for lbl in labels:
        counts[int(lbl)] += 1
    counts  = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes   # normalise
    return weights.to(device)


# ── Knowledge Distillation Loss ────────────────────────────────────────────────

def distillation_loss(student_logits, teacher_logits, true_labels,
                      temperature=4.0, alpha=0.7,
                      label_smoothing=config.LABEL_SMOOTH):
    """
    Combined hard-label CE + soft-label KL distillation loss.
    Loss = alpha * KL(soft) + (1-alpha) * CE(hard)

    Args:
        student_logits : (B, C) from student model
        teacher_logits : (B, C) from teacher model (detach before passing)
        true_labels    : (B,)  ground-truth class indices
        temperature    : higher → softer teacher distribution
        alpha          : weight for the soft (KL) term

    Usage:
        with torch.no_grad():
            t_logits = teacher(bX, bn)
        s_logits = student(bX, bn)
        loss = distillation_loss(s_logits, t_logits, by)
    """
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    kl_loss = F.kl_div(soft_student, soft_targets, reduction="batchmean") * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits, true_labels, label_smoothing=label_smoothing)
    return alpha * kl_loss + (1 - alpha) * ce_loss


# ── Masked Attention ───────────────────────────────────────────────────────────

class MaskedAttention(nn.Module):
    """Attention pooling that ignores padding positions."""

    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, output, lengths):
        B, T, H = output.shape
        mask = (
            torch.arange(T, device=output.device).unsqueeze(0)
            >= torch.tensor(lengths, device=output.device).unsqueeze(1)
        )
        scores  = self.score(output).squeeze(-1)
        scores  = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = (weights.unsqueeze(-1) * output).sum(dim=1)
        return context


# ── Residual BiLSTM Block ──────────────────────────────────────────────────────

class ResidualBiLSTMBlock(nn.Module):
    """
    Single BiLSTM layer with skip connection + LayerNorm.
    Prevents vanishing gradients across stacked layers.
    Input/output shape: (B, T, hidden_size*2)
    """

    def __init__(self, input_size, hidden_size, dropout=0.3):
        super().__init__()
        lstm_out = hidden_size * 2
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.norm    = nn.LayerNorm(lstm_out)
        self.dropout = nn.Dropout(dropout)
        self.proj    = nn.Linear(input_size, lstm_out) if input_size != lstm_out else nn.Identity()

    def forward(self, x, lengths):
        packed        = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        packed_out, _ = self.lstm(packed)
        out, _        = pad_packed_sequence(packed_out, batch_first=True)
        out           = self.dropout(out)
        residual      = self.proj(x)[:, :out.size(1), :]   # align time axis
        return self.norm(out + residual)


# ── Shared Classifier Head ─────────────────────────────────────────────────────

def _build_classifier(in_size, mid_size, num_classes, dropout, use_batchnorm=False):
    """
    3-layer head: in → mid*2 → mid → num_classes.

    use_batchnorm=True  → BatchNorm1d (legacy checkpoints trained with old code)
    use_batchnorm=False → LayerNorm   (current default for new training runs)

    The correct value is detected automatically from the checkpoint by
    _ckpt_uses_batchnorm() and passed through build_model(), so callers
    never need to set this manually.
    """
    norm1 = nn.BatchNorm1d(mid_size * 2) if use_batchnorm else nn.LayerNorm(mid_size * 2)
    norm2 = nn.BatchNorm1d(mid_size)     if use_batchnorm else nn.LayerNorm(mid_size)
    return nn.Sequential(
        nn.Linear(in_size,      mid_size * 2),
        norm1,
        nn.ReLU(),
        nn.Dropout(dropout),

        nn.Linear(mid_size * 2, mid_size),
        norm2,
        nn.ReLU(),
        nn.Dropout(dropout / 2),

        nn.Linear(mid_size,     num_classes),
    )


# ── BiLSTM Model ───────────────────────────────────────────────────────────────

class BiLSTMSignModel(nn.Module):
    """Stacked residual BiLSTM blocks with masked attention pooling."""

    def __init__(self, input_size, num_classes,
                 hidden_size=config.HIDDEN_SIZE,
                 num_layers=config.NUM_LAYERS,
                 dropout=config.DROPOUT,
                 use_batchnorm=False):
        super().__init__()
        lstm_out = hidden_size * 2

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, lstm_out),
            nn.LayerNorm(lstm_out),
            nn.ReLU(),
        )

        self.lstm_blocks = nn.ModuleList([
            ResidualBiLSTMBlock(lstm_out, hidden_size, dropout)
            for _ in range(num_layers)
        ])

        self.attention  = MaskedAttention(lstm_out)
        self.classifier = _build_classifier(
            lstm_out, hidden_size, num_classes, dropout,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, lengths):
        x = self.input_proj(x)
        for block in self.lstm_blocks:
            x = block(x, lengths)
        context = self.attention(x, lengths)
        return self.classifier(context)


# ── Transformer Model ──────────────────────────────────────────────────────────

class TransformerSignModel(nn.Module):
    """
    Transformer encoder with mean pooling over valid frames.
    Advantages: fully parallel on GPU, better long-range dependencies,
    scales well to 2000 classes.
    """

    def __init__(self, input_size, num_classes,
                 hidden_size=config.HIDDEN_SIZE,
                 num_layers=config.NUM_LAYERS,
                 dropout=config.DROPOUT,
                 nhead=8,
                 use_batchnorm=False):
        super().__init__()

        if hidden_size % nhead != 0:
            nhead = 4   # safe fallback

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier  = _build_classifier(
            hidden_size, hidden_size // 2, num_classes, dropout,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, lengths):
        x = self.pos_dropout(self.input_proj(x))

        B, T, _ = x.shape
        pad_mask = (
            torch.arange(T, device=x.device).unsqueeze(0)
            >= torch.tensor(lengths, device=x.device).unsqueeze(1)
        )   # True = padding position

        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # Mean pool over valid (non-padding) frames only
        valid     = ~pad_mask                                          # (B, T)
        lengths_t = valid.sum(dim=1, keepdim=True).float()            # (B, 1)
        context   = (x * valid.unsqueeze(-1)).sum(dim=1) / lengths_t  # (B, H)

        return self.classifier(context)


# ── Public Factory ─────────────────────────────────────────────────────────────

def _ckpt_uses_batchnorm(state_dict):
    """
    Detect whether a checkpoint was saved with BatchNorm or LayerNorm by
    checking for BatchNorm-specific buffer keys (running_mean / running_var).
    These keys are present in BatchNorm but not in LayerNorm.
    """
    return any("running_mean" in k for k in state_dict.keys())


def build_model(input_size, num_classes, device, state_dict=None):
    """
    Builds the model selected by config.USE_TRANSFORMER.

    Args:
        input_size  : number of input features per frame
        num_classes : number of output classes for this model
        device      : torch.device
        state_dict  : (optional) the checkpoint's model_state dict.
                      When provided, the norm type (BatchNorm vs LayerNorm)
                      is detected automatically so the architecture always
                      matches the saved weights — handles mixed checkpoints
                      produced by different training runs.

    Usage (evaluate / predict):
        ckpt  = torch.load(path, map_location=device, weights_only=True)
        model = build_model(ckpt["input_size"], ckpt["num_classes"], device,
                            state_dict=ckpt["model_state"])
        model.load_state_dict(ckpt["model_state"])

    Usage (training — no checkpoint yet):
        model = build_model(input_size, num_classes, device)
    """
    use_transformer = getattr(config, "USE_TRANSFORMER", True)
    use_batchnorm   = _ckpt_uses_batchnorm(state_dict) if state_dict is not None else False

    if use_transformer:
        model = TransformerSignModel(
            input_size=input_size,
            num_classes=num_classes,
            use_batchnorm=use_batchnorm,
        )
        arch = "Transformer"
    else:
        model = BiLSTMSignModel(
            input_size=input_size,
            num_classes=num_classes,
            use_batchnorm=use_batchnorm,
        )
        arch = "BiLSTM+Residual"

    model    = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    norm_tag = "BatchNorm" if use_batchnorm else "LayerNorm"
    print(f"Model built [{arch}|{norm_tag}] — {n_params:,} parameters  "
          f"(input={input_size}, classes={num_classes}, "
          f"hidden={config.HIDDEN_SIZE}, layers={config.NUM_LAYERS})")
    return model