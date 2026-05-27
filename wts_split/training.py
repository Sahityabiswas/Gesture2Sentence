import collections
import json
import math
import os
import random

import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast, get_cosine_schedule_with_warmup

from dataset_word_gen import TRAIN_CSV, VAL_CSV

SEED = 42
MODEL_NAME = "t5-small"
MAX_INPUT_LEN = 64
MAX_TARGET_LEN = 64
BATCH_SIZE = 16
GRAD_ACCUM = 4
EPOCHS = 20
LR = 3e-4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 1e-2
LABEL_SMOOTH = 0.1
EARLY_STOP_PAT = 5
OUTPUT_DIR = "best_model"
INPUT_PREFIX = "generate sentence: "

torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"


class KeywordSentenceDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_in: int, max_tgt: int):
        df = pd.read_csv(csv_path).dropna(subset=["keywords", "sentence"])
        df = df[df["keywords"].str.strip().ne("") & df["sentence"].str.strip().ne("")]
        self.inputs = [INPUT_PREFIX + kw.strip() for kw in df["keywords"].astype(str).tolist()]
        self.targets = df["sentence"].astype(str).str.strip().tolist()
        self.tokenizer = tokenizer
        self.max_in = max_in
        self.max_tgt = max_tgt
        print(f"  Loaded {len(self.inputs):,} pairs from {csv_path}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src = self.tokenizer(
            self.inputs[idx],
            max_length=self.max_in,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt = self.tokenizer(
            self.targets[idx],
            max_length=self.max_tgt,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tgt["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels": labels,
        }


def _tok(text: str):
    return text.lower().split()


def bleu_n(hyp: str, ref: str, n: int) -> float:
    h, r = _tok(hyp), _tok(ref)
    if len(h) < n:
        return 0.0
    hc = collections.Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
    rc = collections.Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
    match = sum(min(c, rc[ng]) for ng, c in hc.items())
    total = max(len(h) - n + 1, 0)
    return match / total if total else 0.0


def bleu4(hyp: str, ref: str) -> float:
    scores = [bleu_n(hyp, ref, n) for n in range(1, 5)]
    if any(s == 0 for s in scores):
        return 0.0
    h, r = _tok(hyp), _tok(ref)
    bp = 1.0 if len(h) >= len(r) else math.exp(1 - len(r) / max(len(h), 1))
    log_avg = sum(math.log(s) for s in scores) / 4
    return bp * math.exp(log_avg)


def rouge_l(hyp: str, ref: str) -> float:
    h, r = _tok(hyp), _tok(ref)
    if not h or not r:
        return 0.0
    m, n = len(h), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if h[i - 1] == r[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    p, rc = lcs / m, lcs / n
    return 2 * p * rc / (p + rc) if p + rc > 0 else 0.0


def keyword_coverage(hyp: str, keywords: str) -> float:
    kws = keywords.lower().split()
    hyp_lower = hyp.lower()
    found = sum(1 for kw in kws if kw in hyp_lower)
    return found / max(len(kws), 1)


@torch.no_grad()
def evaluate(model, tokenizer, val_ds, n_samples: int = 300):
    model.eval()
    indices = random.sample(range(len(val_ds)), min(n_samples, len(val_ds)))

    b4_scores, rl_scores, kc_scores, exact = [], [], [], []
    for idx in indices:
        item = val_ds[idx]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attn_mask = item["attention_mask"].unsqueeze(0).to(device)

        label_ids = item["labels"].clone()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        ref = tokenizer.decode(label_ids, skip_special_tokens=True).strip()

        raw_input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        kw_only = raw_input.replace(INPUT_PREFIX, "").strip()

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_length=MAX_TARGET_LEN,
            num_beams=5,
            length_penalty=0.8,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        hyp = tokenizer.decode(out[0], skip_special_tokens=True).strip()

        b4_scores.append(bleu4(hyp, ref))
        rl_scores.append(rouge_l(hyp, ref))
        kc_scores.append(keyword_coverage(hyp, kw_only))
        exact.append(int(_tok(hyp) == _tok(ref)))

    n = len(b4_scores)
    return {
        "bleu4": sum(b4_scores) / n,
        "rouge_l": sum(rl_scores) / n,
        "kw_cover": sum(kc_scores) / n,
        "exact": sum(exact) / n,
    }


def train():
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print("AMP    : enabled (fp16)")
    else:
        print("CPU mode - training will be slow.")

    print(f"\nLoading {MODEL_NAME} tokenizer and model...")
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    for csv in [TRAIN_CSV, VAL_CSV]:
        if not os.path.exists(csv):
            raise FileNotFoundError(f"'{csv}' not found - run dataset.py first.")

    train_ds = KeywordSentenceDataset(TRAIN_CSV, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
    val_ds = KeywordSentenceDataset(VAL_CSV, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=USE_AMP)

    print("\nTraining config:")
    print(f"  Train batches/epoch : {len(train_loader)}")
    print(f"  Grad accumulation   : {GRAD_ACCUM}")
    print(f"  Effective batch     : {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Total steps         : {total_steps}")
    print(f"  Warmup steps        : {warmup_steps}")
    print(f"  Early stop patience : {EARLY_STOP_PAT} epochs")

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    print("\n" + "=" * 65)
    print("  TRAINING")
    print("=" * 65)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=USE_AMP):
                out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                logits = out.logits
                vocab_sz = logits.size(-1)
                flat_log = torch.log_softmax(logits.view(-1, vocab_sz), dim=-1)
                flat_lbl = labels.view(-1)
                ce_loss = out.loss

                smooth_loss = -flat_log.mean(dim=-1)
                valid_mask = flat_lbl != -100
                smooth_loss = smooth_loss[valid_mask].mean()
                loss = (1 - LABEL_SMOOTH) * ce_loss + LABEL_SMOOTH * smooth_loss
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            n_tok = (labels != -100).sum().item()
            total_loss += ce_loss.item() * n_tok
            total_tokens += n_tok

        train_loss = total_loss / max(total_tokens, 1)

        model.eval()
        val_loss_total, val_tokens = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                with autocast(enabled=USE_AMP):
                    out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                n_tok = (labels != -100).sum().item()
                val_loss_total += out.loss.item() * n_tok
                val_tokens += n_tok
        val_loss = val_loss_total / max(val_tokens, 1)

        metrics = evaluate(model, tokenizer, val_ds, n_samples=200)
        cur_lr = scheduler.get_last_lr()[0]

        print(
            f"\nEpoch {epoch:>3}/{EPOCHS} | Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | LR {cur_lr:.2e}"
        )
        print(
            f"          BLEU-4: {metrics['bleu4']:.4f} | "
            f"ROUGE-L: {metrics['rouge_l']:.4f} | "
            f"KW-Cover: {metrics['kw_cover']:.4f} | "
            f"Exact: {metrics['exact']:.4f}"
        )

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_name": MODEL_NAME,
                        "max_input_len": MAX_INPUT_LEN,
                        "max_target_len": MAX_TARGET_LEN,
                        "input_prefix": INPUT_PREFIX,
                        "best_val_loss": best_val_loss,
                        "best_epoch": epoch,
                    },
                    f,
                    indent=2,
                )
            print(f"  Saved best model -> {OUTPUT_DIR}/ (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{EARLY_STOP_PAT})")
            if epochs_no_improve >= EARLY_STOP_PAT:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    with open("training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 65)
    print("  Training complete.")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Model saved   : {OUTPUT_DIR}/")
    print("  History saved : training_history.json")
    print("=" * 65)


if __name__ == "__main__":
    train()
