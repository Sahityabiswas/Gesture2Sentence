import argparse
import json
import os

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast


MODEL_DIR = "best_model"


def load_model(model_dir: str = MODEL_DIR):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"'{model_dir}/' not found - run training.py first.")

    print(f"Loading model from {model_dir}/ ...")
    tokenizer = T5TokenizerFast.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    cfg_path = os.path.join(model_dir, "training_config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    return model, tokenizer, device, cfg


MODEL, TOKENIZER, DEVICE, CFG = load_model()
INPUT_PREFIX = CFG.get("input_prefix", "generate sentence: ")
MAX_INPUT_LEN = CFG.get("max_input_len", 64)
MAX_TARGET_LEN = CFG.get("max_target_len", 64)

print("Model loaded")
print(f"Device: {DEVICE}\n")


def prepare_input(keywords: str):
    text = INPUT_PREFIX + keywords.strip().lower()
    enc = TOKENIZER(
        text,
        max_length=MAX_INPUT_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.to(DEVICE) for k, v in enc.items()}


def postprocess(text: str):
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def enforce_keywords(sentence: str, keywords: str):
    kws = keywords.lower().split()
    missing = [k for k in kws if k not in sentence.lower()]
    if not missing:
        return sentence
    base = sentence.rstrip(".!?")
    return base + ", particularly " + " and ".join(missing) + "."


@torch.no_grad()
def beam_decode(keywords):
    enc = prepare_input(keywords)
    out = MODEL.generate(
        **enc,
        max_length=MAX_TARGET_LEN,
        num_beams=5,
        length_penalty=0.8,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    text = TOKENIZER.decode(out[0], skip_special_tokens=True)
    return enforce_keywords(postprocess(text), keywords)


@torch.no_grad()
def diverse_beam_decode(keywords):
    enc = prepare_input(keywords)
    try:
        out = MODEL.generate(
            **enc,
            max_length=MAX_TARGET_LEN,
            num_beams=10,
            num_beam_groups=5,
            diversity_penalty=0.8,
            num_return_sequences=5,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    except Exception:
        # Fall back to regular beam variants if grouped beam search is unsupported
        out = MODEL.generate(
            **enc,
            max_length=MAX_TARGET_LEN,
            num_beams=5,
            num_return_sequences=3,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return [
        enforce_keywords(postprocess(TOKENIZER.decode(seq, skip_special_tokens=True)), keywords)
        for seq in out
    ]


@torch.no_grad()
def sample_decode(keywords):
    enc = prepare_input(keywords)
    out = MODEL.generate(
        **enc,
        max_length=MAX_TARGET_LEN,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.92,
        num_return_sequences=2,
        no_repeat_ngram_size=3,
    )
    return [
        enforce_keywords(postprocess(TOKENIZER.decode(seq, skip_special_tokens=True)), keywords)
        for seq in out
    ]


@torch.no_grad()
def constrained_beam_decode(keywords):
    enc = prepare_input(keywords)
    kw_list = keywords.strip().lower().split()

    force_ids = []
    for kw in kw_list:
        ids = TOKENIZER(kw, add_special_tokens=False)["input_ids"]
        if ids:
            force_ids.append(ids)

    if not force_ids:
        return beam_decode(keywords)

    try:
        out = MODEL.generate(
            **enc,
            max_length=MAX_TARGET_LEN,
            num_beams=10,
            force_words_ids=force_ids,
            no_repeat_ngram_size=2,
            length_penalty=1.0,
        )
        return postprocess(TOKENIZER.decode(out[0], skip_special_tokens=True))
    except Exception:
        return beam_decode(keywords)


def run(keywords):
    print("\n" + "=" * 60)
    print(f"Keywords: {keywords}\n")

    print("Beam:")
    print(beam_decode(keywords), "\n")

    print("Constrained Beam:")
    print(constrained_beam_decode(keywords), "\n")

    print("Diverse Beam:")
    for sentence in diverse_beam_decode(keywords):
        print("-", sentence)

    print("\nSampling:")
    for sentence in sample_decode(keywords):
        print("-", sentence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", type=str, default=None, help="Space-separated keywords to generate from.")
    args = parser.parse_args()

    if args.keywords:
        run(args.keywords)
        return

    print("INTERACTIVE MODE (type 'quit')")
    while True:
        try:
            kw = input("\nEnter keywords: ").strip()
        except EOFError:
            break

        if kw.lower() in ("quit", "exit"):
            break
        if not kw:
            continue
        run(kw)


if __name__ == "__main__":
    main()
