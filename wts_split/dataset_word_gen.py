import random
import re
from collections import Counter

import nltk
import pandas as pd
from datasets import load_dataset
from nltk.corpus import stopwords


SEED = 42
TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
random.seed(SEED)


def ensure_nltk_resources():
    for resource, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("wordnet", "corpora/wordnet"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)


ensure_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))


def is_valid_pair(keywords: str, sentence: str) -> bool:
    kw_list = keywords.strip().lower().split()
    sent_lower = sentence.strip().lower()
    words = sent_lower.split()

    if len(kw_list) < 2:
        return False
    if not (5 <= len(words) <= 40):
        return False

    for kw in kw_list:
        if kw not in sent_lower:
            return False

    alpha_ratio = sum(c.isalpha() or c == " " for c in sentence) / max(len(sentence), 1)
    if alpha_ratio < 0.70:
        return False

    if "http" in sent_lower or "www." in sent_lower:
        return False

    return True


def normalise_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def augment_pair(keywords: str, sentence: str, n_shuffles: int = 2):
    words = keywords.split()
    result = [(keywords, sentence)]

    if len(words) >= 3:
        subset = " ".join(words[:-1])
        if all(k in sentence.lower() for k in subset.split()):
            result.append((subset, sentence))

    for _ in range(n_shuffles):
        shuffled = words.copy()
        random.shuffle(shuffled)
        result.append((" ".join(shuffled), sentence))

    return result


def load_commongen(augment: bool = True):
    print("\n[1/4] Loading CommonGen...")
    try:
        ds = load_dataset("allenai/common_gen", trust_remote_code=True)
    except Exception:
        ds = load_dataset("common_gen")

    pairs = []
    for split in ["train", "validation"]:
        if split not in ds:
            continue
        for row in ds[split]:
            concepts = row.get("concepts", [])
            target = row.get("target", "")
            if not concepts or not target:
                continue

            kw_str = " ".join(str(c).lower().strip() for c in concepts)
            sentence = target.strip()
            if is_valid_pair(kw_str, sentence):
                pairs.extend(augment_pair(kw_str, sentence) if augment else [(kw_str, sentence)])

    print(f"   CommonGen pairs (with augmentation): {len(pairs)}")
    return pairs


def extract_webnlg_keywords(triple_str: str) -> str:
    parts = [p.strip().lower() for p in triple_str.split("|")]
    keywords = []
    for part in parts:
        part = re.sub(r"_", " ", part)
        part = re.sub(r"\d+", "", part).strip()
        words = [w for w in part.split() if w.isalpha() and len(w) > 2 and w not in STOP_WORDS]
        keywords.extend(words[:2])

    seen = set()
    unique = []
    for word in keywords:
        if word not in seen:
            seen.add(word)
            unique.append(word)
    return " ".join(unique[:6])


def load_webnlg():
    print("\n[2/4] Loading WebNLG...")
    try:
        ds = load_dataset("web_nlg", "release_v3.0_en", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("web_nlg", "en")
        except Exception as exc:
            print(f"   WebNLG unavailable: {exc}")
            return []

    pairs = []
    for split in ["train", "dev"]:
        if split not in ds:
            continue
        for row in ds[split]:
            triples = row.get("modified_triple_sets", {}) or row.get("original_triple_sets", {})
            refs = row.get("lex", {})
            if not refs:
                continue

            ref_list = refs.get("text", []) if isinstance(refs, dict) else []
            if not ref_list and isinstance(refs, list):
                ref_list = [r.get("text", "") if isinstance(r, dict) else str(r) for r in refs]

            triple_list = []
            if isinstance(triples, dict):
                mtriples = triples.get("mtriple_set", [])
                if mtriples and isinstance(mtriples, list):
                    triple_list = mtriples[0] if isinstance(mtriples[0], list) else mtriples
            elif isinstance(triples, list) and triples:
                triple_list = triples[0] if isinstance(triples[0], list) else triples

            if not triple_list or not ref_list:
                continue

            all_keywords = []
            seen_kw = set()
            for triple in triple_list[:3]:
                kw = extract_webnlg_keywords(str(triple))
                for word in kw.split():
                    if word not in seen_kw:
                        seen_kw.add(word)
                        all_keywords.append(word)

            kw_str = " ".join(all_keywords[:8])
            for ref in ref_list[:2]:
                sentence = str(ref).strip()
                if is_valid_pair(kw_str, sentence):
                    pairs.append((kw_str, sentence))

    print(f"   WebNLG pairs: {len(pairs)}")
    return pairs


def load_e2e():
    print("\n[3/4] Loading E2E NLG...")
    try:
        ds = load_dataset("e2e_nlg", trust_remote_code=True)
    except Exception as exc:
        print(f"   E2E NLG unavailable: {exc}")
        return []

    pairs = []
    for split in ["train", "validation"]:
        if split not in ds:
            continue
        for row in ds[split]:
            meaning_rep = str(row.get("meaning_representation", ""))
            ref = str(row.get("human_reference", "")).strip()
            if not meaning_rep or not ref:
                continue

            slot_values = re.findall(r"\[([^\]]+)\]", meaning_rep)
            keywords = []
            seen_kw = set()
            for val in slot_values:
                val = val.strip().lower()
                if val in ("yes", "no", "true", "false"):
                    continue
                for word in val.split():
                    word = re.sub(r"[^a-z]", "", word)
                    if word and len(word) > 2 and word not in STOP_WORDS and word not in seen_kw:
                        seen_kw.add(word)
                        keywords.append(word)

            kw_str = " ".join(keywords[:6])
            if is_valid_pair(kw_str, ref):
                pairs.append((kw_str, ref))

    print(f"   E2E NLG pairs: {len(pairs)}")
    return pairs


def extract_content_keywords(sentence: str, max_kw: int = 5) -> str:
    try:
        words = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        seen = set()
        unique = []
        for word, pos in tagged:
            word_low = word.lower()
            if (
                pos.startswith("NN")
                and word_low not in STOP_WORDS
                and word.isalpha()
                and len(word) > 3
                and word_low not in seen
            ):
                seen.add(word_low)
                unique.append(word_low)
        return " ".join(unique[:max_kw])
    except Exception:
        return ""


def load_rocstories():
    print("\n[4/4] Loading ROCStories (strictly filtered)...")
    try:
        ds = load_dataset("Xuhui/ROCStories", trust_remote_code=True)
    except Exception as exc:
        print(f"   ROCStories unavailable: {exc}")
        return []

    col_names = ds["train"].column_names
    sent_cols = [c for c in col_names if "sent" in c.lower() or "sentence" in c.lower()]
    if not sent_cols:
        print("   Could not identify sentence columns.")
        return []

    pairs = []
    for row in ds["train"]:
        for col in sent_cols:
            sentence = str(row.get(col, "")).strip()
            if len(sentence.split()) < 6 or len(sentence.split()) > 30:
                continue
            kw = extract_content_keywords(sentence, max_kw=4)
            if not kw or len(kw.split()) < 2:
                continue
            if is_valid_pair(kw, sentence):
                pairs.append((kw, sentence))

    print(f"   ROCStories pairs (strict filter): {len(pairs)}")
    return pairs


def deduplicate(pairs):
    seen = set()
    unique = []
    for kw, sent in pairs:
        key = normalise_sentence(sent)
        if key not in seen:
            seen.add(key)
            unique.append((kw, sent))
    return unique


def build_and_save_datasets(train_csv: str = TRAIN_CSV, val_csv: str = VAL_CSV):
    print("=" * 60)
    print("  DATASET BUILDER")
    print("=" * 60)

    all_pairs = []
    source_counts = {}

    try:
        cg = load_commongen(augment=True)
        source_counts["CommonGen"] = len(cg)
        all_pairs.extend(cg)
    except Exception as exc:
        print(f"   CommonGen failed: {exc}")

    try:
        wn = load_webnlg()
        source_counts["WebNLG"] = len(wn)
        all_pairs.extend(wn)
    except Exception as exc:
        print(f"   WebNLG failed: {exc}")

    try:
        e2e = load_e2e()
        source_counts["E2E"] = len(e2e)
        all_pairs.extend(e2e)
    except Exception as exc:
        print(f"   E2E failed: {exc}")

    if len(all_pairs) < 100_000:
        try:
            roc = load_rocstories()
            source_counts["ROCStories"] = len(roc)
            all_pairs.extend(roc)
        except Exception as exc:
            print(f"   ROCStories failed: {exc}")

    if not all_pairs:
        raise RuntimeError("All data sources failed.")

    all_pairs = deduplicate(all_pairs)
    all_pairs = [(kw, sent) for kw, sent in all_pairs if is_valid_pair(kw, sent)]

    print(f"\n{'=' * 60}")
    print("  SOURCE BREAKDOWN")
    for src, cnt in source_counts.items():
        print(f"    {src:<15}: {cnt:>7,} pairs")
    print(f"  {'-' * 35}")
    print(f"  Total unique/valid : {len(all_pairs):>7,} pairs")
    print(f"{'=' * 60}")

    if not all_pairs:
        raise RuntimeError("0 pairs after filtering.")

    random.shuffle(all_pairs)
    val_size = max(500, int(len(all_pairs) * 0.1))
    train_size = len(all_pairs) - val_size

    train_pairs = all_pairs[:train_size]
    val_pairs = all_pairs[train_size:]

    df_train = pd.DataFrame(train_pairs, columns=["keywords", "sentence"])
    df_val = pd.DataFrame(val_pairs, columns=["keywords", "sentence"])
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)

    print(f"\n  Saved {train_csv} : {len(df_train):,} rows")
    print(f"  Saved {val_csv}   : {len(df_val):,} rows")

    for name, df in [("Train", df_train), ("Val", df_val)]:
        kw_lens = df["keywords"].str.split().str.len()
        sent_lens = df["sentence"].str.split().str.len()
        print(f"\n  {name} stats:")
        print(f"    Keyword  len - mean: {kw_lens.mean():.1f}  min: {kw_lens.min()}  max: {kw_lens.max()}")
        print(f"    Sentence len - mean: {sent_lens.mean():.1f}  min: {sent_lens.min()}  max: {sent_lens.max()}")

    print("\n  Sample pairs:")
    for kw, sent in train_pairs[:5]:
        print(f"    KW  : {kw}")
        print(f"    SENT: {sent}\n")

    print("Data preparation complete.")


if __name__ == "__main__":
    build_and_save_datasets()
