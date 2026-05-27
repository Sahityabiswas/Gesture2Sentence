import argparse
import csv
import json
import os
import pickle

import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

import config
from hierarchical_inference import ensemble_predict, load_artifact_bundles, parse_artifact_dirs
from inference_utils import load_inference_settings
import normalize


WINDOW_SIZE = 3
SEQ2SEQ_MODEL_DIR = os.path.join(os.path.dirname(__file__), "wts_split", "best_model")


class SignToSentencePipeline:
    def __init__(self, device=None, seq2seq_model_dir=SEQ2SEQ_MODEL_DIR, artifact_dirs=None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        infer = load_inference_settings()
        self.artifact_dirs = parse_artifact_dirs(artifact_dirs)
        self.top_k_groups = infer["top_k_groups"]
        self.top_k_classes = infer["top_k_classes"]
        self.temperature = infer["temperature"]
        self.group_score_power = infer["group_score_power"]
        print(f"[Pipeline] Using device: {self.device}")

        print("[Pipeline] Loading sign-language artifacts...")
        with open(config.DATA_MAP_PATH, "rb") as f:
            self.data_map = pickle.load(f)
        with open(config.VID_CLASS_PATH, "rb") as f:
            self.vid_class = pickle.load(f)
        self.bundles = load_artifact_bundles(self.artifact_dirs, self.device)
        self.label_map = self.bundles[0]["label_map"]
        group_info = self.bundles[0]["group_info"]
        self.class_words = self._load_class_words(config.CLASS_MAP_PATH)

        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        self.group_of_class = group_info["group_of_class"]
        self.classes_in_group = group_info["classes_in_group"]
        self.num_groups = group_info["num_groups"]
        self.mean, self.std = normalize.load_stats(os.path.join(self.artifact_dirs[0], "global_stats.pkl"))
        print(f"[Pipeline] Loaded {len(self.bundles)} artifact bundle(s).")

        print(f"[Pipeline] Loading T5 generator from '{seq2seq_model_dir}'...")
        if not os.path.exists(seq2seq_model_dir):
            raise FileNotFoundError(
                f"'{seq2seq_model_dir}' not found. Train the sentence model in wts_split first."
            )

        self.tokenizer = T5TokenizerFast.from_pretrained(seq2seq_model_dir)
        self.seq2seq = T5ForConditionalGeneration.from_pretrained(seq2seq_model_dir)
        self.seq2seq.to(self.device)
        self.seq2seq.eval()

        cfg_path = os.path.join(seq2seq_model_dir, "training_config.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)

        self.input_prefix = cfg.get("input_prefix", "generate sentence: ")
        self.max_input_len = cfg.get("max_input_len", 64)
        self.max_target_len = cfg.get("max_target_len", 64)
        print("[Pipeline] All models ready.\n")

    def _load_class_words(self, csv_path):
        mapping = {}
        if not os.path.exists(csv_path):
            return mapping

        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_id = str(row.get("class", "")).strip()
                word = str(row.get("word", "")).strip()
                if class_id and word:
                    mapping[class_id] = word
        return mapping

    def _display_label(self, global_id):
        raw_label = str(self.inv_label_map.get(global_id, "?"))
        return self.class_words.get(raw_label, raw_label)

    def _preprocess(self, vid_id):
        seq = np.array(self.data_map[vid_id], dtype=np.float32).reshape(-1, self.mean.shape[0])
        seq = (seq - self.mean) / self.std
        sample = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        return sample, [seq.shape[0]]

    def predict_sign(self, vid_id):
        sample, length = self._preprocess(vid_id)
        output = ensemble_predict(
            self.bundles,
            sample,
            length,
            self.top_k_groups,
            self.top_k_classes,
            self.temperature,
            self.group_score_power,
        )

        if not output["sorted_scores"]:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "group_conf": 0.0,
                "class_conf": 0.0,
                "group_id": -1,
            }

        combined, global_id = output["sorted_scores"][0]
        gid = self.group_of_class.get(global_id, -1)
        label = self._display_label(global_id)

        return {
            "label": label,
            "confidence": round(combined, 4),
            "group_conf": 0.0,
            "class_conf": 0.0,
            "group_id": gid,
        }

    def _prepare_generator_input(self, keywords: str):
        text = self.input_prefix + keywords.strip().lower()
        enc = self.tokenizer(
            text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _postprocess(self, text: str):
        text = text.strip()
        if not text:
            return text
        text = text[0].upper() + text[1:]
        if text[-1] not in ".!?":
            text += "."
        return text

    def _enforce_keywords(self, sentence: str, keywords: str):
        missing = [k for k in keywords.lower().split() if k not in sentence.lower()]
        if not missing:
            return sentence
        base = sentence.rstrip(".!?")
        return base + ", particularly " + " and ".join(missing) + "."

    @torch.no_grad()
    def _beam(self, keywords):
        enc = self._prepare_generator_input(keywords)
        out = self.seq2seq.generate(
            **enc,
            max_length=self.max_target_len,
            num_beams=5,
            length_penalty=0.8,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return self._enforce_keywords(self._postprocess(text), keywords)

    @torch.no_grad()
    def _constrained_beam(self, keywords):
        enc = self._prepare_generator_input(keywords)
        kw_list = keywords.strip().lower().split()

        force_ids = []
        for kw in kw_list:
            ids = self.tokenizer(kw, add_special_tokens=False)["input_ids"]
            if ids:
                force_ids.append(ids)

        if not force_ids:
            return self._beam(keywords)

        try:
            out = self.seq2seq.generate(
                **enc,
                max_length=self.max_target_len,
                num_beams=10,
                force_words_ids=force_ids,
                no_repeat_ngram_size=2,
                length_penalty=1.0,
            )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            return self._postprocess(text)
        except Exception:
            return self._beam(keywords)

    @torch.no_grad()
    def _diverse_beam(self, keywords):
        enc = self._prepare_generator_input(keywords)
        try:
            out = self.seq2seq.generate(
                **enc,
                max_length=self.max_target_len,
                num_beams=10,
                num_beam_groups=5,
                diversity_penalty=0.8,
                num_return_sequences=5,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        except Exception:
            out = self.seq2seq.generate(
                **enc,
                max_length=self.max_target_len,
                num_beams=5,
                num_return_sequences=3,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        return [
            self._enforce_keywords(
                self._postprocess(self.tokenizer.decode(seq, skip_special_tokens=True)),
                keywords,
            )
            for seq in out
        ]

    @torch.no_grad()
    def _sample(self, keywords):
        enc = self._prepare_generator_input(keywords)
        out = self.seq2seq.generate(
            **enc,
            max_length=self.max_target_len,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.92,
            num_return_sequences=2,
            no_repeat_ngram_size=3,
        )
        return [
            self._enforce_keywords(
                self._postprocess(self.tokenizer.decode(seq, skip_special_tokens=True)),
                keywords,
            )
            for seq in out
        ]

    def run(self, vid_ids: list[str]) -> dict:
        if len(vid_ids) != WINDOW_SIZE:
            raise ValueError(f"Expected {WINDOW_SIZE} video IDs, got {len(vid_ids)}.")

        signs = []
        for vid in vid_ids:
            if vid not in self.data_map:
                raise KeyError(f"Video ID '{vid}' not found in data_map.")
            pred = self.predict_sign(vid)
            signs.append(pred)
            print(
                f"  [{vid}] -> '{pred['label']}' "
                f"(combined: {pred['confidence']:.1%} | "
                f"group: {pred['group_conf']:.1%} x class: {pred['class_conf']:.1%})"
            )

        keywords = " ".join(s["label"] for s in signs)
        avg_conf = round(sum(s["confidence"] for s in signs) / WINDOW_SIZE, 4)
        print(f"\n  Keywords : '{keywords}'")
        print(f"  Avg conf : {avg_conf:.1%}\n")

        diverse = self._diverse_beam(keywords)
        sampled = self._sample(keywords)
        sentences = {
            "greedy": self._beam(keywords),
            "beam": self._constrained_beam(keywords),
            "sample": sampled[0] if sampled else "",
            "diverse": diverse[0] if diverse else "",
        }

        return {
            "signs": signs,
            "keywords": keywords,
            "avg_conf": avg_conf,
            "sentences": sentences,
            "diverse_candidates": diverse,
            "sample_candidates": sampled,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign -> Sentence pipeline test")
    parser.add_argument("--vids", nargs=3, metavar="VID", help="Exactly 3 video IDs from the dataset")
    parser.add_argument("--artifact-dirs", type=str, default="")
    args = parser.parse_args()

    pipe = SignToSentencePipeline(artifact_dirs=args.artifact_dirs)

    if args.vids:
        vid_ids = args.vids
    else:
        with open(config.VID_SPLITS_PATH, "rb") as f:
            vid_splits = pickle.load(f)
        with open(config.VID_CLASS_PATH, "rb") as f:
            vid_class = pickle.load(f)

        allowed = set(pipe.label_map.keys())
        test_vids = [
            v for v in vid_splits["test"]
            if v in pipe.data_map and vid_class.get(v) in allowed
        ]
        if len(test_vids) < 3:
            raise RuntimeError("Not enough test videos found.")
        vid_ids = test_vids[:3]

    print("=" * 60)
    print(f"Input videos : {vid_ids}")
    print("=" * 60)

    result = pipe.run(vid_ids)

    print("-" * 60)
    print("RESULTS")
    print("-" * 60)
    print("Signs predicted :")
    for sign in result["signs"]:
        print(f"  {sign['label']:<20}  confidence: {sign['confidence']:.1%}")
    print(f"\nKeywords  : {result['keywords']}")
    print(f"Avg conf  : {result['avg_conf']:.1%}")
    print("\nSentences :")
    for name, sentence in result["sentences"].items():
        print(f"  {name.capitalize():<9}: {sentence}")
    print("=" * 60)
