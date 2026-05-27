# demo.py  —  Interactive CLI demo for Sign → Sentence pipeline
#
# Modes:
#   1. Auto mode   : picks 3 random test videos and runs the pipeline
#   2. Manual mode : you type 3 video IDs yourself
#   3. Batch mode  : runs multiple windows of 3 from the test split
#
# Run:
#   python demo.py                  # auto mode (3 random test videos)
#   python demo.py --mode manual    # enter video IDs interactively
#   python demo.py --mode batch --n 5   # 5 windows of 3 test videos

import argparse
import pickle
import random
import textwrap

import config
from pipeline import SignToSentencePipeline

SEPARATOR = "═" * 64


def print_result(result: dict, window_num: int = 1):
    print(f"\n{SEPARATOR}")
    print(f"  WINDOW {window_num}")
    print(SEPARATOR)

    # ── Sign predictions ──────────────────────────────────────────────────────
    print("  Sign Predictions:")
    print(f"  {'Video':<24}  {'Label':<22}  {'Confidence':>10}")
    print("  " + "─" * 58)
    for pred in result["signs"]:
        label = pred["label"]
        conf  = pred["confidence"]
        bar   = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
        print(f"  {label:<46}  {conf:>6.1%}  {bar}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(f"  Keywords  : {result['keywords']}")
    print(f"  Avg conf  : {result['avg_conf']:.1%}")

    # ── Generated sentences ───────────────────────────────────────────────────
    print()
    print("  Generated Sentences:")
    print("  " + "─" * 58)
    for mode, sentence in result["sentences"].items():
        label    = f"  {mode.capitalize():<8}:"
        wrapped  = textwrap.fill(
            sentence if sentence else "(empty)",
            width=54,
            subsequent_indent=" " * 12,
        )
        print(f"{label}  {wrapped}")
    print(SEPARATOR)


def auto_mode(pipe, n_windows=1):
    """Pick n_windows × 3 random test videos and run the pipeline."""
    with open(config.VID_SPLITS_PATH, "rb") as f:
        vid_splits = pickle.load(f)
    with open(config.VID_CLASS_PATH, "rb") as f:
        vid_class = pickle.load(f)

    allowed = set(pipe.label_map.keys())
    pool = [
        v for v in vid_splits["test"]
        if v in pipe.data_map and vid_class.get(v) in allowed
    ]

    if len(pool) < 3 * n_windows:
        print(f"[Warning] Only {len(pool)} valid test videos — "
              f"reducing to {len(pool) // 3} window(s).")
        n_windows = len(pool) // 3

    random.shuffle(pool)
    all_results = []
    for i in range(n_windows):
        window = pool[i * 3: i * 3 + 3]
        print(f"\nProcessing window {i + 1}/{n_windows}: {window}")
        result = pipe.run(window)
        print_result(result, window_num=i + 1)
        all_results.append(result)

    return all_results


def manual_mode(pipe):
    """Let the user type video IDs one by one."""
    print("\nManual mode — enter 3 video IDs.")
    print("(Type 'quit' at any point to exit)\n")

    while True:
        vid_ids = []
        for i in range(1, 4):
            while True:
                vid = input(f"  Video {i}/3 ID: ").strip()
                if vid.lower() == "quit":
                    print("Exiting.")
                    return
                if vid in pipe.data_map:
                    vid_ids.append(vid)
                    break
                else:
                    print(f"    [!] '{vid}' not found in data_map. Try again.")

        result = pipe.run(vid_ids)
        print_result(result)

        again = input("\nRun another? [y/N]: ").strip().lower()
        if again != "y":
            break


def batch_mode(pipe, n_windows=5):
    """Run n_windows non-overlapping windows from the test split in sequence."""
    print(f"\nBatch mode — running {n_windows} windows of 3 test videos.\n")
    results = auto_mode(pipe, n_windows=n_windows)

    # ── Batch summary ─────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  BATCH SUMMARY")
    print(SEPARATOR)
    print(f"  {'Window':<8}  {'Keywords':<36}  {'Avg Conf':>8}")
    print("  " + "─" * 58)
    for i, r in enumerate(results, 1):
        kw   = r["keywords"][:34]
        conf = r["avg_conf"]
        print(f"  {i:<8}  {kw:<36}  {conf:>7.1%}")
    overall = sum(r["avg_conf"] for r in results) / len(results)
    print("  " + "─" * 58)
    print(f"  {'Overall avg conf':<44}  {overall:>7.1%}")
    print(SEPARATOR)


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Interactive demo — Sign Language → Sentence pipeline"
    )
    parser.add_argument(
        "--mode", choices=["auto", "manual", "batch"], default="auto",
        help="auto: 3 random videos | manual: type IDs | batch: multiple windows"
    )
    parser.add_argument(
        "--n", type=int, default=3,
        help="Number of windows to run in batch/auto mode (default: 3)"
    )
    args = parser.parse_args()

    print(SEPARATOR)
    print("  Sign Language → Sentence Generator  (Combined Pipeline)")
    print(SEPARATOR)

    pipe = SignToSentencePipeline()

    if args.mode == "auto":
        auto_mode(pipe, n_windows=args.n)
    elif args.mode == "manual":
        manual_mode(pipe)
    elif args.mode == "batch":
        batch_mode(pipe, n_windows=args.n)


if __name__ == "__main__":
    main()