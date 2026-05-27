# explore_ids.py  —  Find and inspect valid video IDs from your dataset
#
# Run:  python explore_ids.py

import pickle
import config

# ── Load all pkl files ────────────────────────────────────────────────────────
print("Loading pkl files...")
with open(config.DATA_MAP_PATH,   "rb") as f: data_map   = pickle.load(f)
with open(config.VID_CLASS_PATH,  "rb") as f: vid_class  = pickle.load(f)
with open(config.VID_SPLITS_PATH, "rb") as f: vid_splits = pickle.load(f)

# ── 1. What does a video ID look like? ───────────────────────────────────────
print("\n── Sample video IDs (first 10 from data_map) ──")
sample_ids = list(data_map.keys())[:10]
for vid in sample_ids:
    label  = vid_class.get(vid, "NO LABEL")
    frames = len(data_map[vid])
    print(f"  ID: {vid!r:<30}  label: {label!r:<20}  frames: {frames}")

# ── 2. Split breakdown ────────────────────────────────────────────────────────
print("\n── Split sizes ──")
for split_name, ids in vid_splits.items():
    print(f"  {split_name:<10}: {len(ids)} videos")

# ── 3. Show IDs per split ─────────────────────────────────────────────────────
print("\n── First 5 video IDs from each split ──")
for split_name, ids in vid_splits.items():
    print(f"\n  [{split_name}]")
    for vid in list(ids)[:5]:
        label = vid_class.get(vid, "NO LABEL")
        print(f"    {vid!r:<30}  label: {label!r}")

# ── 4. Valid IDs for pipeline (in data_map + have a label) ───────────────────
print("\n── Valid test IDs ready for pipeline.py ──")
valid_test = [
    v for v in vid_splits.get("test", [])
    if v in data_map and v in vid_class
]
print(f"  Total valid test videos: {len(valid_test)}")
print(f"\n  First 10:")
for vid in valid_test[:10]:
    print(f"    {vid!r}")

# ── 5. Quick copy-paste example ───────────────────────────────────────────────
if len(valid_test) >= 3:
    trio = valid_test[:3]
    print(f"\n── Ready-to-use example command ──")
    print(f"  python pipeline.py --vids {trio[0]} {trio[1]} {trio[2]}")
    print(f"\n── Or in Python ──")
    print(f"  from pipeline import SignToSentencePipeline")
    print(f"  pipe = SignToSentencePipeline()")
    print(f"  result = pipe.run({trio})")