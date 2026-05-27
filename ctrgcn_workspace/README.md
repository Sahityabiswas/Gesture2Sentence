# CTRGCN Workspace

This folder is a separate workspace for experimenting with `CTRGCN` without changing any existing files in the main project.

## What Is Included

- `config.py`: local settings and artifact paths
- `graph.py`: editable skeleton graph definition for 29 keypoints
- `dataset_ctrgcn.py`: data loading, normalization, padding, and graph reshaping
- `model_ctrgcn.py`: a compact `CTRGCN`-style model
- `train_ctrgcn.py`: single-stage training script for all classes
- `evaluate_ctrgcn.py`: evaluation script for a saved checkpoint

## Important Note

The raw data appears to be shaped like `(T, 29, 2)`. This workspace uses that directly.

The graph edges in `graph.py` are a safe starter graph for 29 landmarks. If your 29 keypoints follow a known pose layout, update the edges there for better accuracy.

## Run

```powershell
cd E:\project_26\deep_learning\dl_project\sign_language_detection\ctrgcn_workspace
python train_ctrgcn.py
python evaluate_ctrgcn.py --checkpoint artifacts\best_model.pt
```

## Why This Is Safe

Everything here saves into:

`E:\project_26\deep_learning\dl_project\sign_language_detection\ctrgcn_workspace\artifacts`

So your existing Transformer, BiLSTM, hierarchical checkpoints, and scripts remain untouched.
