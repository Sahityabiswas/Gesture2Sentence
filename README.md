# Sign Language Detection and Sentence Generation

This repository contains a sign-language recognition project built around a hierarchical classifier for isolated sign prediction, plus an optional text-generation stage that converts predicted sign keywords into natural-language sentences.

The project includes:

- A hierarchical sign classifier for large-class prediction
- Raw-video inference using extracted keypoints
- Evaluation and inference-setting sweeps
- A sign-to-sentence pipeline based on `T5`
- A separate `CTRGCN` experiment workspace

## Project Overview

The main recognition pipeline works in two stages:

1. A group model predicts which sign group a sample belongs to.
2. A group-specific submodel predicts the final sign class inside that group.

This reduces a very large classification problem into a more manageable hierarchical setup.

The repository also includes a second-stage language model pipeline that takes a short sequence of predicted signs and generates a natural sentence from them.

## Architecture

```mermaid
graph TD
    subgraph Input
        A[Raw Video File]
        A2[Dataset Video ID]
    end

    subgraph Preprocessing
        B[video_keypoint_extractor.py<br/>VideoKeypointExtractor - MediaPipe]
        C[video_preprocess_utils.py<br/>resample + preprocess]
        D[normalize.py<br/>normalize keypoints - global_stats.pkl]
        E[dataset.py / dataset.csv<br/>dataset video keypoints]
    end

    A --> B --> C --> D
    A2 --> E --> D

    subgraph Training["Train_hierarchical.py"]
        F[Build label_map.pkl]
        G[KMeans clustering<br/>classes to groups - group_map.pkl]
        H[Train Group Model - Stage 1]
        I[Train Per-Group Submodels - Stage 2]
    end

    D --> F --> G --> H --> I
    M[model.py<br/>model architectures] --> H
    M --> I
    K[config.py<br/>NUM_GROUPS HIDDEN_SIZE etc] --> Training

    subgraph Inference["hierarchical_inference.py"]
        J1[Stage 1: predict sign group]
        J2[Stage 2: predict class within group]
    end
    H --> J1 --> J2
    I --> J2
    label_map.pkl -.-> J2
    inference_utils.py[inference_utils.py /<br/>inference_settings.json] --> J1

    D --> J1
    J2 --> P1[predict.py<br/>predict by dataset vid]
    J2 --> P2[predict_video.py<br/>predict from raw video]
    C --> P2

    J2 --> EV[evaluate.py<br/>eval + sweep top-k/temperature]
    EV --> REP[evaluation_reports/]

    subgraph SentencePipeline["Sign-to-Sentence Pipeline"]
        S1[pipeline.py<br/>predict 3-video window]
        S2[Convert predictions to keyword text]
        S3[T5 Generator<br/>wts_split/best_model]
    end

    J2 --> S1 --> S2 --> S3 --> S4[Generated Sentence]

    subgraph T5Training["wts_split/"]
        T1[training.py<br/>train T5 on keywords to sentence]
        T2[inference.py<br/>keyword to sentence CLI]
    end
    T1 --> S3
    T2 --> S3

    DEMO[demo.py<br/>interactive CLI demo] --> S1

    subgraph CTRGCN["ctrgcn_workspace/ (isolated experiment)"]
        CT1[train_ctrgcn.py]
        CT2[evaluate_ctrgcn.py]
    end
    D -.shared keypoint format.-> CT1
    CT1 --> CT2

    subgraph Diagnostics["Utility / Exploration Scripts"]
        U1[compare_video_features.py]
        U2[plot_normalize.py]
        U3[show_normalize.py]
        U4[Explore_id.py]
        U5[my_check.py]
    end
    D -.-> Diagnostics
```

## Main Components

- `Train_hierarchical.py`
  Trains the hierarchical classifier:
  - builds label maps
  - normalizes keypoint sequences
  - clusters classes into groups with feature-based `KMeans`
  - trains the stage-1 group model
  - trains one stage-2 submodel per group

- `evaluate.py`
  Evaluates the hierarchical system on the test split and can sweep inference settings such as:
  - top-k groups
  - top-k classes
  - temperature
  - group-score power

- `predict.py`
  Runs prediction for a dataset video ID using saved artifacts.

- `predict_video.py`
  Runs prediction directly from a raw video file by:
  - extracting keypoints
  - resampling and normalizing the sequence
  - running hierarchical inference
  - printing top predictions and basic quality diagnostics

- `pipeline.py`
  Combines sign prediction with a `T5` text generator to produce sentences from a window of predicted signs.

- `demo.py`
  Interactive CLI demo for the end-to-end sign-to-sentence pipeline.

- `ctrgcn_workspace/`
  Separate experimental workspace for a `CTRGCN`-style model using the same keypoint format.

- `wts_split/`
  Training and inference code for the keyword-to-sentence generation model.

## Repository Structure

```text
sign_language_detection/
|-- README.md
|-- .gitignore
|-- config.py
|-- dataset.py
|-- model.py
|-- normalize.py
|-- Train_hierarchical.py
|-- hierarchical_inference.py
|-- evaluate.py
|-- predict.py
|-- predict_video.py
|-- pipeline.py
|-- demo.py
|-- video_keypoint_extractor.py
|-- video_preprocess_utils.py
|-- inference_utils.py
|-- ctrgcn_workspace/
|-- wts_split/
|-- evaluation_reports/
```

Large datasets, checkpoints, and generated artifacts are intentionally excluded from GitHub through `.gitignore`.

## Data and Artifacts

The code expects local dataset and artifact files such as:

- `data_map_FDMSE-ISL_keypoints.pkl`
- `vid_class_FDMSE-ISL.pkl`
- `vid_splits_FDMSE-ISL.pkl`
- `class_map_FDMSE-ISL.csv`
- trained artifact files like:
  - `global_stats.pkl`
  - `label_map.pkl`
  - `group_map.pkl`
  - `group_model.pt`
  - `submodels/group_*.pt`

These files are not suitable for a normal GitHub push because some of them are very large.

## Environment Setup

Create and activate a Python environment, then install the required packages.

Example:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision torchaudio numpy pandas scikit-learn transformers
pip install opencv-python mediapipe
```

Depending on your code path, you may also need:

```powershell
pip install matplotlib
```

If you use GPU acceleration, install the CUDA-compatible version of `PyTorch` for your system.

## Configuration

Main configuration lives in `config.py`.

Important settings include:

- `NUM_GROUPS`
- `HIDDEN_SIZE`
- `NUM_LAYERS`
- `BATCH_SIZE`
- `EPOCHS`
- `LR`
- `GROUP_SCORE_POWER`

Artifacts are controlled through:

- `SLD_ARTIFACT_DIR`
- `SLD_ENSEMBLE_DIRS`
- `SLD_SEED`

By default, the project reads and writes artifacts from the current directory unless `SLD_ARTIFACT_DIR` is set.

## Training the Hierarchical Classifier

Run:

```powershell
python Train_hierarchical.py
```

This script will:

- load raw sign data
- build class labels
- normalize keypoint features
- cluster classes into groups
- train the group classifier
- train one submodel per group
- save training artifacts

## Evaluating the Model

Run a normal evaluation:

```powershell
python evaluate.py
```

Run a hyperparameter sweep:

```powershell
python evaluate.py --sweep --save-best-settings
```

Generated reports are written to `evaluation_reports/`.

## Predicting from Dataset Video IDs

To test on a video ID already present in the dataset:

```powershell
python predict.py --vid YOUR_VIDEO_ID
```

You can also specify alternate artifact directories:

```powershell
python predict.py --vid YOUR_VIDEO_ID --artifact-dirs "artifacts\\run1,artifacts\\run2"
```

## Predicting from a Raw Video File

To predict directly from a raw input video:

```powershell
python predict_video.py --video path\\to\\input.mp4
```

Useful options:

```powershell
python predict_video.py --video path\\to\\input.mp4 --topk 5 --target-frames 150
python predict_video.py --video path\\to\\input.mp4 --match-dataset-scale
```

This path uses `VideoKeypointExtractor` and preprocessing utilities before running hierarchical inference.

## Sign-to-Sentence Pipeline

The repository also includes a pipeline that converts a small sequence of predicted signs into a sentence.

Run the demo:

```powershell
python demo.py
```

Other modes:

```powershell
python demo.py --mode manual
python demo.py --mode batch --n 5
```

The pipeline:

1. Predicts signs for a 3-video window
2. Converts them into keyword text
3. Uses a `T5` generator to produce sentence candidates

## Training the Sentence Generator

Inside `wts_split/`, train the keyword-to-sentence model:

```powershell
cd wts_split
python training.py
```

Run inference for custom keywords:

```powershell
python inference.py --keywords "hello thank-you help"
```

The trained `T5` model is saved under:

```text
wts_split/best_model/
```

This folder is ignored in Git because it contains large model files.

## CTRGCN Experiment Workspace

The `ctrgcn_workspace/` folder is an isolated experiment area for graph-based sign modeling.

Run:

```powershell
cd ctrgcn_workspace
python train_ctrgcn.py
python evaluate_ctrgcn.py --checkpoint artifacts\best_model.pt
```

This keeps graph-model experiments separate from the main hierarchical pipeline.
