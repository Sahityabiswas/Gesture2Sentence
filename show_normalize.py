import pickle
import numpy as np
from normalize import fit, transform   # your file

# Load dataset
with open("data_map_FDMSE-ISL_keypoints.pkl", "rb") as f:
    data = pickle.load(f)

video_id = "0005715"  # change to any train sample

seq = np.array(data[video_id])   # shape (T, 29, 2)

# Flatten to (T, 58)
seq_flat = seq.reshape(seq.shape[0], -1)

# Fit normalization on whole dataset
all_sequences = [np.array(v).reshape(len(v), -1) for v in data.values()]
mean, std = fit(all_sequences)

# Apply normalization
seq_norm = (seq_flat - mean) / std

# Print stats
print("----- BEFORE NORMALIZATION -----")
print("Mean:", seq_flat.mean())
print("Std :", seq_flat.std())
print("Min :", seq_flat.min())
print("Max :", seq_flat.max())

print("\n----- AFTER NORMALIZATION -----")
print("Mean:", seq_norm.mean())
print("Std :", seq_norm.std())
print("Min :", seq_norm.min())
print("Max :", seq_norm.max())