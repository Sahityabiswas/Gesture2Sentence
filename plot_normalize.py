import pickle
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
with open("data_map_FDMSE-ISL_keypoints.pkl", "rb") as f:
    data = pickle.load(f)

video_id = list(data.keys())[0]   # or use your specific ID

seq = np.array(data[video_id])    # (T, 29, 2)

# Flatten → (T, 58)
seq_flat = seq.reshape(seq.shape[0], -1)

# =========================
# NORMALIZATION (fit on dataset)
# =========================
all_sequences = [np.array(v).reshape(len(v), -1) for v in data.values()]
all_frames = np.concatenate(all_sequences, axis=0)

mean = all_frames.mean(axis=0)
std  = all_frames.std(axis=0) + 1e-5

# Apply normalization
seq_norm = (seq_flat - mean) / std

# =========================
# PLOT ONE FRAME
# =========================
frame_id = 0

raw_frame = seq_flat[frame_id].reshape(29, 2)
norm_frame = seq_norm[frame_id].reshape(29, 2)

plt.figure(figsize=(10,4))

# Raw
plt.subplot(1,2,1)
plt.scatter(raw_frame[:,0], raw_frame[:,1])
plt.title("Before Normalization")
plt.gca().invert_yaxis()

# Normalized
plt.subplot(1,2,2)
plt.scatter(norm_frame[:,0], norm_frame[:,1])
plt.title("After Normalization")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()