import numpy as np


DEFAULT_POSE_IDXS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
DEFAULT_HAND_IDXS = [0, 4, 5, 8, 9, 12, 13, 16, 20]


class VideoKeypointExtractor:
    """
    Extract a (T, 29, 2) keypoint sequence from a raw video.

    Important:
    This mapping is a best-effort default using MediaPipe Holistic.
    It must match the preprocessing used to create the training pickle
    for reliable predictions.
    """

    def __init__(self, pose_indices=None, left_hand_indices=None, right_hand_indices=None):
        self.pose_indices = pose_indices or DEFAULT_POSE_IDXS
        self.left_hand_indices = left_hand_indices or DEFAULT_HAND_IDXS
        self.right_hand_indices = right_hand_indices or DEFAULT_HAND_IDXS

        total = len(self.pose_indices) + len(self.left_hand_indices) + len(self.right_hand_indices)
        if total != 29:
            raise ValueError(f"Expected exactly 29 selected keypoints, got {total}.")

        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "Raw-video inference requires 'opencv-python' and 'mediapipe'. "
                "Install them first, then rerun prediction."
            ) from exc

        self.cv2 = cv2
        self.mp = mp

    def _collect_points(self, landmarks, indices, frame_width, frame_height):
        if landmarks is None:
            return [[0.0, 0.0] for _ in indices]

        points = []
        for idx in indices:
            if idx >= len(landmarks.landmark):
                points.append([0.0, 0.0])
                continue
            lm = landmarks.landmark[idx]
            points.append([
                float(lm.x) * float(frame_width),
                float(lm.y) * float(frame_height),
            ])
        return points

    def _fill_missing_keypoints(self, seq):
        """
        Forward/backward fill missing landmarks to avoid collapsing many frames
        into identical all-zero inputs when detections flicker.
        """
        arr = np.asarray(seq, dtype=np.float32).copy()
        valid_mask = np.any(arr != 0.0, axis=2)  # (T, K)
        total_frames, total_keypoints, _ = arr.shape

        for keypoint_idx in range(total_keypoints):
            valid_indices = np.where(valid_mask[:, keypoint_idx])[0]
            if len(valid_indices) == 0:
                continue

            first_valid = valid_indices[0]
            last_valid = valid_indices[-1]
            arr[:first_valid, keypoint_idx, :] = arr[first_valid, keypoint_idx, :]
            arr[last_valid + 1:, keypoint_idx, :] = arr[last_valid, keypoint_idx, :]

            prev = first_valid
            for current in valid_indices[1:]:
                if current - prev > 1:
                    start = arr[prev, keypoint_idx, :]
                    end = arr[current, keypoint_idx, :]
                    gap = current - prev
                    for offset in range(1, gap):
                        alpha = offset / gap
                        arr[prev + offset, keypoint_idx, :] = (1 - alpha) * start + alpha * end
                prev = current

        return arr

    def summarize(self, seq):
        arr = np.asarray(seq, dtype=np.float32)
        zero_mask = np.all(arr == 0.0, axis=2)
        all_zero_frames = np.all(zero_mask, axis=1)

        return {
            "frames": int(arr.shape[0]),
            "keypoints": int(arr.shape[1]),
            "all_zero_frames": int(all_zero_frames.sum()),
            "all_zero_frame_ratio": float(all_zero_frames.mean()),
            "missing_point_ratio": float(zero_mask.mean()),
            "x_mean": float(arr[..., 0].mean()),
            "y_mean": float(arr[..., 1].mean()),
            "x_std": float(arr[..., 0].std()),
            "y_std": float(arr[..., 1].std()),
        }

    def extract(self, video_path):
        cap = self.cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frames = []
        mp_holistic = self.mp.solutions.holistic

        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
        ) as holistic:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_height, frame_width = frame.shape[:2]

                rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                frame_points = []
                frame_points.extend(
                    self._collect_points(results.pose_landmarks, self.pose_indices, frame_width, frame_height)
                )
                frame_points.extend(
                    self._collect_points(results.left_hand_landmarks, self.left_hand_indices, frame_width, frame_height)
                )
                frame_points.extend(
                    self._collect_points(results.right_hand_landmarks, self.right_hand_indices, frame_width, frame_height)
                )
                frames.append(frame_points)

        cap.release()

        if not frames:
            raise ValueError("No frames were extracted from the video.")

        raw = np.asarray(frames, dtype=np.float32)
        return self._fill_missing_keypoints(raw)
