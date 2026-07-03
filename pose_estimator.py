"""
pose_estimator.py — Step 2: MediaPipe Pose Estimation
Plugs into main-2.py's run_radar() loop.

Usage:
    from pose_estimator import PoseEstimator
    pose_est = PoseEstimator()
    joints = pose_est.process_crop(crop_bgr)        # single crop → dict of joints
    pose_est.record(frame_idx, tracker_id, joints)  # accumulate
    pose_est.save("data/pose.csv")                  # write after loop
"""

import csv
import os
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe landmark index → human-readable joint name
# Full 33-landmark set:
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# ---------------------------------------------------------------------------
LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",   "left_eye",    "left_eye_outer",
    "right_eye_inner",  "right_eye",   "right_eye_outer",
    "left_ear",         "right_ear",
    "mouth_left",       "mouth_right",
    "left_shoulder",    "right_shoulder",
    "left_elbow",       "right_elbow",
    "left_wrist",       "right_wrist",
    "left_pinky",       "right_pinky",
    "left_index",       "right_index",
    "left_thumb",       "right_thumb",
    "left_hip",         "right_hip",
    "left_knee",        "right_knee",
    "left_ankle",       "right_ankle",
    "left_heel",        "right_heel",
    "left_foot_index",  "right_foot_index",
]  # 33 landmarks total

# Joints we actually care about for football analysis (subset for lighter CSV)
KEY_JOINTS = [
    "nose",
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_hip",       "right_hip",
    "left_knee",      "right_knee",
    "left_ankle",     "right_ankle",
    "left_heel",      "right_heel",
]  # 15 joints = 30 columns (x, y each)

# Skeleton connections — pairs of joint names to draw lines between
SKELETON_CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("left_ankle",     "left_heel"),
    ("right_ankle",    "right_heel"),
]


# ---------------------------------------------------------------------------
# CSV column headers
# ---------------------------------------------------------------------------
def _build_fieldnames() -> List[str]:
    """Build CSV headers: frame, tracker_id, then x/y per key joint."""
    headers = ["frame", "tracker_id", "detection_confidence"]
    for joint in KEY_JOINTS:
        headers += [f"{joint}_x", f"{joint}_y", f"{joint}_visibility"]
    return headers


FIELDNAMES = _build_fieldnames()


# ---------------------------------------------------------------------------
# PoseEstimator class
# ---------------------------------------------------------------------------
class PoseEstimator:
    """
    Wraps MediaPipe Pose for football player crops.

    Designed to be created once before the frame loop and reused.
    All results are accumulated in memory and written to CSV at the end.

    Parameters
    ----------
    min_detection_confidence : float
        MediaPipe detection threshold (lower = more detections, more noise).
    min_tracking_confidence : float
        MediaPipe tracking threshold between frames.
    use_full_landmarks : bool
        If True, saves all 33 joints. If False (default), saves KEY_JOINTS only.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_full_landmarks: bool = False,
    ):
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=True,          # each crop is independent
            model_complexity=1,              # 0=lite, 1=full, 2=heavy
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._use_full = use_full_landmarks
        self._active_joints = LANDMARK_NAMES if use_full_landmarks else KEY_JOINTS
        self._rows: List[dict] = []

    # ------------------------------------------------------------------
    # Core: process a single player crop
    # ------------------------------------------------------------------
    def process_crop(
        self,
        crop_bgr: np.ndarray,
        min_size: int = 40,
    ) -> Optional[Dict[str, float]]:
        """
        Run MediaPipe Pose on a single player crop (BGR numpy array).

        Returns a dict of {joint_x, joint_y, joint_visibility} for each
        active joint, with coordinates normalised 0-1 within the crop.
        Returns None if the crop is too small or no pose detected.

        Parameters
        ----------
        crop_bgr : np.ndarray
            Player crop in BGR format (from supervision crop_image).
        min_size : int
            Skip crops smaller than this in either dimension.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < min_size or w < min_size:
            return None

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks.landmark
        joints = {}
        for name in self._active_joints:
            idx = LANDMARK_NAMES.index(name)
            lm = landmarks[idx]
            # x, y are already normalised 0-1 within the image
            joints[f"{name}_x"]          = round(lm.x, 4)
            joints[f"{name}_y"]          = round(lm.y, 4)
            joints[f"{name}_visibility"] = round(lm.visibility, 3)

        return joints

    # ------------------------------------------------------------------
    # Accumulate results
    # ------------------------------------------------------------------
    def record(
        self,
        frame_idx: int,
        tracker_id: int,
        joints: Optional[Dict[str, float]],
        confidence: float = 1.0,
    ):
        """
        Store one row. Call after process_crop() for each player.
        If joints is None (no pose detected), stores NaN for all joints.
        """
        row = {
            "frame":                frame_idx,
            "tracker_id":           tracker_id,
            "detection_confidence": round(confidence, 3),
        }
        if joints is not None:
            row.update(joints)
        else:
            # Fill NaN so the CSV row still exists (easier to join later)
            for name in self._active_joints:
                row[f"{name}_x"]          = float("nan")
                row[f"{name}_y"]          = float("nan")
                row[f"{name}_visibility"] = float("nan")
        self._rows.append(row)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Write all accumulated rows to pose.csv."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # Build fieldnames dynamically based on active joints
        fieldnames = ["frame", "tracker_id", "detection_confidence"]
        for name in self._active_joints:
            fieldnames += [f"{name}_x", f"{name}_y", f"{name}_visibility"]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)

        detected = sum(1 for r in self._rows if not _is_nan_row(r))
        total    = len(self._rows)
        print(
            f"[Pose] Saved {total} rows → {path} "
            f"({detected} with pose detected, "
            f"{total - detected} no-detection)"
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        """Release MediaPipe resources."""
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_nan_row(row: dict) -> bool:
    """Check if all joint values are NaN (no detection)."""
    try:
        return all(
            isinstance(v, float) and (v != v)  # NaN check
            for k, v in row.items()
            if k.endswith("_x")
        )
    except Exception:
        return True


def draw_skeleton_on_crop(
    crop_bgr: np.ndarray,
    joints: Dict[str, float],
    color: tuple = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """
    Debug utility — draw skeleton lines on a player crop.
    Useful for visually verifying pose output during development.

    Parameters
    ----------
    crop_bgr : np.ndarray
        The player crop image.
    joints : dict
        Output from process_crop().
    color : tuple
        BGR line colour.
    """
    h, w = crop_bgr.shape[:2]
    vis = crop_bgr.copy()

    def get_pt(name):
        x = joints.get(f"{name}_x")
        y = joints.get(f"{name}_y")
        if x is None or y is None:
            return None
        return (int(x * w), int(y * h))

    for a, b in SKELETON_CONNECTIONS:
        pt_a = get_pt(a)
        pt_b = get_pt(b)
        if pt_a and pt_b:
            cv2.line(vis, pt_a, pt_b, color, thickness, cv2.LINE_AA)

    for name in KEY_JOINTS:
        pt = get_pt(name)
        if pt:
            cv2.circle(vis, pt, 3, (255, 255, 255), -1)

    return vis
