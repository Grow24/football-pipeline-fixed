import argparse
import csv
import os
from collections import defaultdict
from enum import Enum
from typing import Iterator, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from pose_estimator import PoseEstimator

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

# Pitch real-world dimensions in meters (standard FIFA pitch)
PITCH_LENGTH_METERS = 105.0
PITCH_WIDTH_METERS = 68.0

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'


# ---------------------------------------------------------------------------
# NEW: Speed Estimator
# Converts pixel-space field coordinates to meters and computes speed (km/h).
# ---------------------------------------------------------------------------
class SpeedEstimator:
    """
    Estimates speed for tracked objects using consecutive field-coordinate
    positions and video FPS.

    Coordinates must already be in pitch space (pixels on the CONFIG canvas),
    so we scale them to real-world meters using the known pitch dimensions.
    """

    def __init__(self, fps: float, smoothing_window: int = 5):
        self.fps = fps
        self.smoothing_window = smoothing_window
        # tracker_id → deque of (frame_idx, x_m, y_m)
        self._history: dict = defaultdict(list)
        self._frame_idx: int = 0

    def update(
        self,
        tracker_ids: np.ndarray,
        field_xy: np.ndarray,  # shape (N, 2) in CONFIG canvas pixels
    ) -> dict:
        """
        Returns {tracker_id: speed_kmh} for all tracked objects.
        """
        # Scale canvas pixels → meters
        canvas_w = CONFIG.width    # 7000px = 68m wide
        canvas_h = CONFIG.length   # 12000px = 105m long
        scale_x = PITCH_WIDTH_METERS / canvas_w    # 68 / 7000
        scale_y = PITCH_LENGTH_METERS / canvas_h   # 105 / 12000

        self._frame_idx += 1
        speeds = {}

        for tid, (px, py) in zip(tracker_ids, field_xy):
            mx, my = px * scale_x, py * scale_y
            self._history[tid].append((self._frame_idx, mx, my))
            # Keep only the last N frames
            if len(self._history[tid]) > self.smoothing_window:
                self._history[tid].pop(0)

            history = self._history[tid]
            if len(history) >= 2:
                fi, xi, yi = history[0]
                ff, xf, yf = history[-1]
                dt = (ff - fi) / self.fps  # seconds
                dist = np.hypot(xf - xi, yf - yi)  # meters
                speed_ms = dist / dt if dt > 0 else 0.0
                speeds[tid] = min(speed_ms * 3.6, 45.0)  # m/s → km/h, capped at 45
            else:
                speeds[tid] = 0.0

        return speeds


# ---------------------------------------------------------------------------
# NEW: CSV Tracker
# Accumulates per-frame tracking data and writes to disk.
# ---------------------------------------------------------------------------
class CSVTracker:
    """
    Collects one row per detection per frame and writes to a CSV file when
    save() is called.

    Columns:
        frame, tracker_id, team_id, role, field_x_px, field_y_px,
        field_x_m, field_y_m, speed_kmh
    """

    FIELDNAMES = [
        'frame', 'tracker_id', 'team_id', 'role',
        'field_x_px', 'field_y_px',
        'field_x_m', 'field_y_m',
        'speed_kmh',
    ]

    def __init__(self):
        self._rows: List[dict] = []
        self._frame_idx: int = 0

        canvas_w = CONFIG.width
        canvas_h = CONFIG.length
        self._scale_x = PITCH_WIDTH_METERS / canvas_w
        self._scale_y = PITCH_LENGTH_METERS / canvas_h
    def record(
        self,
        tracker_ids: np.ndarray,
        field_xy: np.ndarray,
        team_ids: np.ndarray,
        roles: List[str],
        speeds: dict,
    ):
        self._frame_idx += 1
        for tid, (px, py), team, role in zip(tracker_ids, field_xy, team_ids, roles):
            if not (0 <= px <= 12000 and 0 <= py <= 7000):
                continue
            self._rows.append({
                'frame': self._frame_idx,
                'tracker_id': int(tid),
                'team_id': int(team),
                'role': role,
                'field_x_px': round(float(px), 2),
                'field_y_px': round(float(py), 2),
                'field_x_m': round(float(px) * self._scale_x, 2),
                'field_y_m': round(float(py) * self._scale_y, 2),
                'speed_kmh': round(speeds.get(int(tid), 0.0), 2),
            })

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._rows)
        print(f"[CSV] Saved {len(self._rows)} rows → {path}")


# ---------------------------------------------------------------------------
# NEW: Heatmap Generator
# Accumulates field positions and renders a per-team density heatmap.
# ---------------------------------------------------------------------------
class HeatmapGenerator:
    """
    Accumulates field-coordinate positions and generates a matplotlib heatmap
    image saved to disk after processing.
    """

    def __init__(self):
        # team_id → list of (x_m, y_m)
        self._positions: dict = defaultdict(list)

    def record(self, field_xy: np.ndarray, team_ids: np.ndarray):
        canvas_w = CONFIG.width
        canvas_h = CONFIG.length
        scale_x = PITCH_WIDTH_METERS / canvas_w
        scale_y = PITCH_LENGTH_METERS / canvas_h
        for (px, py), team in zip(field_xy, team_ids):
            self._positions[int(team)].append(
                (float(px) * scale_x, float(py) * scale_y)
            )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        team_colors = {0: COLORS[0], 1: COLORS[1]}
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#1a1a2e')

        for ax, (team_id, color) in zip(axes, team_colors.items()):
            pts = self._positions.get(team_id, [])
            ax.set_facecolor('#1a1a2e')
            ax.set_xlim(0, PITCH_LENGTH_METERS)
            ax.set_ylim(0, PITCH_WIDTH_METERS)
            ax.set_aspect('equal')
            ax.set_title(f'Team {team_id} Heatmap', color='white', fontsize=13)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')

            if pts:
                xs, ys = zip(*pts)
                h = ax.hist2d(
                    xs, ys,
                    bins=[int(PITCH_LENGTH_METERS), int(PITCH_WIDTH_METERS)],
                    range=[[0, PITCH_LENGTH_METERS], [0, PITCH_WIDTH_METERS]],
                    cmap='hot',
                    density=True,
                )
                plt.colorbar(h[3], ax=ax, label='Density').ax.yaxis.label.set_color('white')
            else:
                ax.text(
                    PITCH_LENGTH_METERS / 2, PITCH_WIDTH_METERS / 2,
                    'No data', color='white', ha='center', va='center'
                )

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print(f"[Heatmap] Saved → {path}")

# ---------------------------------------------------------------------------
# NEW: Player Trail Buffer for Step 10 — 2D Tactical Replay
# ---------------------------------------------------------------------------
class PlayerTrailBuffer:
    """
    Stores the last N field positions per tracker_id for drawing movement trails.
    """
    def __init__(self, max_len: int = 30):
        self.max_len = max_len
        self._trails: dict = defaultdict(list)

    def update(self, tracker_ids: np.ndarray, field_xy: np.ndarray, color_lookup: np.ndarray):
        for tid, xy, color in zip(tracker_ids, field_xy, color_lookup):
            self._trails[int(tid)].append((xy[0], xy[1], int(color)))
            if len(self._trails[int(tid)]) > self.max_len:
                self._trails[int(tid)].pop(0)

    def get_trails(self):
        return self._trails
    
def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray,
    ball_field_xy: Optional[np.ndarray] = None,
    speed_labels: Optional[List[str]] = None,
    trails: Optional[dict] = None,
    transformed_xy: Optional[np.ndarray] = None,
) -> np.ndarray:
    if transformed_xy is None:
        if len(keypoints.xy) == 0:
            transformed_xy = np.empty((0, 2), dtype=np.float32)
        else:
            mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
            transformer = ViewTransformer(
                source=keypoints.xy[0][mask].astype(np.float32),
                target=np.array(CONFIG.vertices)[mask].astype(np.float32)
            )
            xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)

    # NEW: draw player trails (fading dots showing recent movement)
    if trails:
        for tid, positions in trails.items():
            n = len(positions)
            for i, (px, py, color_idx) in enumerate(positions[:-1]):
                # fade: older positions are smaller and more transparent
                alpha = (i + 1) / n
                radius = max(4, int(10 * alpha))
                trail_color = sv.Color.from_hex(COLORS[min(color_idx, len(COLORS)-1)])
                radar = draw_points_on_pitch(
                    config=CONFIG,
                    xy=np.array([[px, py]]),
                    face_color=trail_color,
                    radius=radius,
                    pitch=radar,
                )

    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)

    # NEW: draw ball on radar as a white dot
    if ball_field_xy is not None and len(ball_field_xy) > 0:
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=ball_field_xy,
            face_color=sv.Color.from_hex('#FFFFFF'),
            radius=12,
            pitch=radar,
        )

    # NEW: draw speed labels next to each player dot on the radar
    if speed_labels is not None:
        canvas_scale_x = CONFIG.width / PITCH_LENGTH_METERS
        canvas_scale_y = CONFIG.length / PITCH_WIDTH_METERS
        for (cx, cy), label in zip(transformed_xy, speed_labels):
            cv2.putText(
                radar,
                label,
                (int(cx) + 14, int(cy) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return radar, transformed_xy


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    last_valid_transformer = None
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]
        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    last_valid_transformer = None
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def run_radar(
    source_video_path: str,
    device: str,
    csv_path: Optional[str] = None,
    heatmap_path: Optional[str] = None,
    pose_path: Optional[str] = None,
) -> Iterator[np.ndarray]:
    """
    Extended RADAR mode with:
      - Ball shown on minimap
      - Speed labels on minimap and video frame
      - CSV export of all tracking data
      - Heatmap saved after processing
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)

    # Ball detection slicer (same as run_ball_detection)
    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    ball_slicer = sv.InferenceSlicer(
        callback=ball_callback,
        overlap_filter=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )
    ball_tracker_obj = BallTracker(buffer_size=20)

    # Collect crops for team classifier
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # Get FPS for speed estimation
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    fps = video_info.fps

    speed_estimator = SpeedEstimator(fps=fps, smoothing_window=5)
    csv_tracker = CSVTracker() if csv_path else None
    heatmap_gen = HeatmapGenerator() if heatmap_path else None
    trail_buffer = PlayerTrailBuffer(max_len=30)
    pose_estimator  = PoseEstimator(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) if pose_path else None

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    last_valid_transformer = None

    for frame in frame_generator:
        # --- Pitch keypoints ---
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        # --- Player detections ---
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)
        # Step 2: Pose estimation
        if pose_estimator is not None and players.tracker_id is not None:
            for crop, tid in zip(crops, players.tracker_id):
                joints = pose_estimator.process_crop(crop)
                pose_estimator.record(
                    frame_idx=csv_tracker._frame_idx,
                    tracker_id=int(tid),
                    joints=joints,
                )

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        if detections.tracker_id is None:
            continue
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        # --- Ball detection (NEW) ---
        ball_detections = ball_slicer(frame).with_nms(threshold=0.1)
        ball_detections = ball_tracker_obj.update(ball_detections)

        # --- Compute homography transformer (Phase 3: robust) ---
        if len(keypoints.xy) == 0:
            transformer = last_valid_transformer
        else:
            mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
            n_keypoints = mask.sum()
            if n_keypoints >= 6:
                try:
                    transformer = ViewTransformer(
                        source=keypoints.xy[0][mask].astype(np.float32),
                        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
                    )
                    last_valid_transformer = transformer
                except Exception:
                    transformer = last_valid_transformer
            else:
                transformer = last_valid_transformer
        if transformer is None:
            continue

        # --- Transform player positions to field coords ---
        player_xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        field_xy = transformer.transform_points(points=player_xy)

        # --- Phase 3: filter OOB detections ---
        CANVAS_X_MAX = CONFIG.length
        CANVAS_Y_MAX = CONFIG.width
        valid_mask = (
            (field_xy[:, 0] >= 0) & (field_xy[:, 0] <= CANVAS_X_MAX) &
            (field_xy[:, 1] >= 0) & (field_xy[:, 1] <= CANVAS_Y_MAX)
        )
        field_xy = field_xy[valid_mask]
        detections = detections[valid_mask]
        color_lookup = color_lookup[valid_mask]

        # --- Speed estimation (NEW) ---
        valid_tracker_ids = detections.tracker_id

        # --- Update trail buffer (Step 10) ---
        trail_buffer.update(valid_tracker_ids, field_xy, color_lookup)
        speeds = speed_estimator.update(valid_tracker_ids, field_xy)
        speed_labels = [
            f"{speeds.get(int(tid), 0.0):.1f}" for tid in valid_tracker_ids
        ]
        # Labels for video overlay: tracker_id + speed
        video_labels = [
            f"#{tid} {speeds.get(int(tid), 0.0):.1f}km/h"
            for tid in valid_tracker_ids
        ]

        # --- Transform ball to field coords (NEW) ---
        ball_field_xy = None
        if len(ball_detections) > 0:
            ball_xy = ball_detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER)
            ball_field_xy = transformer.transform_points(points=ball_xy)

        # --- CSV tracking (NEW) ---
        if csv_tracker is not None:
            roles = (
                ['player'] * len(players) +
                ['goalkeeper'] * len(goalkeepers) +
                ['referee'] * len(referees)
            )
            team_ids_combined = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
            )
            csv_tracker.record(
                tracker_ids=valid_tracker_ids,
                field_xy=field_xy,
                team_ids=team_ids_combined,
                roles=roles,
                speeds=speeds,
            )

        # --- Heatmap accumulation (NEW) ---
        if heatmap_gen is not None:
            player_and_gk_xy = field_xy[:len(players) + len(goalkeepers)]
            player_and_gk_teams = np.array(
                players_team_id.tolist() + goalkeepers_team_id.tolist()
            )
            if len(player_and_gk_xy) > 0:
                heatmap_gen.record(player_and_gk_xy, player_and_gk_teams)

        # --- Annotate video frame ---
        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        # Use speed-enriched labels on the video frame
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, video_labels,
            custom_color_lookup=color_lookup)

        # --- Build radar with ball + speed (NEW) ---
        h, w, _ = frame.shape
        radar, _ = render_radar(
            detections, keypoints, color_lookup,
            ball_field_xy=ball_field_xy,
            speed_labels=speed_labels,
            trails=trail_buffer.get_trails(),
            transformed_xy=field_xy,
        )
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame

    # --- Save CSV and heatmap after all frames processed (NEW) ---
    if csv_tracker is not None and csv_path:
        csv_tracker.save(csv_path)
    if heatmap_gen is not None and heatmap_path:
        heatmap_gen.save(heatmap_path)
        if pose_estimator is not None and pose_path:
            pose_estimator.save(pose_path)
            pose_estimator.close()


def main(
    source_video_path: str,
    target_video_path: str,
    device: str,
    mode: Mode,
    csv_path: Optional[str] = None,
    heatmap_path: Optional[str] = None,
    pose_path: Optional[str] = None,
) -> None:
    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path,
            device=device,
            csv_path=csv_path,
            heatmap_path=heatmap_path,
            pose_path=pose_path,
        )
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)
            pass  # cv2.imshow disabled for Colab


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soccer AI Video Analysis')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION)
    # NEW arguments
    parser.add_argument(
        '--csv_path', type=str, default=None,
        help='Path to save tracking CSV (RADAR mode only). E.g. output/tracking.csv')
    parser.add_argument(
        '--heatmap_path', type=str, default=None,
        help='Path to save heatmap PNG (RADAR mode only). E.g. output/heatmap.png')
    parser.add_argument(
        '--pose_path', type=str, default=None,
        help='Path to save pose CSV (RADAR mode only). E.g. output/pose.csv')
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        csv_path=args.csv_path,
        heatmap_path=args.heatmap_path,
        pose_path=args.pose_path,
    )
