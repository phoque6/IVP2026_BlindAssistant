#!/usr/bin/env python3
"""
Team Bravo Vision Assistant v10 — beginner LiDAR SLAM floor-plan map
==================================================================================

v10: Scan-to-scan motion estimate builds a world occupancy floor plan as you walk
     (MathWorks-style layout). Camera, OCR, voice, and rainbow LiDAR panel kept.

Hardware:
  - Raspberry Pi 5
  - Raspberry Pi AI HAT / AI accelerator (26 TOPS, Hailo)
  - Raspberry Pi Camera
  - D6 AA55 2D LiDAR (USB serial)

Install:
    sudo apt update
    sudo apt install python3-serial python3-pygame python3-opencv python3-numpy espeak-ng
    sudo apt install python3-picamera2
    sudo apt install tesseract-ocr python3-pytesseract

Raspberry Pi AI HAT / AI Kit (Hailo):
    Follow Raspberry Pi AI Kit documentation to install Hailo runtime.
    Place model at: models/yolov8n.hef
    Place labels at: models/coco_labels.txt
    Optional OpenCV DNN fallback: models/yolov8n.onnx

    Insert Hailo SDK code in init_ai_hat() / run_ai_hat_inference() where marked.

Run:
    python3 team_bravo_aihat_camera_lidar_vision_assistant_v10_beginner_slam_floorplan.py

Safety:
    Prototype assistive navigation aid only — NOT the sole safety device for a blind
    person. LiDAR may miss glass, shiny surfaces, low objects, soft materials.
    Camera AI may misclassify. OCR may misread signs. AI HAT supports but does not
    replace LiDAR distance safety. Test with human supervision.
"""

from __future__ import annotations

import csv
import difflib
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

try:
    import cv2
except Exception:
    cv2 = None

try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None
    list_ports = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None

# AI HAT SDK placeholder import. Keep this safe and optional.
try:
    import hailo_platform  # noqa: F401
    HAILO_AVAILABLE = True
except Exception:
    HAILO_AVAILABLE = False


# =============================================================================
# REQUIRED USER SPEC CONSTANTS
# =============================================================================
SIMULATED_MODE = False
ENABLE_CAMERA = True
ENABLE_AI_HAT = True
ENABLE_LIDAR = True
ENABLE_OCR = True
ENABLE_VOICE_ALERTS = True

# Camera source: "auto", "picamera2", "usb", "sim", "none"
CAMERA_BACKEND = "auto"
PREFER_USB_CAMERA = True
USB_CAMERA_INDEXES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
ACTIVE_USB_CAMERA_INDEX = None  # runtime: see active_usb_camera_index
FORCE_CAMERA_SIMULATION = False

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_DISPLAY_WIDTH = 320
CAMERA_DISPLAY_HEIGHT = 240
CAMERA_TARGET_FPS = 30
CAMERA_USB_BUFFER_SIZE = 1
USB_USE_MJPEG = True
USB_WARMUP_FRAMES = 10
CAMERA_RETRY_SECONDS = 3.0
CAMERA_FREEZE_SECONDS = 2.0
SHOW_CAMERA_ERROR_PANEL = True

AI_MODELS_DIR = "models"
AI_MODEL_PATH = "models/yolov8n.hef"
AI_LABELS_PATH = "models/coco_labels.txt"
AI_DNN_MODEL_PATH = "models/yolov8n.onnx"
AI_CONFIDENCE_THRESHOLD = 0.28
AI_NMS_THRESHOLD = 0.45
AI_PROCESS_WIDTH = 416
AI_PROCESS_HEIGHT = 312
AI_DNN_INPUT_SIZE = 640
DNN_EVERY_N_FRAMES = 1  # try DNN every AI cycle when model exists

# Fallback COCO-80 names if labels file is missing
COCO_DEFAULT_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Time-based camera processing (decoupled from capture rate)
OCR_INTERVAL_SECONDS = 1.6
OCR_VOICE_REPEAT_SECONDS = 6.0
OCR_PERSIST_SECONDS = 14.0
OCR_FRAME_SCALE = 3.2
AI_DETECTION_INTERVAL_SECONDS = 0.40
CAMERA_SLEEP_SECONDS = 0.001

OCR_MIN_TEXT_LENGTH = 2
OCR_MIN_NUMBER_LENGTH = 1
OCR_MAX_TEXT_LENGTH = 72  # street names / longer signs
OCR_WHOLE_FRAME_FALLBACK = False
# Vocabulary corrects common signs (EX1T->EXIT) but does NOT block unknown text
OCR_USE_SIGN_VOCABULARY = False
OCR_VOCABULARY_CORRECTION = True
OCR_MIN_CONFIDENCE_SCORE = 0.20
OCR_REQUIRE_STABLE_READS = 1
OCR_MANUAL_CONFIRM_SCORE = 0.22
OCR_AUTO_SINGLE_READ_SCORE = 0.36
OCR_STABLE_WINDOW_SECONDS = 12.0
OCR_FUZZY_MATCH_THRESHOLD = 0.70
OCR_VOCAB_MAX_WORDS = 3
OCR_VOCAB_MAX_CHARS = 18
STREET_NAME_HINTS = (
    "ROAD", "STREET", "ST", "AVE", "AVENUE", "LANE", "DRIVE", "DR",
    "BLVD", "BOULEVARD", "WAY", "PLACE", "PL", "COURT", "CT", "CRESCENT",
    "HIGHWAY", "HWY", "CLOSE", "WALK", "PATH", "BRIDGE", "JALAN", "LORONG",
)
STREET_NAME_COMPACT_HINTS = (
    "AVENUE", "STREET", "ROAD", "AVE", "LANE", "DRIVE", "CRESCENT",
    "BOULEVARD", "HIGHWAY", "JALAN", "LORONG",
)
# Centre first, but always fall back to extra ROIs if centre is weak
OCR_CENTRE_FIRST_ONLY = False
OCR_MAX_EXTRA_ROIS = 8
OCR_EARLY_EXIT_SCORE = 0.70
OCR_STREET_EARLY_EXIT_SCORE = 0.48

KNOWN_SIGN_WORDS = [
    "EXIT",
    "STOP",
    "PUSH",
    "PULL",
    "TOILET",
    "OFFICE",
    "STAIRS",
    "LIFT",
    "ENTRANCE",
    "NO ENTRY",
    "DANGER",
    "CAUTION",
    "LEFT",
    "RIGHT",
    "OPEN",
    "CLOSED",
    "FIRE EXIT",
    "FIRST AID",
]

VOWELLESS_OK = frozenset({"STOP", "EXIT", "PUSH", "PULL", "STAIR", "STAIRS", "LIFT"})
CAUTION_DISTANCE_M = 1.2
ALERT_DISTANCE_M = 1.0
STRONG_WARNING_DISTANCE_M = 0.75
VERY_CLOSE_DISTANCE_M = 0.40

# Sign / OCR voice — speak after 1 stable read for classroom reliability
SIGN_CONFIRM_DETECTIONS = 1
SIGN_REPEAT_SECONDS = 5.0

# Camera object voice — only speak when object is confidently identified
CAMERA_OBJECT_CONFIRM_DETECTIONS = 5
CAMERA_OBJECT_REPEAT_SECONDS = 12.0
CAMERA_OBJECT_MAX_ANNOUNCE = 1
CAMERA_OBJECT_VOICE_MIN_CONFIDENCE = 0.55
CAMERA_OBJECT_ALLOWED_SOURCES = ("opencv_dnn", "hailo", "hog")

# LiDAR obstacle voice — confirm faster; do not clear on one CLEAR frame
LIDAR_CONFIRM_SCANS = 4
OBSTACLE_REPEAT_SECONDS = 20.0
VERY_CLOSE_REPEAT_SECONDS = 20.0

# Temporary mute (button / N key) — auto-unmute after this many seconds
VOICE_TEMP_MUTE_SECONDS = 30.0

# SLAM-style LiDAR point-cloud panel (visual only — D6 remains 2D)
SLAM_LIDAR_HISTORY_SCANS = 40
SLAM_LIDAR_ISO_SKEW = 0.35
SLAM_LIDAR_POINT_RADIUS = 2
SLAM_LIDAR_MAX_COLOR_DIST_M = 5.0

# Beginner SLAM — estimate motion between scans, grow world floor-plan map
ENABLE_BEGINNER_SLAM = True
SLAM_MATCH_MAX_POINTS = 48
SLAM_ICP_ITERATIONS = 6
SLAM_MIN_MATCH_POINTS = 12
SLAM_MAX_STEP_XY_M = 0.45
SLAM_MAX_STEP_YAW_DEG = 25.0
SLAM_MIN_STEP_XY_M = 0.02
SLAM_MIN_STEP_YAW_DEG = 1.5
SLAM_POSE_HISTORY = 400
SLAM_ACCEPT_MEAN_ERR_M = 0.35

# Path clear voice — require sustained clear before resetting obstacle alert
CLEAR_CONFIRM_SCANS = 6
CLEAR_REPEAT_SECONDS = 20.0

ESPEAK_SPEED = 95
ESPEAK_AMPLITUDE = 200
ESPEAK_WORD_GAP_MS = 45
ESPEAK_PITCH = 40
ESPEAK_VOICE = "en"
VOICE_WORD_PAUSE_SECONDS = 0.22  # extra pause after each spoken word
WINDOWS_TTS_RATE = -4

USEFUL_OBJECT_LABELS = (
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "bench",
    "chair", "couch", "table", "dining table", "bed", "toilet", "door",
    "backpack", "handbag", "suitcase", "umbrella", "bag", "bottle", "cup",
    "laptop", "cell phone", "book", "clock", "tv", "dog", "cat",
    "traffic light", "stop sign", "potted plant", "obstacle", "sign",
)

# Additional tuned constants.
ZONE_MIN_POINTS = 3

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
HEADER_HEIGHT = 54
FOOTER_HEIGHT = 58
FPS = 30
PIXELS_PER_METER_DEFAULT = 95.0
GRID_RESOLUTION_M = 0.05
OCCUPIED_MIN_HITS = 3
WALL_STRONG_HITS = 6
FREE_MIN_HITS = 2
MIN_WALL_COMPONENT_SIZE = 4
POLAR_BIN_DEG = 1.0
MIN_RANGE_CM = 8
MAX_RANGE_M = 6.0
MAX_RANGE_CM = int(MAX_RANGE_M * 100)

SERIAL_PORT = os.environ.get("LIDAR_PORT", "/dev/ttyUSB0")
SERIAL_BAUD = 230400
SERIAL_TIMEOUT = 0.02

LIDAR_CSV = "team_bravo_lidar_points.csv"
OCCUPANCY_CSV = "team_bravo_occupancy_grid.csv"
CAMERA_DETECTIONS_CSV = "team_bravo_camera_detections.csv"
OCR_CSV = "team_bravo_ocr_text.csv"
DASHBOARD_PNG = "team_bravo_aihat_dashboard.png"

COLOR_BG = (10, 14, 20)
COLOR_PANEL = (18, 24, 32)
COLOR_PANEL_BORDER = (70, 110, 140)
COLOR_TITLE = (120, 220, 255)
COLOR_TEXT = (190, 215, 240)
COLOR_MUTED = (110, 130, 155)
COLOR_GREEN = (56, 199, 99)
COLOR_YELLOW = (245, 211, 59)
COLOR_ORANGE = (245, 145, 60)
COLOR_RED = (235, 70, 70)
COLOR_CYAN = (70, 200, 255)
COLOR_BLUE = (75, 120, 245)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    distance_m: Optional[float]
    source: str
    timestamp: float


@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    timestamp: float
    raw_text: str = ""
    cleaned_text: str = ""
    matched_text: str = ""
    score: float = 0.0


running = True
simulation_paused = False
zones_fullscreen = False
fullscreen = False
focused_panel = 0  # 0=quad, 1=camera, 2=zones, 3=lidar, 4=map
view_status_text = "Quad view"
debug_enabled = False
fusion_enabled = True
voice_enabled = ENABLE_VOICE_ALERTS
voice_muted_until = 0.0
ocr_enabled = ENABLE_OCR
ai_overlay_enabled = True
lidar_enabled = ENABLE_LIDAR
camera_enabled = ENABLE_CAMERA
pixels_per_meter = PIXELS_PER_METER_DEFAULT

data_lock = threading.Lock()
camera_lock = threading.Lock()

latest_scan_points: List[Tuple[float, float, float, float]] = []
latest_polar_points: List[Tuple[float, float]] = []
slam_lidar_history: deque = deque(maxlen=SLAM_LIDAR_HISTORY_SCANS)
occupied_grid: Dict[Tuple[int, int], int] = {}
free_grid: Dict[Tuple[int, int], int] = {}
last_zone_counts = {"front": 0, "left": 0, "right": 0, "back": 0}
direction_distances = {"front": None, "left": None, "right": None, "back": None}

# Beginner SLAM world pose (x_m, y_m, yaw_rad) and previous scan for matching
robot_pose = [0.0, 0.0, 0.0]
prev_slam_scan_xy: List[Tuple[float, float]] = []
pose_history: deque = deque(maxlen=SLAM_POSE_HISTORY)
pose_history.append((0.0, 0.0, 0.0))
slam_scan_count = 0
slam_last_match_err = -1.0
slam_status_text = "SLAM idle"

latest_camera_rgb: Optional[np.ndarray] = None
latest_raw_camera_bgr: Optional[np.ndarray] = None
latest_display_camera_rgb: Optional[np.ndarray] = None
latest_frame_time = 0.0
latest_frame_id = 0
camera_capture_fps = 0.0
camera_drop_count = 0
camera_source = "none"
camera_available = False
camera_error_message = "Camera not initialised"
active_usb_camera_index: Optional[int] = None
prefer_usb_camera = PREFER_USB_CAMERA
_picam_instance = None
_usb_cap = None
_last_camera_retry_time = 0.0
_last_ai_detection_time = 0.0
_last_ocr_scan_time = 0.0
last_successful_frame_time = 0.0
using_explicit_simulation = False
latest_camera_detections: List[Detection] = []
latest_ocr_results: List[OCRResult] = []
last_ocr_text = ""
last_ocr_update_time = 0.0
last_ocr_persist_until = 0.0
last_camera_banner = "No detections"

# OCR debug + voting (v9 sign-aware, any readable text)
ocr_debug_raw = ""
ocr_debug_cleaned = ""
ocr_debug_matched = ""
ocr_debug_confirmed = ""
ocr_vote_counts: Dict[str, int] = {}
ocr_vote_events: List[Tuple[float, str]] = []
ocr_last_vote_time = 0.0
ocr_last_candidates: List[str] = []
ocr_read_now_event = threading.Event()

# Sign / OCR voice state
sign_candidate_text = ""
sign_candidate_count = 0
confirmed_sign_text = ""
last_spoken_sign_text = ""
last_sign_voice_time = 0.0

# Camera object voice state (surroundings: list of (label, direction))
object_candidate_label = ""
object_candidate_direction = ""
object_candidate_key = ""
object_candidate_count = 0
confirmed_object_label = ""
confirmed_object_direction = ""
confirmed_surroundings: List[Tuple[str, str]] = []
last_spoken_object_alert = ""
last_object_voice_time = 0.0

# LiDAR obstacle voice state
raw_lidar_alert = "CLEAR"
lidar_candidate_alert = "CLEAR"
lidar_candidate_count = 0
confirmed_lidar_alert = "CLEAR"
last_spoken_lidar_alert = ""
last_lidar_voice_time = 0.0
lidar_clear_streak = 0

# Global voice tracking
last_spoken_message = ""
last_voice_time = 0.0
last_clear_voice_time = 0.0
last_spoken_was_danger = False
tts_checked = False
tts_executable: Optional[str] = None
tts_backend = "none"
_tts_busy = False
_tts_lock = threading.Lock()
current_voice_process: Optional[subprocess.Popen] = None
ui_button_rects: List[Tuple[pygame.Rect, str]] = []

camera_frame_counter = 0
sim_phase = 0.0

lidar_log_rows: List[List[object]] = []
camera_log_rows: List[List[object]] = []
ocr_log_rows: List[List[object]] = []

AI_HAT_RUNTIME_AVAILABLE = HAILO_AVAILABLE
AI_HAT_STATUS = "OFF"
ai_hat_active = False
ai_hat_device = None
ai_inference_fps = 0.0
latest_detection_source = "none"
camera_fps = 0.0
_last_camera_fps_time = time.time()
_last_camera_fps_count = 0
_last_ai_fps_time = time.time()
_last_ai_fps_count = 0

ai_labels: List[str] = []
dnn_net = None
hog_detector = None


def load_labels(path: str) -> List[str]:
    if not os.path.isfile(path):
        return list(COCO_DEFAULT_LABELS)
    labels = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                txt = line.strip()
                if txt:
                    labels.append(txt)
    except OSError:
        return list(COCO_DEFAULT_LABELS)
    return labels or list(COCO_DEFAULT_LABELS)


def _pick_first_existing(candidates: List[str]) -> Optional[str]:
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _list_model_files(ext: str) -> List[str]:
    if not os.path.isdir(AI_MODELS_DIR):
        return []
    files = []
    try:
        for name in sorted(os.listdir(AI_MODELS_DIR)):
            if name.lower().endswith(ext.lower()):
                files.append(os.path.join(AI_MODELS_DIR, name))
    except OSError:
        return []
    return files


def resolve_ai_model_paths() -> None:
    """
    Auto-detect HEF/ONNX/labels in models/ so Pi downloads like
    yolov8s_h8l.hef work without renaming to yolov8n.hef.
    """
    global AI_MODEL_PATH, AI_DNN_MODEL_PATH, AI_LABELS_PATH
    os.makedirs(AI_MODELS_DIR, exist_ok=True)

    hef_files = _list_model_files(".hef")
    onnx_files = _list_model_files(".onnx")
    label_candidates = [
        os.path.join(AI_MODELS_DIR, "coco_labels.txt"),
        os.path.join(AI_MODELS_DIR, "coco.names"),
        os.path.join(AI_MODELS_DIR, "labels.txt"),
        os.path.join(AI_MODELS_DIR, "coco.txt"),
    ]
    for path in _list_model_files(".txt") + _list_model_files(".names"):
        low = os.path.basename(path).lower()
        if "coco" in low or "label" in low or low.endswith(".names"):
            if path not in label_candidates:
                label_candidates.append(path)

    preferred_hef = [
        os.path.join(AI_MODELS_DIR, "yolov8n.hef"),
        os.path.join(AI_MODELS_DIR, "yolov8s_h8l.hef"),
        os.path.join(AI_MODELS_DIR, "yolov8s.hef"),
        os.path.join(AI_MODELS_DIR, "yolov8m.hef"),
        os.path.join(AI_MODELS_DIR, "yolov11n.hef"),
        os.path.join(AI_MODELS_DIR, "yolov11s.hef"),
    ]
    # Prefer nano/small Hailo-8L names from directory listing
    for path in hef_files:
        low = os.path.basename(path).lower()
        if "yolov8n" in low or "yolo8n" in low:
            preferred_hef.insert(0, path)
        elif "h8l" in low and "yolo" in low:
            preferred_hef.append(path)
        elif "yolo" in low:
            preferred_hef.append(path)
    preferred_hef.extend(hef_files)

    preferred_onnx = [
        os.path.join(AI_MODELS_DIR, "yolov8n.onnx"),
        os.path.join(AI_MODELS_DIR, "yolov8s.onnx"),
    ]
    for path in onnx_files:
        low = os.path.basename(path).lower()
        if "yolov8n" in low:
            preferred_onnx.insert(0, path)
        else:
            preferred_onnx.append(path)
    preferred_onnx.extend(onnx_files)

    hef = _pick_first_existing(preferred_hef)
    onnx = _pick_first_existing(preferred_onnx)
    labels = _pick_first_existing(label_candidates)

    if hef:
        AI_MODEL_PATH = hef
    if onnx:
        AI_DNN_MODEL_PATH = onnx
    if labels:
        AI_LABELS_PATH = labels

    print("AI model paths:")
    print(f"- models dir: {os.path.abspath(AI_MODELS_DIR)}")
    print(f"- HEF:  {AI_MODEL_PATH} ({'OK' if os.path.isfile(AI_MODEL_PATH) else 'MISSING'})")
    print(f"- ONNX: {AI_DNN_MODEL_PATH} ({'OK' if os.path.isfile(AI_DNN_MODEL_PATH) else 'MISSING'})")
    print(f"- Labels: {AI_LABELS_PATH} ({'OK' if os.path.isfile(AI_LABELS_PATH) else 'using built-in COCO-80'})")
    if hef_files or onnx_files:
        print(f"- Found in models/: {[os.path.basename(p) for p in hef_files + onnx_files]}")
    else:
        print("- No .hef/.onnx found yet. Place them in the models/ folder next to this script.")


def list_serial_ports() -> List[str]:
    ports = []
    if list_ports is not None:
        try:
            ports = [p.device for p in list_ports.comports()]
        except Exception:
            ports = []
    if not ports:
        for root in ("/dev",):
            if os.path.isdir(root):
                for name in sorted(os.listdir(root)):
                    if name.startswith("ttyUSB") or name.startswith("ttyACM"):
                        ports.append(os.path.join(root, name))
    return ports


def open_serial_port() -> Optional[object]:
    if serial is None:
        return None
    candidates = [SERIAL_PORT] + [p for p in list_serial_ports() if p != SERIAL_PORT]
    for p in candidates:
        try:
            conn = serial.Serial(p, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
            print(f"LiDAR connected: {p} @ {SERIAL_BAUD}")
            return conn
        except Exception:
            continue
    print("WARNING: LiDAR serial unavailable. Falling back to simulated LiDAR.")
    return None


def read_packet(connection) -> Optional[bytes]:
    """Search for AA55 header and return one complete packet."""
    while running:
        try:
            b = connection.read(1)
        except Exception:
            return None
        if not b:
            return None
        if b[0] == 0xAA:
            second = connection.read(1)
            if second and second[0] == 0x55:
                header_rest = connection.read(8)
                if len(header_rest) != 8:
                    return None
                lsn = header_rest[1]
                if lsn <= 0 or lsn > 100:
                    return None
                sample_data = connection.read(lsn * 2)
                if len(sample_data) != lsn * 2:
                    return None
                return bytes([0xAA, 0x55]) + header_rest + sample_data
    return None


def parse_packet(packet: Optional[bytes]) -> List[Tuple[float, float]]:
    """Return list of (angle_deg, distance_cm) from AA55 packet."""
    if packet is None or len(packet) < 10:
        return []
    lsn = packet[3]
    if lsn <= 0:
        return []
    fsa_raw = struct.unpack_from("<H", packet, 4)[0]
    lsa_raw = struct.unpack_from("<H", packet, 6)[0]
    start_angle = (fsa_raw >> 1) / 64.0
    end_angle = (lsa_raw >> 1) / 64.0
    angle_diff = end_angle - start_angle
    if angle_diff < -180:
        angle_diff += 360
    elif angle_diff > 180:
        angle_diff -= 360
    points = []
    offset = 10
    for i in range(lsn):
        if offset + 2 > len(packet):
            break
        raw_sample = struct.unpack_from("<H", packet, offset)[0]
        offset += 2
        distance_cm = (raw_sample / 4.0) / 10.0
        angle_deg = start_angle + (angle_diff * i / (lsn - 1) if lsn > 1 else 0.0)
        angle_deg %= 360.0
        if MIN_RANGE_CM <= distance_cm <= MAX_RANGE_CM:
            points.append((angle_deg, distance_cm))
    return points


def smooth_scan_polar(scan_polar: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    bins = {}
    for a, d in scan_polar:
        k = int(round(a / POLAR_BIN_DEG))
        bins.setdefault(k, []).append(d)
    out = []
    for k, ds in bins.items():
        ds.sort()
        out.append(((k * POLAR_BIN_DEG) % 360.0, ds[len(ds) // 2]))
    return out


def polar_to_xy(angle_deg: float, distance_cm: float) -> Tuple[float, float, float, float]:
    distance_m = distance_cm / 100.0
    r = math.radians(angle_deg)
    x = distance_m * math.cos(r)
    y = distance_m * math.sin(r)
    return x, y, distance_m, angle_deg


def grid_index(x_m: float, y_m: float) -> Tuple[int, int]:
    return round(x_m / GRID_RESOLUTION_M), round(y_m / GRID_RESOLUTION_M)


def bresenham_line_cells(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    cells = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


def carve_ray_to_obstacle(x_m: float, y_m: float) -> None:
    """Egocentric carve (robot at origin) — kept for non-SLAM fallback."""
    end_ix, end_iy = grid_index(x_m, y_m)
    line = bresenham_line_cells(0, 0, end_ix, end_iy)
    for i, cell in enumerate(line):
        if i == len(line) - 1:
            occupied_grid[cell] = occupied_grid.get(cell, 0) + 1
        elif occupied_grid.get(cell, 0) < OCCUPIED_MIN_HITS:
            free_grid[cell] = free_grid.get(cell, 0) + 1


def carve_ray_world(ox: float, oy: float, wx: float, wy: float) -> None:
    """Carve free space and occupied endpoint in world frame for floor-plan map."""
    start = grid_index(ox, oy)
    end = grid_index(wx, wy)
    line = bresenham_line_cells(start[0], start[1], end[0], end[1])
    for i, cell in enumerate(line):
        if i == len(line) - 1:
            occupied_grid[cell] = occupied_grid.get(cell, 0) + 1
        elif occupied_grid.get(cell, 0) < OCCUPIED_MIN_HITS:
            free_grid[cell] = free_grid.get(cell, 0) + 1


def subsample_xy(points_xy: List[Tuple[float, float]], max_n: int) -> List[Tuple[float, float]]:
    if len(points_xy) <= max_n:
        return list(points_xy)
    step = max(1, len(points_xy) // max_n)
    return points_xy[::step][:max_n]


def transform_xy(
    points_xy: List[Tuple[float, float]], dx: float, dy: float, dtheta: float
) -> List[Tuple[float, float]]:
    c, s = math.cos(dtheta), math.sin(dtheta)
    out = []
    for x, y in points_xy:
        out.append((c * x - s * y + dx, s * x + c * y + dy))
    return out


def nearest_mean_error(
    src: List[Tuple[float, float]], dst: List[Tuple[float, float]]
) -> float:
    if not src or not dst:
        return 999.0
    total = 0.0
    for x, y in src:
        best = 1e9
        for u, v in dst:
            d2 = (x - u) * (x - u) + (y - v) * (y - v)
            if d2 < best:
                best = d2
        total += math.sqrt(best)
    return total / len(src)


def estimate_motion_icp(
    prev_xy: List[Tuple[float, float]], curr_xy: List[Tuple[float, float]]
) -> Tuple[float, float, float, float]:
    """
    Estimate robot motion (dx, dy, dtheta) in previous robot frame so that
    transforming curr toward prev aligns scans. Returns (dx, dy, dtheta, mean_err).
    """
    prev = subsample_xy(prev_xy, SLAM_MATCH_MAX_POINTS)
    curr = subsample_xy(curr_xy, SLAM_MATCH_MAX_POINTS)
    if len(prev) < SLAM_MIN_MATCH_POINTS or len(curr) < SLAM_MIN_MATCH_POINTS:
        return 0.0, 0.0, 0.0, 999.0

    # Coarse yaw search, then refine translation
    best = (0.0, 0.0, 0.0, 999.0)
    for yaw_deg in range(-18, 19, 3):
        dth = math.radians(yaw_deg)
        rotated = transform_xy(curr, 0.0, 0.0, dth)
        # centroid shift as translation seed
        px = sum(p[0] for p in prev) / len(prev)
        py = sum(p[1] for p in prev) / len(prev)
        cx = sum(p[0] for p in rotated) / len(rotated)
        cy = sum(p[1] for p in rotated) / len(rotated)
        dx0, dy0 = px - cx, py - cy
        aligned = [(x + dx0, y + dy0) for x, y in rotated]
        err = nearest_mean_error(aligned, prev)
        if err < best[3]:
            best = (dx0, dy0, dth, err)

    dx, dy, dth, err = best
    # Fine ICP-like iterations around best
    for _ in range(SLAM_ICP_ITERATIONS):
        transformed = transform_xy(curr, dx, dy, dth)
        # match each transformed curr point to nearest prev
        pairs_src = []
        pairs_dst = []
        for x, y in transformed:
            best_d = 1e9
            bu = bv = 0.0
            for u, v in prev:
                d2 = (x - u) * (x - u) + (y - v) * (y - v)
                if d2 < best_d:
                    best_d = d2
                    bu, bv = u, v
            if best_d < (0.6 * 0.6):
                pairs_src.append((x, y))
                pairs_dst.append((bu, bv))
        if len(pairs_src) < SLAM_MIN_MATCH_POINTS:
            break
        # average residual translation
        mx = sum(d[0] - s[0] for s, d in zip(pairs_src, pairs_dst)) / len(pairs_src)
        my = sum(d[1] - s[1] for s, d in zip(pairs_src, pairs_dst)) / len(pairs_src)
        dx += mx
        dy += my
        # small yaw from cross product of matched vectors around centroids
        scx = sum(s[0] for s in pairs_src) / len(pairs_src)
        scy = sum(s[1] for s in pairs_src) / len(pairs_src)
        dcx = sum(d[0] for d in pairs_dst) / len(pairs_dst)
        dcy = sum(d[1] for d in pairs_dst) / len(pairs_dst)
        num = den = 0.0
        for s, d in zip(pairs_src, pairs_dst):
            sx, sy = s[0] - scx, s[1] - scy
            dxp, dyp = d[0] - dcx, d[1] - dcy
            num += sx * dyp - sy * dxp
            den += sx * dxp + sy * dyp
        if den != 0.0:
            dth += math.atan2(num, den) * 0.5
        err = nearest_mean_error(transform_xy(curr, dx, dy, dth), prev)

    return dx, dy, dth, err


def robot_to_world(x_r: float, y_r: float, pose: List[float]) -> Tuple[float, float]:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return pose[0] + c * x_r - s * y_r, pose[1] + s * x_r + c * y_r


def reset_slam_map() -> None:
    global occupied_grid, free_grid, robot_pose, prev_slam_scan_xy, pose_history
    global slam_scan_count, slam_last_match_err, slam_status_text
    with data_lock:
        occupied_grid = {}
        free_grid = {}
        robot_pose = [0.0, 0.0, 0.0]
        prev_slam_scan_xy = []
        pose_history.clear()
        pose_history.append((0.0, 0.0, 0.0))
        slam_scan_count = 0
        slam_last_match_err = -1.0
        slam_status_text = "SLAM map reset"
    print("SLAM floor-plan map reset (pose at origin)")


def process_scan(scan_polar: List[Tuple[float, float]]) -> None:
    global latest_polar_points, latest_scan_points, prev_slam_scan_xy, slam_scan_count
    global slam_last_match_err, slam_status_text
    smoothed = smooth_scan_polar(scan_polar)
    xy_points: List[Tuple[float, float, float, float]] = []
    curr_xy: List[Tuple[float, float]] = []
    ts = round(time.time(), 3)
    for a, d_cm in smoothed:
        x, y, d_m, a_deg = polar_to_xy(a, d_cm)
        xy_points.append((x, y, d_m, a_deg))
        curr_xy.append((x, y))
        lidar_log_rows.append([ts, f"{a_deg:.2f}", f"{d_cm:.2f}", f"{x:.3f}", f"{y:.3f}", f"{d_m:.3f}"])

    with data_lock:
        latest_polar_points = smoothed
        latest_scan_points = xy_points
        if xy_points:
            slam_lidar_history.append(list(xy_points))

        if not ENABLE_BEGINNER_SLAM:
            for x, y, _d, _a in xy_points:
                carve_ray_to_obstacle(x, y)
            return

        # First scan seeds the map at origin
        if not prev_slam_scan_xy:
            for x, y, _d, _a in xy_points:
                wx, wy = robot_to_world(x, y, robot_pose)
                carve_ray_world(robot_pose[0], robot_pose[1], wx, wy)
            prev_slam_scan_xy = curr_xy
            pose_history.append((robot_pose[0], robot_pose[1], robot_pose[2]))
            slam_scan_count = 1
            slam_status_text = "SLAM mapping..."
            return

        dx, dy, dth, err = estimate_motion_icp(prev_slam_scan_xy, curr_xy)
        slam_last_match_err = err
        step = math.hypot(dx, dy)
        yaw_deg = abs(math.degrees(dth))
        accepted = (
            err < SLAM_ACCEPT_MEAN_ERR_M
            and step <= SLAM_MAX_STEP_XY_M
            and yaw_deg <= SLAM_MAX_STEP_YAW_DEG
        )
        moved = step >= SLAM_MIN_STEP_XY_M or yaw_deg >= SLAM_MIN_STEP_YAW_DEG

        if accepted and moved:
            # Motion of robot in previous frame
            c, s = math.cos(robot_pose[2]), math.sin(robot_pose[2])
            robot_pose[0] += c * dx - s * dy
            robot_pose[1] += s * dx + c * dy
            robot_pose[2] = (robot_pose[2] + dth + math.pi) % (2 * math.pi) - math.pi
            pose_history.append((robot_pose[0], robot_pose[1], robot_pose[2]))
            slam_status_text = f"SLAM move {step:.2f}m err={err:.2f}"
        elif accepted:
            slam_status_text = f"SLAM hold err={err:.2f}"
        else:
            slam_status_text = f"SLAM reject err={err:.2f}"

        # Always integrate current scan into world map at current pose
        for x, y, _d, _a in xy_points:
            wx, wy = robot_to_world(x, y, robot_pose)
            carve_ray_world(robot_pose[0], robot_pose[1], wx, wy)

        prev_slam_scan_xy = curr_xy
        slam_scan_count += 1


def simulated_lidar_scan() -> List[Tuple[float, float]]:
    global sim_phase
    if not simulation_paused:
        sim_phase += 0.04
    scan = []
    for a in range(360):
        wall_m = 2.5 + 0.4 * math.sin(math.radians(a * 2))
        obstacle = 99.0
        # front "person", right "chair", and occasional close obstacle
        if -15 <= ((a + 180) % 360) - 180 <= 15:
            obstacle = min(obstacle, 1.0 + 0.1 * math.sin(sim_phase * 2))
        if 60 <= a <= 95:
            obstacle = min(obstacle, 1.35 + 0.15 * math.cos(sim_phase * 1.7))
        if 300 <= a <= 325:
            obstacle = min(obstacle, 0.8 + 0.1 * math.sin(sim_phase * 1.3))
        dist_m = min(wall_m, obstacle)
        scan.append((float(a), dist_m * 100.0))
    return scan


def lidar_thread_fn() -> None:
    conn = open_serial_port() if (ENABLE_LIDAR and not SIMULATED_MODE and lidar_enabled) else None
    while running:
        if not lidar_enabled:
            time.sleep(0.05)
            continue
        if conn is not None:
            packet = read_packet(conn)
            pts = parse_packet(packet)
            if pts:
                process_scan(pts)
            else:
                time.sleep(0.005)
        else:
            process_scan(simulated_lidar_scan())
            time.sleep(0.04)


def _zone_nearest(points, pred):
    nearest = None
    for x_m, y_m, d_m, _a in points:
        if pred(x_m, y_m, d_m):
            nearest = d_m if nearest is None else min(nearest, d_m)
    return nearest


def detect_obstacles_for_blind_user(points):
    zc = {"front": 0, "left": 0, "right": 0, "back": 0, "vc_front": 0, "vc_left": 0, "vc_right": 0, "vc_back": 0}
    for x_m, y_m, d_m, _a in points:
        if x_m > 0 and abs(y_m) <= 0.45 and d_m <= ALERT_DISTANCE_M:
            zc["front"] += 1
        if x_m < 0 and abs(y_m) <= 0.45 and d_m <= 0.8:
            zc["back"] += 1
        if y_m < -0.35 and -0.3 <= x_m <= 1.2 and d_m <= ALERT_DISTANCE_M:
            zc["left"] += 1
        if y_m > 0.35 and -0.3 <= x_m <= 1.2 and d_m <= ALERT_DISTANCE_M:
            zc["right"] += 1
        if d_m <= VERY_CLOSE_DISTANCE_M:
            if x_m > 0 and abs(y_m) <= 0.45:
                zc["vc_front"] += 1
            elif x_m < 0 and abs(y_m) <= 0.45:
                zc["vc_back"] += 1
            elif y_m < -0.25:
                zc["vc_left"] += 1
            elif y_m > 0.25:
                zc["vc_right"] += 1

    nf = _zone_nearest(points, lambda x, y, d: x > 0 and abs(y) <= 0.45 and d <= ALERT_DISTANCE_M)
    nl = _zone_nearest(points, lambda x, y, d: y < -0.35 and -0.3 <= x <= 1.2 and d <= ALERT_DISTANCE_M)
    nr = _zone_nearest(points, lambda x, y, d: y > 0.35 and -0.3 <= x <= 1.2 and d <= ALERT_DISTANCE_M)
    nb = _zone_nearest(points, lambda x, y, d: x < 0 and abs(y) <= 0.45 and d <= 0.8)

    if zc["vc_front"] >= ZONE_MIN_POINTS:
        return "VERY_CLOSE_FRONT", nf, zc
    if zc["vc_left"] >= ZONE_MIN_POINTS:
        return "VERY_CLOSE_LEFT", nl, zc
    if zc["vc_right"] >= ZONE_MIN_POINTS:
        return "VERY_CLOSE_RIGHT", nr, zc
    if zc["vc_back"] >= ZONE_MIN_POINTS:
        return "VERY_CLOSE_BACK", nb, zc
    if zc["front"] >= ZONE_MIN_POINTS and nf is not None and nf <= STRONG_WARNING_DISTANCE_M:
        return "STRONG_FRONT", nf, zc
    if zc["left"] >= ZONE_MIN_POINTS and nl is not None and nl <= STRONG_WARNING_DISTANCE_M:
        return "STRONG_LEFT", nl, zc
    if zc["right"] >= ZONE_MIN_POINTS and nr is not None and nr <= STRONG_WARNING_DISTANCE_M:
        return "STRONG_RIGHT", nr, zc
    if zc["front"] >= ZONE_MIN_POINTS:
        return "NORMAL_FRONT", nf, zc
    if zc["left"] >= ZONE_MIN_POINTS and zc["right"] >= ZONE_MIN_POINTS:
        return "BOTH_SIDES", min(nl or 99.0, nr or 99.0), zc
    if zc["left"] >= ZONE_MIN_POINTS:
        return "NORMAL_LEFT", nl, zc
    if zc["right"] >= ZONE_MIN_POINTS:
        return "NORMAL_RIGHT", nr, zc
    if zc["back"] >= ZONE_MIN_POINTS:
        return "BACK", nb, zc
    return "CLEAR", None, zc


def find_tts_executable() -> Optional[str]:
    for name in ("espeak-ng", "espeak"):
        p = shutil.which(name)
        if p:
            return p
    for p in ("/usr/bin/espeak-ng", "/usr/bin/espeak"):
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def init_voice_system() -> str:
    """Detect TTS backend. Returns backend name."""
    global tts_checked, tts_executable, tts_backend
    tts_executable = find_tts_executable()
    tts_checked = True
    if tts_executable:
        tts_backend = "espeak"
        print(f"Voice OK: espeak ({tts_executable})")
    elif sys.platform == "win32":
        tts_backend = "windows"
        print("Voice OK: Windows System.Speech")
    else:
        tts_backend = "console"
        print("Voice fallback: console beep (install espeak-ng on Pi)")
    return tts_backend


def check_tts() -> bool:
    global tts_checked
    if not tts_checked:
        init_voice_system()
    return tts_backend != "console"


def stop_current_voice() -> None:
    global current_voice_process, _tts_busy
    if current_voice_process is not None and current_voice_process.poll() is None:
        try:
            current_voice_process.terminate()
            current_voice_process.wait(timeout=0.4)
        except Exception:
            try:
                current_voice_process.kill()
            except Exception:
                pass
    current_voice_process = None
    with _tts_lock:
        _tts_busy = False


def _run_tts_windows(text: str) -> bool:
    global _tts_busy

    def _worker() -> None:
        global _tts_busy
        with _tts_lock:
            _tts_busy = True
        try:
            words = [w for w in re.split(r"\s+", text.strip()) if w]
            if not words:
                return
            # Speak one word at a time so each word is clear before the next
            parts = []
            for w in words:
                safe = w.replace("'", "''")
                parts.append(f"$s.Speak('{safe}'); Start-Sleep -Milliseconds {int(VOICE_WORD_PAUSE_SECONDS * 1000)}")
            ps = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Rate = {WINDOWS_TTS_RATE}; "
                "$s.Volume = 100; "
                + " ".join(parts)
            )
            timeout_s = max(25, 3 + len(words) * 2)
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
                check=False,
            )
        except Exception as exc:
            print(f"Windows TTS error: {exc}")
        finally:
            with _tts_lock:
                _tts_busy = False

    threading.Thread(target=_worker, daemon=True).start()
    return True


def clarify_speech_text(text: str) -> str:
    """Normalize speech text; keep wording simple for clear word-by-word reading."""
    t = " ".join(text.replace(",", " ").replace(".", " ").split())
    t = re.sub(r"\bAve\b", "Avenue", t, flags=re.IGNORECASE)
    t = re.sub(r"\bSt\b", "Street", t, flags=re.IGNORECASE)
    t = re.sub(r"\bRd\b", "Road", t, flags=re.IGNORECASE)
    return t


def _speak_words_espeak(words: List[str]) -> bool:
    """Speak each word fully before starting the next (clearest assistive mode)."""
    global current_voice_process
    if not tts_executable or not words:
        return False
    for word in words:
        args = [
            tts_executable,
            "-v", ESPEAK_VOICE,
            "-s", str(ESPEAK_SPEED),
            "-a", str(ESPEAK_AMPLITUDE),
            "-g", str(ESPEAK_WORD_GAP_MS),
            "-p", str(ESPEAK_PITCH),
            word,
        ]
        try:
            current_voice_process = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            current_voice_process.wait(timeout=8)
        except Exception:
            print("\a", end="", flush=True)
            return False
        finally:
            current_voice_process = None
        time.sleep(VOICE_WORD_PAUSE_SECONDS)
    return True


def run_tts(text: str, force: bool = False) -> bool:
    """Speak text one word at a time. force=True bypasses mute/off (for tests)."""
    global current_voice_process, _tts_busy
    if not force and not is_voice_output_allowed():
        return False
    if not tts_checked:
        init_voice_system()

    spoken = clarify_speech_text(text)
    words = [w for w in spoken.split() if w]
    if not words:
        return False

    if tts_backend == "espeak" and tts_executable:
        # Run word-by-word in a worker so UI stays responsive
        def _worker() -> None:
            global _tts_busy
            with _tts_lock:
                _tts_busy = True
            try:
                _speak_words_espeak(words)
            finally:
                with _tts_lock:
                    _tts_busy = False

        threading.Thread(target=_worker, daemon=True).start()
        return True

    if tts_backend == "windows":
        return _run_tts_windows(spoken)

    print(f"[VOICE] {' | '.join(words)}")
    print("\a", end="", flush=True)
    return True


def is_voice_output_allowed() -> bool:
    """False when voice is off or temporarily muted."""
    if not voice_enabled:
        return False
    if time.time() < voice_muted_until:
        return False
    return True


def voice_mute_remaining_seconds() -> float:
    return max(0.0, voice_muted_until - time.time())


def mute_voice_temporarily(seconds: Optional[float] = None) -> None:
    """Temporarily silence alerts; auto-unmutes after VOICE_TEMP_MUTE_SECONDS."""
    global voice_muted_until
    secs = VOICE_TEMP_MUTE_SECONDS if seconds is None else float(seconds)
    voice_muted_until = time.time() + secs
    stop_current_voice()
    print(f"Voice muted for {secs:.0f}s (press Mute again or V to manage)")
    run_tts(f"Muted for {int(secs)} seconds", force=True)


def unmute_voice_temporary() -> None:
    global voice_muted_until
    if voice_muted_until <= 0 and voice_mute_remaining_seconds() <= 0:
        return
    voice_muted_until = 0.0
    print("Temporary mute cleared")
    run_tts("Voice unmuted", force=True)


def toggle_temporary_mute() -> None:
    if voice_mute_remaining_seconds() > 0:
        unmute_voice_temporary()
    else:
        mute_voice_temporarily()


def is_voice_speaking() -> bool:
    with _tts_lock:
        if _tts_busy:
            return True
    return current_voice_process is not None and current_voice_process.poll() is None


def test_voice() -> None:
    stop_current_voice()
    ok = run_tts("Team Bravo vision assistant ready. Voice test OK.", force=True)
    print(f"Voice test {'OK' if ok else 'FAILED'}")


def test_voice_sign() -> None:
    stop_current_voice()
    run_tts("Sign says Exit", force=True)


def test_voice_obstacle() -> None:
    stop_current_voice()
    run_tts("Obstacle ahead", force=True)


def test_voice_stop() -> None:
    stop_current_voice()
    run_tts("Stop. Obstacle very close ahead", force=True)


def toggle_voice_enabled() -> None:
    global voice_enabled, voice_muted_until
    voice_enabled = not voice_enabled
    if not voice_enabled:
        stop_current_voice()
        voice_muted_until = 0.0
    else:
        voice_muted_until = 0.0
    print(f"Voice {'ON' if voice_enabled else 'OFF'}")
    run_tts(f"Voice {'on' if voice_enabled else 'off'}", force=True)


def print_voice_settings() -> None:
    print("Voice settings:")
    print(f"- Sign confirm detections: {SIGN_CONFIRM_DETECTIONS}")
    print(f"- Sign repeat: {SIGN_REPEAT_SECONDS:.0f} seconds")
    print(f"- LiDAR confirm scans: {LIDAR_CONFIRM_SCANS}")
    print(f"- Obstacle repeat: {OBSTACLE_REPEAT_SECONDS:.0f} seconds")
    print(f"- Very close repeat: {VERY_CLOSE_REPEAT_SECONDS:.0f} seconds")
    print(f"- Object confirm: {CAMERA_OBJECT_CONFIRM_DETECTIONS}")
    print(f"- Object min confidence: {CAMERA_OBJECT_VOICE_MIN_CONFIDENCE:.2f}")
    print(f"- Object repeat: {CAMERA_OBJECT_REPEAT_SECONDS:.0f} seconds")
    print(f"- Voice: word-by-word pause {VOICE_WORD_PAUSE_SECONDS:.2f}s speed={ESPEAK_SPEED}")
    print(f"- Temporary mute: {VOICE_TEMP_MUTE_SECONDS:.0f} seconds (button Mute / N)")
    print(f"- Alert distance: {ALERT_DISTANCE_M:.1f} m")
    print(f"- Strong warning distance: {STRONG_WARNING_DISTANCE_M:.2f} m")
    print(f"- Very close distance: {VERY_CLOSE_DISTANCE_M:.2f} m")


def bbox_direction(bbox: Tuple[int, int, int, int], frame_w: int) -> str:
    x1, _y1, x2, _y2 = bbox
    cx = (x1 + x2) / 2.0
    third = frame_w / 3.0
    if cx < third:
        return "LEFT"
    if cx > 2.0 * third:
        return "RIGHT"
    return "FRONT"


def normalize_object_label(label: str) -> str:
    low = label.lower().strip()
    mapping = (
        ("stop sign", "Stop sign"),
        ("traffic light", "Traffic light"),
        ("dining table", "Table"),
        ("cell phone", "Phone"),
        ("potted plant", "Plant"),
        ("person", "Person"),
        ("bicycle", "Bicycle"),
        ("motorcycle", "Motorcycle"),
        ("truck", "Truck"),
        ("bus", "Bus"),
        ("car", "Car"),
        ("bench", "Bench"),
        ("chair", "Chair"),
        ("couch", "Couch"),
        ("table", "Table"),
        ("bed", "Bed"),
        ("toilet", "Toilet"),
        ("door", "Door"),
        ("backpack", "Bag"),
        ("handbag", "Bag"),
        ("suitcase", "Suitcase"),
        ("umbrella", "Umbrella"),
        ("bag", "Bag"),
        ("bottle", "Bottle"),
        ("cup", "Cup"),
        ("laptop", "Laptop"),
        ("book", "Book"),
        ("clock", "Clock"),
        ("tv", "TV"),
        ("dog", "Dog"),
        ("cat", "Cat"),
        ("sign", "Sign"),
        ("obstacle", "Obstacle"),
    )
    for key, nice in mapping:
        if key in low:
            return nice
    return label.replace("_", " ").strip().title() or "Object"


def lidar_alert_direction(alert: str) -> str:
    if "LEFT" in alert:
        return "left"
    if "RIGHT" in alert:
        return "right"
    if "BACK" in alert:
        return "behind"
    return "ahead"


def pick_best_camera_object(
    detections: List[Detection], frame_w: int
) -> Tuple[str, str]:
    surroundings = pick_surrounding_objects(detections, frame_w, max_n=1)
    if surroundings:
        return surroundings[0]
    return "", ""


def pick_surrounding_objects(
    detections: List[Detection], frame_w: int, max_n: int = CAMERA_OBJECT_MAX_ANNOUNCE
) -> List[Tuple[str, str]]:
    """Pick only confidently identified useful objects for voice announcement."""
    if not detections:
        return []
    scored: List[Tuple[float, str, str]] = []
    for det in detections:
        # Certainty gate: high confidence + trusted detector source
        if det.confidence < CAMERA_OBJECT_VOICE_MIN_CONFIDENCE:
            continue
        if det.source not in CAMERA_OBJECT_ALLOWED_SOURCES:
            continue
        low = det.label.lower()
        if not any(p in low for p in USEFUL_OBJECT_LABELS):
            continue
        # Contour-style generic "obstacle" is never certain enough to speak
        if low.strip() in ("obstacle", "object"):
            continue
        direction = bbox_direction(det.bbox, frame_w)
        x1, y1, x2, y2 = det.bbox
        area = max(1, (x2 - x1) * (y2 - y1))
        # Reject tiny uncertain boxes
        if area < 1800:
            continue
        cx = (x1 + x2) / 2.0
        centre_bonus = 1.0 - abs(cx - frame_w / 2.0) / max(1.0, frame_w / 2.0)
        score = det.confidence * 3.0 + min(1.2, area / 50000.0) + 0.35 * centre_bonus
        if "person" in low:
            score += 0.6
        scored.append((score, det.label, direction))
    scored.sort(key=lambda t: t[0], reverse=True)

    chosen: List[Tuple[str, str]] = []
    used_dirs: set = set()
    used_labels: set = set()
    for _score, label, direction in scored:
        nice = normalize_object_label(label)
        if direction in used_dirs or nice in used_labels:
            continue
        chosen.append((label, direction))
        used_dirs.add(direction)
        used_labels.add(nice)
        if len(chosen) >= max_n:
            return chosen
    return chosen


def update_sign_voice_state_machine(ocr_raw_text: str) -> None:
    global sign_candidate_text, sign_candidate_count, confirmed_sign_text, last_ocr_text
    cleaned = match_known_sign_text(ocr_raw_text) or clean_ocr_text(ocr_raw_text)
    if not cleaned or not is_valid_sign_text(cleaned):
        return
    last_ocr_text = cleaned
    if cleaned == sign_candidate_text:
        sign_candidate_count += 1
    else:
        sign_candidate_text = cleaned
        sign_candidate_count = 1
    if sign_candidate_count >= SIGN_CONFIRM_DETECTIONS:
        confirmed_sign_text = sign_candidate_text


def update_object_voice_state_machine(detections: List[Detection], frame_w: int) -> None:
    global object_candidate_label, object_candidate_direction, object_candidate_key
    global object_candidate_count, confirmed_object_label, confirmed_object_direction
    global confirmed_surroundings
    surroundings = pick_surrounding_objects(detections, frame_w)
    if not surroundings:
        if object_candidate_key or confirmed_surroundings:
            object_candidate_label = ""
            object_candidate_direction = ""
            object_candidate_key = ""
            object_candidate_count = 0
            confirmed_object_label = ""
            confirmed_object_direction = ""
            confirmed_surroundings = []
        return

    key = ";".join(f"{normalize_object_label(l)}:{d}" for l, d in surroundings)
    if key == object_candidate_key:
        object_candidate_count += 1
    else:
        object_candidate_key = key
        object_candidate_label = surroundings[0][0]
        object_candidate_direction = surroundings[0][1]
        object_candidate_count = 1

    if object_candidate_count >= CAMERA_OBJECT_CONFIRM_DETECTIONS:
        confirmed_surroundings = list(surroundings)
        confirmed_object_label = surroundings[0][0]
        confirmed_object_direction = surroundings[0][1]


def update_lidar_voice_state_machine(raw_alert: str) -> None:
    """Confirm obstacles after several scans; only clear after sustained CLEAR."""
    global raw_lidar_alert, lidar_candidate_alert, lidar_candidate_count
    global confirmed_lidar_alert, lidar_clear_streak
    raw_lidar_alert = raw_alert

    if raw_alert == "CLEAR":
        lidar_clear_streak += 1
        if lidar_clear_streak >= CLEAR_CONFIRM_SCANS:
            if confirmed_lidar_alert != "CLEAR":
                print("LiDAR path clear confirmed")
            confirmed_lidar_alert = "CLEAR"
            lidar_candidate_alert = "CLEAR"
            lidar_candidate_count = 0
        return

    lidar_clear_streak = 0
    if raw_alert == lidar_candidate_alert:
        lidar_candidate_count += 1
    else:
        lidar_candidate_alert = raw_alert
        lidar_candidate_count = 1

    if lidar_candidate_count >= LIDAR_CONFIRM_SCANS:
        if confirmed_lidar_alert != lidar_candidate_alert:
            print(f"LiDAR obstacle confirmed: {lidar_candidate_alert}")
        confirmed_lidar_alert = lidar_candidate_alert


def confirm_sign_text(text: str, speak_now: bool = True) -> None:
    """Mark sign as confirmed and optionally speak immediately."""
    global confirmed_sign_text, ocr_debug_confirmed, sign_candidate_text, sign_candidate_count
    if not text:
        return
    confirmed_sign_text = text
    ocr_debug_confirmed = text
    sign_candidate_text = text
    sign_candidate_count = SIGN_CONFIRM_DETECTIONS
    print(f"OCR CONFIRMED: {text}")
    if speak_now and is_voice_output_allowed():
        now = time.time()
        if (
            text != last_spoken_sign_text
            or (now - last_sign_voice_time) >= OCR_VOICE_REPEAT_SECONDS
        ):
            speak_chosen_message(build_sign_speech(text), "sign", text, True)


def matching_camera_object(
    lidar_alert: str, detections: List[Detection], frame_w: int
) -> Optional[str]:
    if not fusion_enabled or not detections:
        return None
    if "LEFT" in lidar_alert:
        want_dir = "LEFT"
    elif "RIGHT" in lidar_alert:
        want_dir = "RIGHT"
    else:
        want_dir = "FRONT"
    label, direction = pick_best_camera_object(detections, frame_w)
    if not label:
        return None
    if direction == want_dir:
        return label
    if want_dir == "FRONT" and direction == "FRONT":
        return label
    return None


def build_object_speech(label: str, direction: str) -> str:
    obj = normalize_object_label(label)
    if direction == "LEFT":
        return f"{obj} on your left"
    if direction == "RIGHT":
        return f"{obj} on your right"
    return f"{obj} ahead"


def build_surroundings_speech(items: List[Tuple[str, str]]) -> str:
    """One confident object at a time for clearer speech."""
    if not items:
        return ""
    return build_object_speech(items[0][0], items[0][1])


def build_lidar_speech(lidar_alert: str, object_label: Optional[str] = None) -> str:
    obj = normalize_object_label(object_label) if object_label else None
    direction = lidar_alert_direction(lidar_alert)

    if lidar_alert == "BOTH_SIDES":
        return "Obstacles on both sides"

    if lidar_alert.startswith("VERY_CLOSE_"):
        if obj and obj != "Object":
            if direction == "ahead":
                return f"Stop. {obj} very close ahead"
            return f"Stop. {obj} very close on your {direction}"
        if direction == "ahead":
            return "Stop. Obstacle very close ahead"
        if direction == "behind":
            return "Stop. Obstacle very close behind you"
        return f"Stop. Obstacle very close on your {direction}"

    if lidar_alert.startswith("STRONG_"):
        if obj and obj != "Object":
            if direction == "ahead":
                return f"Careful. {obj} ahead"
            return f"Careful. {obj} on your {direction}"
        if direction == "ahead":
            return "Careful. Obstacle ahead"
        return f"Careful. Obstacle on your {direction}"

    if obj and obj != "Object":
        if direction == "ahead":
            return f"{obj} ahead"
        if direction == "behind":
            return f"{obj} behind you"
        return f"{obj} on your {direction}"

    if lidar_alert == "BACK":
        return "Obstacle behind you"
    if direction == "ahead":
        return "Obstacle ahead"
    return f"Obstacle on your {direction}"


def build_sign_speech(text: str) -> str:
    compact = text.replace(" ", "")
    if compact.isdigit():
        return "Number " + " ".join(list(compact))
    alpha = sum(1 for c in compact if c.isalpha())
    digit = sum(1 for c in compact if c.isdigit())
    if digit >= 1 and alpha <= 2:
        return "Number " + " ".join(list(compact))
    words = text.title().split()
    expanded = []
    for w in words:
        low = w.lower()
        if low == "ave":
            expanded.append("Avenue")
        elif low == "st":
            expanded.append("Street")
        elif low == "rd":
            expanded.append("Road")
        else:
            expanded.append(w)
    return "Sign says " + " ".join(expanded)


def choose_voice_message(
    detections: List[Detection], frame_w: int
) -> Optional[Tuple[str, str, str, bool]]:
    """
    Return one voice message: (spoken_text, category_key, voice_track_key, interrupt).
    Priority: OCR sign > VERY_CLOSE > STRONG > lidar normal > camera objects > path clear.
    """
    now = time.time()

    # Sign text has priority over all obstacle warnings when due to speak
    if confirmed_sign_text:
        if (
            confirmed_sign_text != last_spoken_sign_text
            or (now - last_sign_voice_time) >= OCR_VOICE_REPEAT_SECONDS
        ):
            msg = build_sign_speech(confirmed_sign_text)
            return msg, "sign", confirmed_sign_text, True

    if confirmed_lidar_alert.startswith("VERY_CLOSE_"):
        if (
            confirmed_lidar_alert != last_spoken_lidar_alert
            or (now - last_lidar_voice_time) >= VERY_CLOSE_REPEAT_SECONDS
        ):
            obj = matching_camera_object(confirmed_lidar_alert, detections, frame_w)
            msg = build_lidar_speech(confirmed_lidar_alert, obj)
            return msg, "lidar_very_close", confirmed_lidar_alert, True

    if confirmed_lidar_alert.startswith("STRONG_"):
        if (
            confirmed_lidar_alert != last_spoken_lidar_alert
            or (now - last_lidar_voice_time) >= OBSTACLE_REPEAT_SECONDS
        ):
            obj = matching_camera_object(confirmed_lidar_alert, detections, frame_w)
            msg = build_lidar_speech(confirmed_lidar_alert, obj)
            return msg, "lidar_strong", confirmed_lidar_alert, False

    if (
        confirmed_lidar_alert not in ("CLEAR",)
        and not confirmed_lidar_alert.startswith(("VERY_CLOSE_", "STRONG_"))
    ):
        if (
            confirmed_lidar_alert != last_spoken_lidar_alert
            or (now - last_lidar_voice_time) >= OBSTACLE_REPEAT_SECONDS
        ):
            obj = matching_camera_object(confirmed_lidar_alert, detections, frame_w)
            msg = build_lidar_speech(confirmed_lidar_alert, obj)
            return msg, "lidar_normal", confirmed_lidar_alert, False

    if confirmed_surroundings and not confirmed_lidar_alert.startswith(("VERY_CLOSE_", "STRONG_")):
        alert_key = ";".join(
            f"{normalize_object_label(l)}:{d}" for l, d in confirmed_surroundings
        )
        if (
            alert_key != last_spoken_object_alert
            or (now - last_object_voice_time) >= CAMERA_OBJECT_REPEAT_SECONDS
        ):
            msg = build_surroundings_speech(confirmed_surroundings)
            return msg, "camera_object", alert_key, False

    if (
        last_spoken_was_danger
        and confirmed_lidar_alert == "CLEAR"
        and lidar_clear_streak >= CLEAR_CONFIRM_SCANS
        and (now - last_clear_voice_time) >= CLEAR_REPEAT_SECONDS
    ):
        return "Path clear", "clear", "CLEAR", False

    return None


def speak_chosen_message(
    spoken_text: str, category: str, track_key: str, interrupt: bool
) -> bool:
    global last_spoken_message, last_voice_time, last_spoken_was_danger
    global last_spoken_sign_text, last_sign_voice_time
    global last_spoken_object_alert, last_object_voice_time
    global last_spoken_lidar_alert, last_lidar_voice_time, last_clear_voice_time

    if not is_voice_output_allowed():
        return False
    if interrupt or category in ("sign", "lidar_very_close"):
        if is_voice_speaking():
            stop_current_voice()
    ok = run_tts(spoken_text)
    if not ok:
        return False

    now = time.time()
    last_spoken_message = spoken_text
    last_voice_time = now

    if category == "sign":
        last_spoken_sign_text = track_key
        last_sign_voice_time = now
        last_spoken_was_danger = False
    elif category == "camera_object":
        last_spoken_object_alert = track_key
        last_object_voice_time = now
        last_spoken_was_danger = False
    elif category == "clear":
        last_clear_voice_time = now
        last_spoken_lidar_alert = "CLEAR"
        last_spoken_was_danger = False
    elif category.startswith("lidar"):
        last_spoken_lidar_alert = track_key
        last_lidar_voice_time = now
        last_spoken_was_danger = True

    return True


def display_alert_summary() -> str:
    mute_left = voice_mute_remaining_seconds()
    if mute_left > 0:
        return f"MUTE {mute_left:.0f}s"
    if confirmed_sign_text:
        return f"SIGN:{confirmed_sign_text}"
    if confirmed_lidar_alert != "CLEAR":
        return confirmed_lidar_alert
    if confirmed_surroundings:
        bits = [normalize_object_label(l) for l, _d in confirmed_surroundings[:3]]
        return "OBJ:" + ",".join(bits)
    if confirmed_object_label:
        return f"OBJ:{confirmed_object_label}"
    return "CLEAR"


def process_voice_alerts() -> str:
    with data_lock:
        pts = list(latest_scan_points)
    with camera_lock:
        dets = list(latest_camera_detections)
        ocr_items = list(latest_ocr_results)

    lidar_alert, _nearest, zc = detect_obstacles_for_blind_user(pts)
    update_lidar_voice_state_machine(lidar_alert)

    ocr_text = ""
    if ocr_items:
        ocr_text = ocr_items[0].text
    update_object_voice_state_machine(dets, CAMERA_WIDTH)

    last_zone_counts["front"] = zc["front"]
    last_zone_counts["left"] = zc["left"]
    last_zone_counts["right"] = zc["right"]
    last_zone_counts["back"] = zc["back"]
    direction_distances["front"] = _zone_nearest(pts, lambda x, y, d: x > 0 and abs(y) <= 0.45 and d <= CAUTION_DISTANCE_M)
    direction_distances["left"] = _zone_nearest(pts, lambda x, y, d: y < -0.35 and d <= CAUTION_DISTANCE_M)
    direction_distances["right"] = _zone_nearest(pts, lambda x, y, d: y > 0.35 and d <= CAUTION_DISTANCE_M)
    direction_distances["back"] = _zone_nearest(pts, lambda x, y, d: x < 0 and abs(y) <= 0.45 and d <= 0.8)

    choice = choose_voice_message(dets, CAMERA_WIDTH)
    if choice is not None:
        spoken_text, category, track_key, interrupt = choice
        if speak_chosen_message(spoken_text, category, track_key, interrupt):
            if category.startswith("lidar"):
                print(f"Voice: {spoken_text}")

    return display_alert_summary()


def print_camera_settings() -> None:
    print("Camera settings:")
    print(f"- CAMERA_BACKEND: {CAMERA_BACKEND}")
    print(f"- Resolution: {CAMERA_WIDTH} x {CAMERA_HEIGHT}")
    print(f"- PREFER_USB_CAMERA: {prefer_usb_camera}")
    print(f"- USB_CAMERA_INDEXES: {USB_CAMERA_INDEXES}")
    print(f"- ACTIVE_USB_CAMERA_INDEX: {ACTIVE_USB_CAMERA_INDEX}")
    print(f"- OCR_INTERVAL_SECONDS: {OCR_INTERVAL_SECONDS}")
    print(f"- AI_DETECTION_INTERVAL_SECONDS: {AI_DETECTION_INTERVAL_SECONDS}")
    print(f"- AI DNN model path: {AI_DNN_MODEL_PATH} "
          f"({'FOUND' if os.path.isfile(AI_DNN_MODEL_PATH) else 'MISSING — using HOG/sign/contour fallback'})")
    print(f"- Hailo model path: {AI_MODEL_PATH} "
          f"({'FOUND' if os.path.isfile(AI_MODEL_PATH) else 'MISSING'})")
    print("View: 0=quad | 1=camera | 2=zones | 3=LiDAR | 4=floor-plan | Z=zones overlay | F=fullscreen")
    print("Camera: C=retry | X=on/off | P=USB/PiCam priority | R=read sign now | M=reset SLAM map | N=mute 30s")
    print("Troubleshooting:")
    print("  libcamera-hello")
    print("  rpicam-hello")
    print("  ls /dev/video*")
    print("  v4l2-ctl --list-devices")
    print(f"Picamera2 import: {'OK' if Picamera2 is not None else 'MISSING'}")
    print(f"OpenCV import: {'OK' if cv2 is not None else 'MISSING'}")
    print(f"pytesseract import: {'OK' if pytesseract is not None else 'MISSING'}")


def init_ocr_system() -> bool:
    """Verify Tesseract + OpenCV for sign OCR (Pi 5 / Linux)."""
    if not ENABLE_OCR:
        print("OCR disabled in settings (ENABLE_OCR=False).")
        return False
    if cv2 is None:
        print("OCR OFF: install python3-opencv  (sudo apt install python3-opencv)")
        return False
    if pytesseract is None:
        print("OCR OFF: install python3-pytesseract  (sudo apt install python3-pytesseract)")
        return False
    tess = shutil.which("tesseract") or "/usr/bin/tesseract"
    if not os.path.isfile(tess):
        print("OCR OFF: install tesseract-ocr  (sudo apt install tesseract-ocr)")
        return False
    pytesseract.pytesseract.tesseract_cmd = tess
    try:
        ver = pytesseract.get_tesseract_version()
        print(f"OCR OK: Tesseract {ver} at {tess}")
    except Exception as exc:
        print(f"OCR WARNING: Tesseract found but test failed: {exc}")
        return False
    print(
        f"OCR every {OCR_INTERVAL_SECONDS:.0f}s; "
        f"sign voice repeat every {OCR_VOICE_REPEAT_SECONDS:.0f}s; "
        f"stable reads {OCR_REQUIRE_STABLE_READS} in {OCR_STABLE_WINDOW_SECONDS:.0f}s; "
        f"vocabulary correction={'ON' if OCR_VOCABULARY_CORRECTION else 'OFF'} "
        f"(reads numbers, street names, signs up to {OCR_MAX_TEXT_LENGTH} chars)"
    )
    return True


def get_centre_ocr_box(frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """Centre sign-reading region — hold the sign inside this yellow box."""
    x1 = int(frame_w * 0.10)
    y1 = int(frame_h * 0.15)
    x2 = int(frame_w * 0.90)
    y2 = int(frame_h * 0.80)
    return x1, y1, x2, y2


def get_upper_ocr_box(frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """Upper sign band — signs are often mounted high."""
    return 0, 0, frame_w, max(60, int(frame_h * 0.55))


def is_known_sign_word(text: str) -> bool:
    return text in KNOWN_SIGN_WORDS


def extract_known_sign_from_text(text: str) -> str:
    """Find a known sign word inside noisy OCR (e.g. 'THE EXIT AHEAD' -> EXIT)."""
    if not text:
        return ""
    t = re.sub(r"[^A-Z0-9 ]+", " ", text.upper())
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    if is_known_sign_word(t):
        return t
    # Prefer longer known phrases first
    for known in sorted(KNOWN_SIGN_WORDS, key=len, reverse=True):
        if known in t:
            return known
    # Per-word fuzzy match
    for word in t.split():
        fixed = _fix_ocr_chars_in_word(word)
        if is_known_sign_word(fixed):
            return fixed
        best_word = ""
        best_ratio = 0.0
        for known in KNOWN_SIGN_WORDS:
            if " " in known:
                continue
            ratio = difflib.SequenceMatcher(None, fixed, known).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_word = known
        if best_ratio >= OCR_FUZZY_MATCH_THRESHOLD and _fuzzy_length_ok(fixed, best_word):
            return best_word
    return ""


def is_number_like_text(text: str) -> bool:
    compact = text.replace(" ", "")
    if not compact:
        return False
    digit = sum(1 for c in compact if c.isdigit())
    alpha = sum(1 for c in compact if c.isalpha())
    if digit < OCR_MIN_NUMBER_LENGTH:
        return False
    if digit >= 1 and alpha == 0 and len(compact) <= 8:
        return True
    # Room / door style: 12A, B12, 3F
    if digit >= 1 and alpha <= 2 and len(compact) <= 8:
        return True
    return False


def looks_like_street_name(text: str) -> bool:
    words = text.upper().split()
    compact = text.upper().replace(" ", "")
    if len(words) >= 2 and any(w in STREET_NAME_HINTS for w in words):
        return True
    if any(h in compact for h in STREET_NAME_COMPACT_HINTS) and len(compact) >= 6:
        return True
    if len(words) >= 2 and len(compact) >= 8:
        return True
    return False


def is_valid_sign_text(text: str) -> bool:
    """True if OCR text looks like readable sign/number/street text."""
    if not text:
        return False
    t = text.strip().upper()
    compact = t.replace(" ", "")
    if len(compact) > OCR_MAX_TEXT_LENGTH:
        return False
    if is_number_like_text(t):
        return True
    if len(t) < OCR_MIN_TEXT_LENGTH:
        return False
    alpha = sum(1 for c in compact if c.isalpha())
    digit = sum(1 for c in compact if c.isdigit())
    if looks_like_street_name(t):
        return alpha >= 4
    if alpha < max(2, int(len(compact) * 0.40)) and digit == 0:
        return False
    if len(compact) >= 3 and len(set(compact)) == 1:
        return False
    return True


def should_apply_vocab_correction(text: str) -> bool:
    """Only fuzzy-correct short sign-like phrases — not street names or numbers."""
    if not text:
        return False
    if is_number_like_text(text) or looks_like_street_name(text):
        return False
    words = text.split()
    compact = text.replace(" ", "")
    if len(words) > OCR_VOCAB_MAX_WORDS:
        return False
    if len(compact) > OCR_VOCAB_MAX_CHARS:
        return False
    return True


def _fuzzy_length_ok(candidate: str, known: str) -> bool:
    cl = len(candidate.replace(" ", ""))
    kl = len(known.replace(" ", ""))
    return abs(cl - kl) <= max(2, int(kl * 0.2))


def pick_ocr_final_text(cleaned: str, matched: str) -> str:
    """Prefer real OCR text when vocabulary would wrongly replace numbers/streets."""
    if not cleaned and not matched:
        return ""
    if not matched:
        return cleaned
    if not cleaned:
        return matched
    if cleaned == matched:
        return cleaned
    if is_number_like_text(cleaned) or looks_like_street_name(cleaned):
        return cleaned

    c_words = cleaned.split()
    m_words = matched.split()
    c_len = len(cleaned.replace(" ", ""))
    m_len = len(matched.replace(" ", ""))

    # Do not replace multi-word / long OCR (e.g. street name) with one vocabulary word
    if len(c_words) > 1 and len(m_words) == 1 and is_known_sign_word(matched):
        if not is_known_sign_word(cleaned):
            return cleaned
    if c_len > m_len + 3 and is_known_sign_word(matched) and not is_known_sign_word(cleaned):
        return cleaned
    if should_apply_vocab_correction(cleaned):
        return matched
    return cleaned


def _fix_ocr_chars_in_word(word: str) -> str:
    """Replace common OCR digit mistakes inside letter words; keep real numbers."""
    if not word:
        return word
    upper = word.upper()
    digit_n = sum(1 for c in upper if c.isdigit())
    alpha_n = sum(1 for c in upper if c.isalpha())
    # Keep number-like tokens intact (12, 3A, B12)
    if digit_n >= 1 and digit_n >= alpha_n:
        return upper
    out: List[str] = []
    for i, c in enumerate(upper):
        if c == "0":
            out.append("O")
        elif c == "1":
            prev = upper[i - 1] if i > 0 else ""
            out.append("I" if prev in ("T", "E", "X", "F", "I") else "L")
        elif c == "5":
            out.append("S")
        elif c == "8":
            out.append("B")
        else:
            out.append(c)
    return "".join(out)


def clean_ocr_text(text: str) -> str:
    """Clean OCR text; allow numbers and longer street-style phrases."""
    t = text.strip().upper()
    t = re.sub(r"[^A-Z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""

    words = [_fix_ocr_chars_in_word(w) for w in t.split()]
    t = " ".join(words).strip()

    corrections = {
        "EX1T": "EXIT", "EX1 T": "EXIT", "EX1": "EXIT", "E X I T": "EXIT", "EXlT": "EXIT", "EX T": "EXIT",
        "ST0P": "STOP", "ST0 P": "STOP", "5TOP": "STOP", "STQP": "STOP",
        "P0SH": "PUSH", "P0LL": "PULL",
        "T0ILET": "TOILET", "T01LET": "TOILET", "TO1LET": "TOILET",
        "OFF1CE": "OFFICE", "0FFICE": "OFFICE",
        "F1RE EX1T": "FIRE EXIT", "FIRE EX1T": "FIRE EXIT",
        "N0 ENTRY": "NO ENTRY", "NO ENT RY": "NO ENTRY",
        "ENTRANCE": "ENTRANCE", "PUSH": "PUSH", "PULL": "PULL",
        "STA1RS": "STAIRS", "CAUT1ON": "CAUTION", "DANGER": "DANGER",
    }
    if t in corrections:
        t = corrections[t]
    else:
        # Per-word exact fixes only — never rewrite a long phrase because one word matches
        t = " ".join(corrections.get(w, w) for w in t.split())

    compact = t.replace(" ", "")
    if is_number_like_text(t):
        return t
    if len(compact) < OCR_MIN_TEXT_LENGTH:
        return ""
    if len(compact) > OCR_MAX_TEXT_LENGTH:
        return ""

    if len(compact) >= 3 and len(set(compact)) == 1:
        return ""
    if len(compact) >= 4 and compact.count(compact[0]) / len(compact) > 0.85:
        return ""

    if not any(c in "AEIOU" for c in compact):
        if (
            len(compact) > 6
            and t not in VOWELLESS_OK
            and not any(t == w or t.startswith(w + " ") for w in VOWELLESS_OK)
            and not looks_like_street_name(t)
        ):
            return ""

    return t


def match_known_sign_text(text: str) -> str:
    """Fuzzy-match short sign phrases; also pull known signs out of longer OCR noise."""
    extracted = extract_known_sign_from_text(text)
    if extracted:
        return extracted

    cleaned = clean_ocr_text(text)
    candidates: List[str] = []
    if cleaned:
        candidates.append(cleaned)
    rough = re.sub(r"[^A-Z0-9 ]+", " ", text.upper())
    rough = re.sub(r"\s+", " ", rough).strip()
    if rough and rough not in candidates:
        candidates.append(rough)

    if not candidates:
        return ""

    primary = cleaned or rough
    if not should_apply_vocab_correction(primary):
        return primary

    for cand in candidates:
        if is_known_sign_word(cand):
            return cand

    if OCR_VOCABULARY_CORRECTION:
        best_word = ""
        best_ratio = 0.0
        best_cand = ""
        for cand in candidates:
            for known in KNOWN_SIGN_WORDS:
                ratio = difflib.SequenceMatcher(None, cand, known).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_word = known
                    best_cand = cand
        if (
            best_ratio >= OCR_FUZZY_MATCH_THRESHOLD
            and best_word
            and _fuzzy_length_ok(best_cand, best_word)
        ):
            return best_word

    return primary


def score_ocr_result(
    text: str,
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
) -> float:
    """Score OCR candidate — signs, numbers, and street names welcome."""
    if not text:
        return 0.0
    fh, fw = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    box_area = bw * bh
    frame_area = max(1, fw * fh)
    area_ratio = box_area / frame_area

    score = 0.40
    word_count = len(text.split())
    if is_known_sign_word(text) and word_count <= 2:
        score += 0.20
    elif looks_like_street_name(text):
        score += 0.18
    elif word_count >= 3:
        score += 0.12

    text_len = len(text.replace(" ", ""))
    if is_number_like_text(text):
        score += 0.20
    elif 3 <= text_len <= 40:
        score += 0.12
    elif text_len > 60:
        score -= 0.08

    digit_count = sum(1 for c in text if c.isdigit())
    if digit_count > 0 and is_number_like_text(text):
        score += 0.08
    elif digit_count > 4 and not looks_like_street_name(text):
        score -= 0.05

    compact = text.replace(" ", "")
    if len(compact) >= 4 and len(set(compact)) <= 2:
        score -= 0.30

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    centre_box = get_centre_ocr_box(fw, fh)
    upper_box = get_upper_ocr_box(fw, fh)
    if _bbox_overlap_ratio(bbox, centre_box) > 0.35:
        score += 0.20
    elif _bbox_overlap_ratio(bbox, upper_box) > 0.25:
        score += 0.12

    if area_ratio > 0.85:
        score -= 0.35
    elif area_ratio > 0.55:
        score -= 0.15

    if text in ("EXIT", "STOP", "PUSH", "PULL", "TOILET", "OFFICE", "ENTRANCE", "FIRE EXIT", "NO ENTRY"):
        score += 0.10

    return max(0.0, min(1.0, score))


def _bbox_overlap_ratio(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / a_area


def _ocr_tesseract_configs(fast: bool = False, street_line: bool = False) -> List[str]:
    """Sign-friendly Tesseract configs. street_line prioritises single horizontal lines."""
    whitelist = " -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if street_line:
        return [
            f"--oem 3 --psm 7{whitelist}",
            f"--oem 3 --psm 6{whitelist}",
            f"--oem 3 --psm 7",
            f"--oem 3 --psm 13{whitelist}",
        ]
    if fast:
        return [
            f"--oem 3 --psm 7{whitelist}",
            f"--oem 3 --psm 8{whitelist}",
            f"--oem 3 --psm 6{whitelist}",
            "--oem 3 --psm 6",
        ]
    return [
        f"--oem 3 --psm 7{whitelist}",
        f"--oem 3 --psm 8{whitelist}",
        f"--oem 3 --psm 6{whitelist}",
        f"--oem 3 --psm 11{whitelist}",
        "--oem 3 --psm 6",
        "--oem 3 --psm 7",
    ]


def _prepare_ocr_variants(gray_roi: np.ndarray) -> List[np.ndarray]:
    """Multiple contrast variants so light/dark signs both work."""
    if cv2 is None:
        return [gray_roi]
    roi = gray_roi
    try:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        roi = clahe.apply(roi)
    except Exception:
        pass
    roi = cv2.convertScaleAbs(roi, alpha=1.5, beta=8)
    variants = [roi]
    _, otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.bitwise_not(otsu))
    variants.append(
        cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
    )
    return variants


def _prepare_color_sign_variants(
    frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]
) -> List[np.ndarray]:
    """Extra variants for white-on-green / coloured street signs."""
    if cv2 is None or frame_bgr is None:
        return []
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 12 or y2 - y1 < 8:
        return []
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    scale = max(2.5, OCR_FRAME_SCALE)
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    variants: List[np.ndarray] = []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass
    variants.append(gray)
    # White-on-green: boost brightness where colour is pale
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 160), (180, 80, 255))
    variants.append(white)
    variants.append(cv2.bitwise_not(white))
    # Green-channel difference (text often brighter than green paint)
    _b, g, _r = cv2.split(crop)
    mix = cv2.normalize(cv2.subtract(gray, cv2.divide(g, 3)), None, 0, 255, cv2.NORM_MINMAX)
    variants.append(mix)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.bitwise_not(otsu))
    return variants


def _bbox_is_street_line(bbox: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    return w >= 80 and (w / float(h)) >= 2.2


def _ocr_single_roi(
    gray: np.ndarray,
    bbox: Tuple[int, int, int, int],
    results: List[OCRResult],
    frame_shape: Tuple[int, int],
    fast: bool = False,
    frame_bgr: Optional[np.ndarray] = None,
) -> None:
    if pytesseract is None or cv2 is None:
        return
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 12 or y2 - y1 < 10:
        return
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return
    scale = max(2.0, OCR_FRAME_SCALE)
    roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    street_line = _bbox_is_street_line((x1, y1, x2, y2))
    variants = _prepare_color_sign_variants(frame_bgr, (x1, y1, x2, y2)) if frame_bgr is not None else []
    variants.extend(_prepare_ocr_variants(roi))
    seen: set = set()
    configs = _ocr_tesseract_configs(fast=fast, street_line=street_line)

    def _consider_raw(raw_txt: str) -> None:
        raw_stripped = raw_txt.strip().upper()
        if not raw_stripped:
            return
        cleaned = clean_ocr_text(raw_txt)
        matched = match_known_sign_text(raw_txt)
        final_text = pick_ocr_final_text(cleaned, matched)
        known_hit = extract_known_sign_from_text(raw_txt) or extract_known_sign_from_text(cleaned)
        if known_hit:
            if not final_text:
                final_text = known_hit
            elif should_apply_vocab_correction(final_text) and is_known_sign_word(known_hit):
                final_text = known_hit
        if not final_text or not is_valid_sign_text(final_text):
            return
        if final_text in seen:
            return
        seen.add(final_text)
        score = score_ocr_result(final_text, (x1, y1, x2, y2), frame_shape)
        centre = get_centre_ocr_box(frame_shape[1], frame_shape[0])
        if _bbox_overlap_ratio((x1, y1, x2, y2), centre) > 0.20:
            score = min(1.0, score + 0.15)
        if is_known_sign_word(final_text):
            score = min(1.0, score + 0.15)
        if is_number_like_text(final_text) or looks_like_street_name(final_text):
            score = min(1.0, score + 0.18)
        if street_line and looks_like_street_name(final_text):
            score = min(1.0, score + 0.10)
        if score < OCR_MIN_CONFIDENCE_SCORE:
            return
        results.append(OCRResult(
            text=final_text,
            confidence=score,
            bbox=(x1, y1, x2, y2),
            timestamp=time.time(),
            raw_text=raw_stripped[:80],
            cleaned_text=cleaned or final_text,
            matched_text=final_text,
            score=score,
        ))
        ocr_log_rows.append([
            round(time.time(), 3), final_text, f"{score:.3f}", x1, y1, x2, y2,
        ])

    for img in variants:
        # Word-level OCR reconstructs street lines better than whole-block string
        if not fast:
            try:
                data = pytesseract.image_to_data(
                    img,
                    config="--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    output_type=pytesseract.Output.DICT,
                )
                words = []
                for i, txt in enumerate(data.get("text", [])):
                    conf = int(float(data["conf"][i])) if str(data["conf"][i]).lstrip("-").isdigit() else -1
                    t = (txt or "").strip()
                    if conf >= 35 and t:
                        words.append(t.upper())
                if words:
                    _consider_raw(" ".join(words))
            except Exception:
                pass
        for cfg in configs:
            try:
                raw_txt = pytesseract.image_to_string(img, config=cfg)
            except Exception:
                continue
            _consider_raw(raw_txt)
            if results:
                top = max(results, key=lambda r: r.score)
                if top.score >= OCR_EARLY_EXIT_SCORE and is_known_sign_word(top.text):
                    return
                if top.score >= OCR_STREET_EARLY_EXIT_SCORE and looks_like_street_name(top.text):
                    return


def _append_mask_rois(
    mask: np.ndarray,
    rois: List[Tuple[int, int, int, int]],
    frame_w: int,
    frame_h: int,
    y_offset: int = 0,
    min_area: int = 400,
    prefer_wide: bool = False,
) -> None:
    if cv2 is None:
        return
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area or bw < 24 or bh < 10:
            continue
        aspect = bw / float(max(1, bh))
        if prefer_wide and aspect < 1.6:
            continue
        pad_x, pad_y = 10, 8
        box = (
            max(0, x - pad_x),
            max(0, y + y_offset - pad_y),
            min(frame_w, x + bw + pad_x),
            min(frame_h, y + y_offset + bh + pad_y),
        )
        score = area * (1.4 if aspect >= 2.5 else 1.0)
        scored.append((score, box))
    scored.sort(key=lambda t: t[0], reverse=True)
    for _s, box in scored[:OCR_MAX_EXTRA_ROIS]:
        rois.append(box)


def find_color_sign_rois(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Find sign regions: green street signs, red STOP, blue boards, bright panels."""
    if cv2 is None:
        return []
    h, w = frame_bgr.shape[:2]
    upper_h = max(1, int(h * 0.75))
    upper = frame_bgr[0:upper_h, :]
    rois: List[Tuple[int, int, int, int]] = []
    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)

    # Singapore-style green street name plates (wide, not tall foliage blobs)
    green = cv2.inRange(hsv, (35, 50, 50), (95, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((5, 25), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    green_rois: List[Tuple[int, int, int, int]] = []
    _append_mask_rois(green, green_rois, w, h, 0, min_area=600, prefer_wide=True)
    for box in green_rois:
        _x1, y1, _x2, y2 = box
        if (y2 - y1) <= int(h * 0.28):  # reject tall tree blobs
            rois.append(box)

    # Red STOP / warning signs
    red1 = cv2.inRange(hsv, (0, 70, 60), (14, 255, 255))
    red2 = cv2.inRange(hsv, (160, 70, 60), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    _append_mask_rois(red_mask, rois, w, h, 0, min_area=350, prefer_wide=False)

    # Blue information boards
    blue = cv2.inRange(hsv, (95, 60, 50), (130, 255, 255))
    _append_mask_rois(blue, rois, w, h, 0, min_area=450, prefer_wide=False)

    # Bright white / pale boards
    gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
    _append_mask_rois(bright, rois, w, h, 0, min_area=400, prefer_wide=False)

    # Deduplicate
    unique: List[Tuple[int, int, int, int]] = []
    seen: set = set()
    for box in rois:
        key = (box[0] // 25, box[1] // 25, box[2] // 25, box[3] // 25)
        if key in seen:
            continue
        seen.add(key)
        unique.append(box)
    return unique[:OCR_MAX_EXTRA_ROIS]


def run_ocr_on_signs(frame_bgr: np.ndarray, detections: List[Detection]) -> List[OCRResult]:
    if not ocr_enabled or pytesseract is None or cv2 is None:
        return []
    h, w = frame_bgr.shape[:2]
    frame_shape = (h, w)
    gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    centre = get_centre_ocr_box(w, h)
    results: List[OCRResult] = []

    def _good_enough() -> bool:
        if not results:
            return False
        best = max(results, key=lambda r: r.score)
        if best.score >= OCR_EARLY_EXIT_SCORE and is_known_sign_word(best.text):
            return True
        if best.score >= OCR_STREET_EARLY_EXIT_SCORE and looks_like_street_name(best.text):
            return True
        return False

    # 1) Coloured street-sign ROIs first (green Ave plates, etc.)
    color_rois = find_color_sign_rois(frame_bgr)
    for box in color_rois:
        _ocr_single_roi(gray_full, box, results, frame_shape, fast=False, frame_bgr=frame_bgr)
        if _good_enough():
            results.sort(key=lambda r: (-r.score, -r.confidence))
            return results[:5]

    # 2) Centre hold-to-read box
    _ocr_single_roi(gray_full, centre, results, frame_shape, fast=False, frame_bgr=frame_bgr)
    if _good_enough():
        results.sort(key=lambda r: (-r.score, -r.confidence))
        return results[:5]

    # 3) Detection boxes + upper band fallback
    need_more = (not results) or (max(r.score for r in results) < OCR_STREET_EARLY_EXIT_SCORE)
    if need_more:
        extra_rois: List[Tuple[int, int, int, int]] = []
        sign_like = ("sign", "stop", "exit", "text", "poster")
        for d in detections:
            if any(s in d.label.lower() for s in sign_like):
                extra_rois.append(d.bbox)
        extra_rois.append(get_upper_ocr_box(w, h))
        if OCR_WHOLE_FRAME_FALLBACK:
            extra_rois.append((0, 0, w, h))

        seen_boxes: set = set()
        for box in [centre] + color_rois:
            seen_boxes.add((box[0] // 20, box[1] // 20, box[2] // 20, box[3] // 20))
        for box in extra_rois[:OCR_MAX_EXTRA_ROIS]:
            key = (box[0] // 20, box[1] // 20, box[2] // 20, box[3] // 20)
            if key in seen_boxes:
                continue
            seen_boxes.add(key)
            _ocr_single_roi(gray_full, box, results, frame_shape, fast=True, frame_bgr=frame_bgr)
            if _good_enough():
                break

    results.sort(key=lambda r: (-r.score, -r.confidence))
    return results[:5]


def _prune_ocr_votes(now: float) -> None:
    global ocr_vote_events, ocr_vote_counts
    ocr_vote_events = [(t, txt) for t, txt in ocr_vote_events if now - t <= OCR_STABLE_WINDOW_SECONDS]
    ocr_vote_counts = {}
    for _t, txt in ocr_vote_events:
        ocr_vote_counts[txt] = ocr_vote_counts.get(txt, 0) + 1


def record_ocr_vote(text: str) -> int:
    """Record a stable OCR vote; return count for this text in the current window."""
    global ocr_vote_events, ocr_last_vote_time, ocr_vote_counts
    now = time.time()
    _prune_ocr_votes(now)
    if text:
        ocr_vote_events.append((now, text))
        ocr_last_vote_time = now
    _prune_ocr_votes(now)
    return ocr_vote_counts.get(text, 0)


def apply_ocr_scan_results(new_items: List[OCRResult], manual: bool = False) -> None:
    """Persist OCR results, vote for stable reads, speak only after confirmation."""
    global latest_ocr_results, last_ocr_text, last_ocr_update_time, last_ocr_persist_until
    global ocr_debug_raw, ocr_debug_cleaned, ocr_debug_matched, ocr_debug_confirmed
    global ocr_last_candidates, confirmed_sign_text, sign_candidate_text, sign_candidate_count

    now = time.time()
    if not new_items:
        if last_ocr_persist_until > 0 and now > last_ocr_persist_until:
            with camera_lock:
                if latest_ocr_results and now - last_ocr_update_time >= OCR_PERSIST_SECONDS:
                    latest_ocr_results = []
        return

    best = max(new_items, key=lambda r: (r.score, r.confidence))
    ocr_last_candidates = [f"{r.text}({r.score:.2f})" for r in new_items[:8]]
    ocr_debug_raw = best.raw_text or best.text
    ocr_debug_cleaned = best.cleaned_text or best.text
    ocr_debug_matched = best.matched_text or best.text

    if manual:
        print("--- Manual OCR read (R) ---")
        for r in new_items[:8]:
            print(f"  candidate: raw={r.raw_text!r} cleaned={r.cleaned_text!r} "
                  f"matched={r.matched_text!r} score={r.score:.2f}")

    vote_text = best.matched_text or best.text
    if not vote_text:
        return

    vote_count = record_ocr_vote(vote_text)
    print(f"OCR vote: {vote_text} ({vote_count}/{OCR_REQUIRE_STABLE_READS})")

    with camera_lock:
        latest_ocr_results = [best]
    last_ocr_update_time = now
    last_ocr_persist_until = now + OCR_PERSIST_SECONDS
    last_ocr_text = vote_text

    if vote_count >= OCR_REQUIRE_STABLE_READS:
        confirm_sign_text(vote_text, speak_now=True)
    elif manual and vote_text and is_valid_sign_text(vote_text):
        confirm_sign_text(vote_text, speak_now=True)
        print(f"Manual OCR confirmed: {vote_text} (score={best.score:.2f})")
    elif (
        not manual
        and best.score >= OCR_AUTO_SINGLE_READ_SCORE
        and (
            is_known_sign_word(vote_text)
            or looks_like_street_name(vote_text)
            or is_number_like_text(vote_text)
        )
    ):
        confirm_sign_text(vote_text, speak_now=True)
        print(f"High-confidence OCR confirmed: {vote_text} (score={best.score:.2f})")
    elif manual:
        print(f"OCR not confirmed yet — need {OCR_REQUIRE_STABLE_READS} reads in "
              f"{OCR_STABLE_WINDOW_SECONDS:.0f}s (have {vote_count})")


def approximate_distance_from_bbox(frame_w: int, bbox: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    rel = box_w / max(1, frame_w)
    return max(0.3, min(5.0, 1.9 / (rel + 1e-3)))


def try_hailo_inference_placeholder(frame_bgr: np.ndarray) -> Optional[List[Detection]]:
    """Internal Hailo inference hook — returns None until SDK code is inserted."""
    return run_ai_hat_inference(frame_bgr)


def update_ai_hat_status() -> None:
    global AI_HAT_STATUS
    if not ENABLE_AI_HAT:
        AI_HAT_STATUS = "OFF"
    elif not HAILO_AVAILABLE:
        AI_HAT_STATUS = "RUNTIME_MISSING"
    elif not os.path.isfile(AI_MODEL_PATH):
        AI_HAT_STATUS = "MODEL_MISSING"
    elif ai_hat_active:
        AI_HAT_STATUS = "ACTIVE"
    elif HAILO_AVAILABLE and os.path.isfile(AI_MODEL_PATH):
        AI_HAT_STATUS = "PLACEHOLDER"
    else:
        AI_HAT_STATUS = "FALLBACK"


def init_hailo_yolo_detector() -> bool:
    """Initialise Hailo YOLO detector. Returns True only when fully wired."""
    if not HAILO_AVAILABLE or not os.path.isfile(AI_MODEL_PATH):
        return False
    # PLACEHOLDER: insert Hailo SDK initialisation here.
    return False


def run_hailo_yolo_detector(frame_bgr: np.ndarray) -> Optional[List[Detection]]:
    """Run Hailo YOLO on BGR frame. Returns None if not implemented."""
    if not ai_hat_active or not ENABLE_AI_HAT:
        return None
    try:
        # PLACEHOLDER: run Hailo inference and parse_ai_hat_results()
        return None
    except Exception as exc:
        print(f"Hailo inference error: {exc}")
        return None


def init_ai_hat() -> bool:
    """Initialise AI HAT / Hailo runtime with honest status reporting."""
    global ai_hat_active, ai_hat_device, AI_HAT_RUNTIME_AVAILABLE
    if not ENABLE_AI_HAT:
        print("AI HAT disabled in settings.")
        update_ai_hat_status()
        return False
    if not HAILO_AVAILABLE:
        print("AI HAT runtime missing — using OpenCV fallback.")
        AI_HAT_RUNTIME_AVAILABLE = False
        update_ai_hat_status()
        return False
    try:
        AI_HAT_RUNTIME_AVAILABLE = True
        if init_hailo_yolo_detector():
            ai_hat_active = True
            print("AI HAT: ACTIVE (Hailo inference running)")
        elif os.path.isfile(AI_MODEL_PATH):
            ai_hat_active = False
            print(
                "AI HAT runtime found but inference is placeholder. "
                "Camera is using OpenCV fallback."
            )
        else:
            ai_hat_active = False
            print(f"AI HAT model not found: {AI_MODEL_PATH}")
        update_ai_hat_status()
        return ai_hat_active
    except Exception as exc:
        print(f"AI HAT init failed: {exc}")
        AI_HAT_RUNTIME_AVAILABLE = False
        ai_hat_active = False
        update_ai_hat_status()
        return False


def load_ai_model() -> bool:
    """Load HEF model from AI_MODEL_PATH. Returns True if model ready."""
    global ai_hat_active
    if not os.path.isfile(AI_MODEL_PATH):
        print(f"AI model not found: {AI_MODEL_PATH}")
        ai_hat_active = False
        return False
    # PLACEHOLDER: load HEF with Hailo SDK
    # hef = HEF(AI_MODEL_PATH)
  # configure input/output vstreams on ai_hat_device
    ai_hat_active = False  # set True when SDK pipeline is wired
    return ai_hat_active


def parse_ai_hat_results(raw_output) -> List[Detection]:
    """Decode Hailo raw tensor output into Detection list."""
    # PLACEHOLDER: parse bounding boxes, class ids, scores from raw_output
    _ = raw_output
    return []


def run_ai_hat_inference(frame_bgr: np.ndarray) -> Optional[List[Detection]]:
    """Run AI HAT inference — delegates to Hailo hook."""
    return run_hailo_yolo_detector(frame_bgr)


def draw_ai_detections(frame_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw bounding boxes, labels, confidence, distance on camera frame."""
    if cv2 is None:
        return frame_bgr
    out = frame_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        col = (40, 220, 120) if d.source in ("AI_HAT", "hailo", "opencv_dnn") else (240, 180, 80)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        dist = f" {d.distance_m:.1f}m" if d.distance_m else ""
        txt = f"{d.label}{dist} {int(d.confidence * 100)}%"
        cv2.putText(out, txt, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out


def detect_with_opencv_dnn(frame_bgr: np.ndarray) -> List[Detection]:
    """Run YOLOv8 ONNX via OpenCV DNN (handles (1,84,N) output layout)."""
    global dnn_net
    if cv2 is None:
        return []
    if dnn_net is None and os.path.isfile(AI_DNN_MODEL_PATH):
        try:
            dnn_net = cv2.dnn.readNet(AI_DNN_MODEL_PATH)
            print(f"OpenCV DNN loaded: {AI_DNN_MODEL_PATH}")
        except Exception as exc:
            print(f"OpenCV DNN load failed: {exc}")
            dnn_net = None
    if dnn_net is None:
        return []

    h, w = frame_bgr.shape[:2]
    inp = AI_DNN_INPUT_SIZE
    try:
        blob = cv2.dnn.blobFromImage(
            frame_bgr, 1 / 255.0, (inp, inp), swapRB=True, crop=False
        )
        dnn_net.setInput(blob)
        outs = dnn_net.forward()
    except Exception as exc:
        print(f"OpenCV DNN forward failed: {exc}")
        return []

    pred = outs[0] if isinstance(outs, (list, tuple)) and len(outs) else outs
    if pred is None or not hasattr(pred, "shape"):
        return []
    pred = np.asarray(pred)
    if pred.ndim == 3:
        pred = pred[0]
    # YOLOv8: (84, 8400) -> (8400, 84); YOLOv5 may already be (N, 85)
    if pred.ndim != 2:
        return []
    if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 85:
        pred = pred.T

    boxes: List[List[int]] = []
    confidences: List[float] = []
    class_ids: List[int] = []
    sx = w / float(inp)
    sy = h / float(inp)

    for row in pred:
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        if row.size < 6:
            continue
        # YOLOv8: [cx,cy,w,h, class0..class79]  (84 values)
        # YOLOv5: [cx,cy,w,h, obj, class0..]   (85 values)
        if row.size == 84 or (80 <= row.size <= 84):
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])
        else:
            obj = float(row[4])
            class_scores = row[5:]
            if class_scores.size == 0:
                continue
            class_id = int(np.argmax(class_scores))
            conf = obj * float(class_scores[class_id])
        if conf < AI_CONFIDENCE_THRESHOLD:
            continue

        cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        x1 = int((cx - bw / 2.0) * sx)
        y1 = int((cy - bh / 2.0) * sy)
        bw_i = int(bw * sx)
        bh_i = int(bh * sy)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        bw_i = max(1, min(w - x1, bw_i))
        bh_i = max(1, min(h - y1, bh_i))
        boxes.append([x1, y1, bw_i, bh_i])
        confidences.append(conf)
        class_ids.append(class_id)

    if not boxes:
        return []

    try:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, AI_CONFIDENCE_THRESHOLD, AI_NMS_THRESHOLD)
    except Exception:
        indices = list(range(len(boxes)))

    dets: List[Detection] = []
    if indices is None or len(indices) == 0:
        return []
    flat = np.array(indices).reshape(-1)
    labels = ai_labels if ai_labels else COCO_DEFAULT_LABELS
    for i in flat:
        i = int(i)
        x, y, bw_i, bh_i = boxes[i]
        bbox = (x, y, x + bw_i, y + bh_i)
        cid = class_ids[i]
        label = labels[cid] if 0 <= cid < len(labels) else f"class_{cid}"
        dets.append(
            Detection(
                label,
                float(confidences[i]),
                bbox,
                approximate_distance_from_bbox(w, bbox),
                "opencv_dnn",
                time.time(),
            )
        )
    dets.sort(key=lambda d: d.confidence, reverse=True)
    return dets[:12]


def detect_with_hog(frame_bgr: np.ndarray) -> List[Detection]:
    global hog_detector
    if cv2 is None:
        return []
    if hog_detector is None:
        hog_detector = cv2.HOGDescriptor()
        hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog_detector.detectMultiScale(
        frame_bgr, winStride=(4, 4), padding=(12, 12), scale=1.04
    )
    h, w = frame_bgr.shape[:2]
    dets = []
    for (x, y, rw, rh), wt in zip(rects, weights):
        conf = float(wt[0] if hasattr(wt, "__len__") else wt)
        if conf < 0.15:
            continue
        bbox = (int(x), int(y), int(x + rw), int(y + rh))
        dets.append(
            Detection(
                "person",
                min(0.92, 0.35 + conf / 2.0),
                bbox,
                approximate_distance_from_bbox(w, bbox),
                "hog",
                time.time(),
            )
        )
    return dets


def detect_colored_sign_objects(frame_bgr: np.ndarray) -> List[Detection]:
    """Detect green/red/blue sign boards as objects when YOLO model is missing."""
    if cv2 is None:
        return []
    h, w = frame_bgr.shape[:2]
    rois = find_color_sign_rois(frame_bgr)
    dets: List[Detection] = []
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    for box in rois[:6]:
        x1, y1, x2, y2 = box
        if x2 - x1 < 20 or y2 - y1 < 10:
            continue
        patch = hsv[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        mean_h = float(np.mean(patch[:, :, 0]))
        mean_s = float(np.mean(patch[:, :, 1]))
        aspect = (x2 - x1) / float(max(1, y2 - y1))
        if 35 <= mean_h <= 95 and mean_s > 40:
            label = "sign"
            conf = 0.62 if aspect >= 2.0 else 0.50
        elif mean_h <= 14 or mean_h >= 160:
            label = "stop sign" if aspect < 1.8 else "sign"
            conf = 0.58
        elif 95 <= mean_h <= 130:
            label = "sign"
            conf = 0.52
        else:
            label = "sign"
            conf = 0.45
        dets.append(
            Detection(
                label,
                conf,
                box,
                approximate_distance_from_bbox(w, box),
                "color_sign",
                time.time(),
            )
        )
    return dets


def detect_with_contours(frame_bgr: np.ndarray) -> List[Detection]:
    if cv2 is None:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 130)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame_bgr.shape[:2]
    frame_area = float(max(1, h * w))
    dets = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 900 or area > frame_area * 0.55:
            continue
        x, y, rw, rh = cv2.boundingRect(c)
        if rw < 22 or rh < 22:
            continue
        # Prefer mid-frame obstacles (not sky strip)
        if y < int(h * 0.08) and rh < int(h * 0.2):
            continue
        bbox = (x, y, x + rw, y + rh)
        conf = min(0.55, 0.28 + area / 25000.0)
        dets.append(
            Detection(
                "obstacle",
                conf,
                bbox,
                approximate_distance_from_bbox(w, bbox),
                "contour",
                time.time(),
            )
        )
    dets.sort(key=lambda d: d.confidence, reverse=True)
    return dets[:6]


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def merge_detections(groups: List[List[Detection]], iou_thresh: float = 0.45) -> List[Detection]:
    """Merge multi-source detections, keeping highest-confidence overlaps."""
    merged: List[Detection] = []
    for group in groups:
        for det in group:
            replaced = False
            for i, existing in enumerate(merged):
                if _bbox_iou(det.bbox, existing.bbox) >= iou_thresh:
                    if det.confidence >= existing.confidence:
                        merged[i] = det
                    replaced = True
                    break
            if not replaced:
                merged.append(det)
    merged.sort(key=lambda d: d.confidence, reverse=True)
    return merged[:12]


def run_camera_detections(frame_bgr: np.ndarray, frame_n: int) -> List[Detection]:
    """AI detection — YOLO ONNX on full frame; HOG/signs fill gaps."""
    global latest_detection_source, _last_ai_fps_count
    if cv2 is None:
        return []
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(
        frame_bgr, (AI_PROCESS_WIDTH, AI_PROCESS_HEIGHT), interpolation=cv2.INTER_LINEAR
    )
    hailo_out = run_hailo_yolo_detector(small)
    if hailo_out:
        latest_detection_source = "hailo"
        _last_ai_fps_count += 1
        return scale_detections_to_frame(hailo_out, AI_PROCESS_WIDTH, AI_PROCESS_HEIGHT, w, h)

    # Full-resolution YOLO for accuracy (model expects 640 blob inside)
    dnn: List[Detection] = []
    if os.path.isfile(AI_DNN_MODEL_PATH) and frame_n % max(1, DNN_EVERY_N_FRAMES) == 0:
        dnn = detect_with_opencv_dnn(frame_bgr)

    hog = detect_with_hog(small)
    signs = detect_colored_sign_objects(frame_bgr)

    groups = [
        dnn,
        scale_detections_to_frame(hog, AI_PROCESS_WIDTH, AI_PROCESS_HEIGHT, w, h) if hog else [],
        signs,
    ]
    # Contours only when YOLO has nothing useful (avoid noisy boxes)
    if not dnn:
        contours = detect_with_contours(small)
        if contours:
            groups.append(
                scale_detections_to_frame(contours, AI_PROCESS_WIDTH, AI_PROCESS_HEIGHT, w, h)
            )

    dets = merge_detections(groups)
    if dnn:
        latest_detection_source = "opencv_dnn"
    elif hog and signs:
        latest_detection_source = "hog+sign"
    elif hog:
        latest_detection_source = "hog"
    elif signs:
        latest_detection_source = "color_sign"
    else:
        latest_detection_source = "none"
    if dets:
        _last_ai_fps_count += 1
    return dets


def simulated_camera_frame_and_detections() -> Tuple[np.ndarray, List[Detection], List[OCRResult]]:
    global sim_phase
    frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    frame[:, :] = (20, 20, 30)
    t = time.time()
    person_x = int(260 + 90 * math.sin(sim_phase))
    chair_x = int(430 + 40 * math.cos(sim_phase * 0.8))
    cv2.rectangle(frame, (person_x, 160), (person_x + 100, 420), (90, 200, 255), 2)
    cv2.rectangle(frame, (chair_x, 260), (chair_x + 120, 430), (90, 255, 90), 2)
    cv2.rectangle(frame, (70, 90), (200, 160), (0, 0, 255), -1)
    cv2.rectangle(frame, (440, 90), (580, 160), (255, 255, 255), -1)
    cv2.putText(frame, "STOP", (86, 138), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, "EXIT", (468, 138), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    dets = [
        Detection("person", 0.84, (person_x, 160, person_x + 100, 420), 1.1, "simulated", t),
        Detection("chair", 0.71, (chair_x, 260, chair_x + 120, 430), 1.4, "simulated", t),
        Detection("stop sign", 0.95, (70, 90, 200, 160), 2.0, "simulated", t),
        Detection("exit sign", 0.92, (440, 90, 580, 160), 2.0, "simulated", t),
    ]
    ocr = [OCRResult("STOP", 0.99, (70, 90, 200, 160), t), OCRResult("EXIT", 0.99, (440, 90, 580, 160), t)]
    return frame, dets, ocr


def choose_camera_alert(detections: List[Detection], ocr_items: List[OCRResult]) -> Tuple[str, str]:
    ocr_keyword = ""
    for item in ocr_items:
        txt = item.text
        if "STOP" in txt:
            ocr_keyword = "STOP"
            break
        if "EXIT" in txt:
            ocr_keyword = "EXIT"
            break
        if "STAIR" in txt:
            ocr_keyword = "STAIR"
            break
    if ocr_keyword:
        return f"OCR_SIGN_{ocr_keyword}", f"OCR sign: {ocr_keyword}"
    if detections:
        best = max(detections, key=lambda d: d.confidence)
        return f"CAMERA_{best.label.upper().replace(' ', '_')}", f"{best.label} {best.confidence:.2f}"
    return "CLEAR", "No detections"


def scale_detections_to_frame(
    dets: List[Detection], src_w: int, src_h: int, dst_w: int, dst_h: int
) -> List[Detection]:
    sx = dst_w / max(1, src_w)
    sy = dst_h / max(1, src_h)
    out: List[Detection] = []
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        out.append(
            Detection(
                d.label,
                d.confidence,
                (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)),
                d.distance_m,
                d.source,
                d.timestamp,
            )
        )
    return out


def _valid_frame(frame: Optional[np.ndarray]) -> bool:
    return (
        frame is not None
        and hasattr(frame, "size")
        and frame.size > 0
        and len(frame.shape) >= 2
        and frame.shape[0] > 10
        and frame.shape[1] > 10
    )


def camera_display_name() -> str:
    """Short label for header/debug."""
    src = camera_source.lower()
    if src == "usb" and active_usb_camera_index is not None:
        return f"USB{active_usb_camera_index}"
    if src in ("picamera2", "picam"):
        return "PiCam"
    if src == "sim":
        return "SIM"
    return "NONE"


def test_usb_camera_index(index: int):
    """Try one USB index; return open VideoCapture or None."""
    if cv2 is None:
        return None
    cap = None
    try:
        if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index)

        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_USB_BUFFER_SIZE)
        if USB_USE_MJPEG and hasattr(cv2, "VideoWriter_fourcc"):
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass

        valid_frame = None
        for _ in range(USB_WARMUP_FRAMES):
            ok, frame = cap.read()
            if ok and _valid_frame(frame):
                valid_frame = frame
            time.sleep(0.03)

        if valid_frame is None:
            cap.release()
            return None
        return cap
    except Exception as exc:
        print(f"USB camera index {index} error: {exc}")
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        return None


def init_picamera2() -> bool:
    """Initialise Pi Camera via Picamera2."""
    global _picam_instance, camera_available, camera_source, camera_error_message, using_explicit_simulation
    if Picamera2 is None or cv2 is None:
        camera_error_message = "Picamera2 not available"
        return False
    try:
        picam = Picamera2()
        try:
            cfg = picam.create_video_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
                controls={"FrameRate": CAMERA_TARGET_FPS},
            )
        except Exception:
            cfg = picam.create_preview_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
            )
        picam.configure(cfg)
        picam.start()
        time.sleep(0.5)
        test = picam.capture_array()
        if not _valid_frame(test):
            raise RuntimeError("Pi Camera test frame invalid")
        _picam_instance = picam
        camera_available = True
        camera_source = "picamera2"
        camera_error_message = ""
        using_explicit_simulation = False
        print("Camera OK: Pi Camera (Picamera2)")
        return True
    except Exception as exc:
        print(f"Pi Camera failed: {exc}")
        camera_error_message = f"Pi Camera failed: {exc}"
        _picam_instance = None
        return False


def init_usb_camera() -> bool:
    """Scan USB_CAMERA_INDEXES and use the first working device."""
    global _usb_cap, camera_available, camera_source, camera_error_message
    global active_usb_camera_index, using_explicit_simulation, ACTIVE_USB_CAMERA_INDEX

    if cv2 is None:
        camera_error_message = "OpenCV not available"
        return False

    active_usb_camera_index = None
    ACTIVE_USB_CAMERA_INDEX = None
    _usb_cap = None

    for idx in USB_CAMERA_INDEXES:
        print(f"Trying USB camera index {idx}...")
        cap = test_usb_camera_index(idx)
        if cap is not None:
            _usb_cap = cap
            camera_available = True
            camera_source = "usb"
            active_usb_camera_index = idx
            ACTIVE_USB_CAMERA_INDEX = idx
            camera_error_message = ""
            using_explicit_simulation = False
            print(f"Camera OK: USB camera index {idx}, {CAMERA_WIDTH}x{CAMERA_HEIGHT}, MJPG")
            return True
        print(f"USB camera index {idx} failed")

    camera_error_message = "No working USB camera found after scanning indexes 0-9"
    print(camera_error_message)
    return False


def release_camera() -> None:
    """Release Pi Camera and USB capture handles."""
    global _picam_instance, _usb_cap, active_usb_camera_index, ACTIVE_USB_CAMERA_INDEX
    if _picam_instance is not None:
        try:
            _picam_instance.stop()
        except Exception:
            pass
        _picam_instance = None
    if _usb_cap is not None:
        try:
            _usb_cap.release()
        except Exception:
            pass
        _usb_cap = None
    active_usb_camera_index = None
    ACTIVE_USB_CAMERA_INDEX = None


def release_camera_source() -> None:
    release_camera()


def init_camera() -> bool:
    """Select working camera per CAMERA_BACKEND and prefer_usb_camera."""
    global camera_available, camera_source, camera_error_message, using_explicit_simulation

    release_camera()
    camera_available = False
    camera_source = "none"
    camera_error_message = ""
    using_explicit_simulation = False

    if not ENABLE_CAMERA:
        camera_error_message = "Camera disabled"
        return False

    backend = CAMERA_BACKEND.lower().strip()
    if FORCE_CAMERA_SIMULATION:
        backend = "sim"

    if backend == "none":
        camera_error_message = "Camera backend set to none"
        return False

    if backend == "sim":
        camera_source = "sim"
        camera_available = True
        using_explicit_simulation = True
        camera_error_message = ""
        print("Camera simulation selected")
        return True

    if backend == "usb":
        return init_usb_camera()

    if backend == "picamera2":
        return init_picamera2()

    if backend == "auto":
        if prefer_usb_camera:
            if init_usb_camera():
                return True
            if init_picamera2():
                return True
        else:
            if init_picamera2():
                return True
            if init_usb_camera():
                return True

    camera_source = "none"
    camera_available = False
    if not camera_error_message:
        camera_error_message = "No working camera found"
    print(f"NO CAMERA: {camera_error_message}")
    return False


def init_camera_source() -> str:
    """Legacy wrapper — returns camera_source label after init."""
    init_camera()
    return camera_source


def grab_camera_frame_bgr() -> Optional[np.ndarray]:
    """Capture one BGR frame from active camera source."""
    src = camera_source.lower()
    if src == "picamera2" and _picam_instance is not None and cv2 is not None:
        try:
            arr = _picam_instance.capture_array()
            if not _valid_frame(arr):
                return None
            if len(arr.shape) == 3 and arr.shape[2] == 3:
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr
        except Exception as exc:
            globals()["camera_error_message"] = f"Pi Camera capture error: {exc}"
            return None
    if src == "usb" and _usb_cap is not None:
        try:
            ok, frame = _usb_cap.read()
            if ok and _valid_frame(frame):
                return frame
            globals()["camera_error_message"] = f"USB camera read failed (index {active_usb_camera_index})"
            return None
        except Exception as exc:
            globals()["camera_error_message"] = f"USB capture error: {exc}"
            return None
    if src == "sim" and using_explicit_simulation:
        frame, _d, _o = simulated_camera_frame_and_detections()
        return frame
    return None


def make_camera_error_frame(message: str) -> np.ndarray:
    """BGR error panel when no real camera frame is available."""
    if cv2 is None:
        return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    frame[:] = (18, 18, 28)
    lines = [
        "NO CAMERA",
        f"Source: {camera_display_name()}",
        message or camera_error_message or "Unknown error",
        "Try: libcamera-hello",
        "Try: rpicam-hello",
        "Try: ls /dev/video*",
        "Try: v4l2-ctl --list-devices",
        "Press C to retry camera",
        "Press 0 quad | 1 camera fullscreen",
    ]
    y = 36
    for i, ln in enumerate(lines):
        col = (0, 0, 220) if i == 0 else (220, 220, 240)
        scale = 1.1 if i == 0 else 0.55
        thick = 3 if i == 0 else 1
        cv2.putText(frame, ln, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)
        y += 44 if i == 0 else 30
    return frame


def draw_camera_overlays(frame_bgr: np.ndarray, dets: List[Detection], ocr_items: List[OCRResult]) -> np.ndarray:
    if cv2 is None:
        return frame_bgr
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    cx1, cy1, cx2, cy2 = get_centre_ocr_box(w, h)
    cv2.rectangle(out, (cx1, cy1), (cx2, cy2), (0, 220, 255), 2)
    cv2.putText(
        out, "OCR SIGN AREA", (cx1 + 4, max(16, cy1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        out, "Hold street/sign text in yellow box — R=read now",
        (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA,
    )

    if ai_overlay_enabled:
        out = draw_ai_detections(out, dets)
    for o in ocr_items:
        x1, y1, x2, y2 = o.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 220, 255), 2)
        label = o.matched_text or o.text
        cv2.putText(
            out, f"TEXT: {label}", (x1, min(out.shape[0] - 8, y2 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 255), 2, cv2.LINE_AA,
        )
    return out


def retry_camera_initialisation() -> None:
    global camera_enabled, _last_camera_retry_time
    print("Retrying camera detection...")
    _last_camera_retry_time = time.time()
    if not camera_enabled:
        camera_enabled = True
    init_camera()


def toggle_camera_priority() -> None:
    global prefer_usb_camera
    prefer_usb_camera = not prefer_usb_camera
    if prefer_usb_camera:
        print("Camera priority: USB first")
    else:
        print("Camera priority: Pi Camera first")
    release_camera()
    init_camera()


def camera_capture_thread_fn() -> None:
    """Fast capture only — never runs OCR or heavy AI."""
    global latest_camera_rgb, latest_raw_camera_bgr, latest_display_camera_rgb
    global latest_frame_time, latest_frame_id, camera_capture_fps, camera_drop_count
    global last_camera_banner, camera_frame_counter, camera_fps
    global _last_camera_fps_count, _last_camera_fps_time, last_successful_frame_time
    global _last_camera_retry_time

    init_camera()
    if not ENABLE_CAMERA:
        return

    while running:
        if not camera_enabled:
            time.sleep(0.05)
            continue

        now = time.time()
        with camera_lock:
            prev_frame_time = latest_frame_time
        if prev_frame_time > 0 and now - prev_frame_time > CAMERA_FREEZE_SECONDS:
            camera_drop_count += 1
            print(f"Camera freeze watchdog: frame stale ({now - prev_frame_time:.1f}s), reopening...")
            _last_camera_retry_time = now
            init_camera()

        frame_bgr = grab_camera_frame_bgr()
        now = time.time()

        if not _valid_frame(frame_bgr):
            if now - _last_camera_retry_time >= CAMERA_RETRY_SECONDS:
                _last_camera_retry_time = now
                init_camera()
                frame_bgr = grab_camera_frame_bgr()

            if not _valid_frame(frame_bgr):
                if SHOW_CAMERA_ERROR_PANEL:
                    err = make_camera_error_frame(camera_error_message)
                    rgb = cv2.cvtColor(err, cv2.COLOR_BGR2RGB) if cv2 is not None else err
                    with camera_lock:
                        latest_camera_rgb = rgb
                time.sleep(CAMERA_SLEEP_SECONDS)
                continue

        if frame_bgr.shape[1] != CAMERA_WIDTH or frame_bgr.shape[0] != CAMERA_HEIGHT:
            if cv2 is not None:
                frame_bgr = cv2.resize(frame_bgr, (CAMERA_WIDTH, CAMERA_HEIGHT))

        with camera_lock:
            latest_raw_camera_bgr = frame_bgr.copy()
            latest_frame_time = now
            latest_frame_id += 1
            dets_snapshot = list(latest_camera_detections)
            ocr_snapshot = list(latest_ocr_results)

        last_successful_frame_time = now
        camera_frame_counter += 1
        _last_camera_fps_count += 1
        if now - _last_camera_fps_time >= 1.0:
            camera_capture_fps = _last_camera_fps_count / (now - _last_camera_fps_time)
            camera_fps = camera_capture_fps
            _last_camera_fps_count = 0
            _last_camera_fps_time = now

        _cam_alert, banner = choose_camera_alert(dets_snapshot, ocr_snapshot)
        last_camera_banner = banner

        display_bgr = draw_camera_overlays(frame_bgr, dets_snapshot, ocr_snapshot)
        frame_age = now - latest_frame_time
        if frame_age > CAMERA_FREEZE_SECONDS and cv2 is not None:
            cv2.putText(
                display_bgr, "CAMERA FRAME STALE", (12, CAMERA_HEIGHT - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2, cv2.LINE_AA,
            )

        rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB) if cv2 is not None else display_bgr
        with camera_lock:
            latest_camera_rgb = rgb
            latest_display_camera_rgb = rgb

        time.sleep(CAMERA_SLEEP_SECONDS)

    release_camera()


def camera_ai_thread_fn() -> None:
    """Periodic AI detection on latest frame snapshot — does not block capture."""
    global latest_camera_detections, ai_inference_fps, latest_detection_source
    global _last_ai_fps_count, _last_ai_fps_time, last_camera_banner
    global AI_HAT_STATUS

    ai_log_counter = 0
    while running:
        if not camera_enabled:
            time.sleep(0.1)
            continue

        time.sleep(AI_DETECTION_INTERVAL_SECONDS)

        with camera_lock:
            frame = latest_raw_camera_bgr.copy() if latest_raw_camera_bgr is not None else None
            frame_id = latest_frame_id
            ocr_snapshot = list(latest_ocr_results)

        if frame is None:
            continue

        dets = run_camera_detections(frame, frame_id)
        with camera_lock:
            latest_camera_detections = dets

        if dets and AI_HAT_STATUS in ("PLACEHOLDER", "RUNTIME_MISSING", "MODEL_MISSING", "OFF"):
            if latest_detection_source in ("opencv_dnn", "hog", "contour"):
                AI_HAT_STATUS = "FALLBACK"

        _cam_alert, banner = choose_camera_alert(dets, ocr_snapshot)
        last_camera_banner = banner

        ai_log_counter += 1
        if ai_log_counter % 4 == 0:
            for d in dets:
                camera_log_rows.append([
                    round(d.timestamp, 3), d.label, f"{d.confidence:.3f}",
                    d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
                    "" if d.distance_m is None else f"{d.distance_m:.3f}", d.source,
                ])

        _last_ai_fps_count += 1
        now = time.time()
        if now - _last_ai_fps_time >= 1.0:
            ai_inference_fps = _last_ai_fps_count / max(0.001, now - _last_ai_fps_time)
            _last_ai_fps_count = 0
            _last_ai_fps_time = now


def camera_ocr_thread_fn() -> None:
    """Periodic OCR on frame snapshot — must not block live camera capture."""
    while running:
        if not camera_enabled or not ocr_enabled:
            time.sleep(0.2)
            continue

        manual = ocr_read_now_event.wait(timeout=OCR_INTERVAL_SECONDS)
        if manual:
            ocr_read_now_event.clear()

        with camera_lock:
            frame = latest_raw_camera_bgr.copy() if latest_raw_camera_bgr is not None else None
            dets = list(latest_camera_detections)

        if frame is None:
            continue

        apply_ocr_scan_results(run_ocr_on_signs(frame, dets), manual=manual)


def request_manual_ocr_read() -> None:
    """Trigger immediate OCR on latest frame (R key) — handled by OCR thread."""
    print("Manual OCR read requested (R)...")
    ocr_read_now_event.set()


def start_camera_workers() -> Tuple[threading.Thread, threading.Thread, threading.Thread]:
    """Start capture, AI, and OCR camera worker threads."""
    capture_thread = threading.Thread(target=camera_capture_thread_fn, daemon=True)
    ai_thread = threading.Thread(target=camera_ai_thread_fn, daemon=True)
    ocr_thread = threading.Thread(target=camera_ocr_thread_fn, daemon=True)
    capture_thread.start()
    ai_thread.start()
    ocr_thread.start()
    return capture_thread, ai_thread, ocr_thread


def panel_rect(col: int, row: int) -> pygame.Rect:
    margin = 8
    inner_top = HEADER_HEIGHT + margin
    inner_bottom = SCREEN_HEIGHT - FOOTER_HEIGHT - margin
    avail_h = inner_bottom - inner_top
    avail_w = SCREEN_WIDTH - margin * 3
    pw = avail_w // 2
    ph = (avail_h - margin) // 2
    x = margin + col * (pw + margin)
    y = inner_top + row * (ph + margin)
    return pygame.Rect(x, y, pw, ph)


def full_content_rect() -> pygame.Rect:
    margin = 8
    inner_top = HEADER_HEIGHT + margin
    inner_bottom = SCREEN_HEIGHT - FOOTER_HEIGHT - margin
    return pygame.Rect(margin, inner_top, SCREEN_WIDTH - margin * 2, inner_bottom - inner_top)


def set_focused_panel(panel_id: int) -> None:
    global focused_panel, view_status_text
    focused_panel = max(0, min(4, panel_id))
    names = {
        0: "Quad view (click a panel title to enlarge)",
        1: "Camera + Sign Reading (full)",
        2: "Obstacle Zones (full)",
        3: "SLAM LiDAR Point Cloud (full)",
        4: "SLAM Floor Plan (full)",
    }
    view_status_text = names.get(focused_panel, "Quad view")
    print(f"View: {view_status_text}")


def panel_id_at_pos(pos: Tuple[int, int]) -> int:
    """Return 1-4 if click is on a quad panel frame, else 0."""
    if focused_panel != 0:
        return 0
    mapping = [
        (1, panel_rect(0, 0)),
        (2, panel_rect(1, 0)),
        (3, panel_rect(0, 1)),
        (4, panel_rect(1, 1)),
    ]
    for pid, rect in mapping:
        if rect.collidepoint(pos):
            return pid
    return 0


def handle_mouse_click(pos: Tuple[int, int]) -> None:
    for rect, action in ui_button_rects:
        if rect.collidepoint(pos):
            handle_ui_button(action)
            return
    pid = panel_id_at_pos(pos)
    if pid:
        set_focused_panel(pid)


def draw_panel_frame(screen: pygame.Surface, rect: pygame.Rect, title: str) -> pygame.Rect:
    pygame.draw.rect(screen, COLOR_PANEL, rect)
    pygame.draw.rect(screen, COLOR_PANEL_BORDER, rect, 2)
    title_text = title
    if focused_panel == 0:
        title_text = f"{title}  [click to enlarge]"
    title_surf = pygame.font.SysFont("monospace", 14, bold=True).render(title_text, True, COLOR_TITLE)
    screen.blit(title_surf, (rect.x + 8, rect.y + 6))
    return pygame.Rect(rect.x + 4, rect.y + 24, rect.width - 8, rect.height - 28)


def draw_ai_camera_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    frame_snapshot: Optional[np.ndarray],
    detections_snapshot: List[Detection],
    ocr_snapshot: List[OCRResult],
) -> None:
    """Draw live camera panel (overlays already baked into frame by camera thread)."""
    fs = 14 if focused_panel == 1 else 12
    font = pygame.font.SysFont("monospace", fs)
    if frame_snapshot is not None:
        surf = pygame.surfarray.make_surface(frame_snapshot.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (rect.width, rect.height))
        screen.blit(surf, rect.topleft)
    else:
        pygame.draw.rect(screen, (30, 35, 45), rect)
        screen.blit(font.render("NO CAMERA FRAME", True, COLOR_RED), (rect.x + 10, rect.y + 20))

    frame_age = time.time() - latest_frame_time if latest_frame_time else -1.0
    y = rect.y + 4
    cap_fps = camera_capture_fps if camera_capture_fps > 0 else camera_fps
    screen.blit(
        font.render(
            f"Source:{camera_display_name()}  {cap_fps:.0f}fps  AI:{ai_inference_fps:.0f}fps  "
            f"dets:{len(detections_snapshot)}  age:{frame_age:.1f}s",
            True, COLOR_CYAN,
        ),
        (rect.x + 6, y),
    )
    y += fs + 4
    ocr_stat = "ON" if ocr_enabled and pytesseract else "OFF"
    ocr_txt = ocr_snapshot[0].text if ocr_snapshot else (last_ocr_text or "--")
    ai_hat_lbl = AI_HAT_STATUS if ENABLE_AI_HAT else "OFF"
    screen.blit(
        font.render(
            f"OCR:{ocr_stat}  Last:{ocr_txt}  AI HAT:{ai_hat_lbl}  Det:{latest_detection_source}",
            True, COLOR_YELLOW,
        ),
        (rect.x + 6, y),
    )
    y += fs + 4
    vote_key = ""
    vote_n = 0
    if ocr_vote_counts:
        vote_key, vote_n = max(ocr_vote_counts.items(), key=lambda kv: kv[1])
    dbg_fs = max(10, fs - 2)
    dbg_font = pygame.font.SysFont("monospace", dbg_fs)
    ocr_lines = [
        f"OCR raw: {ocr_debug_raw or '--'}",
        f"OCR cleaned: {ocr_debug_cleaned or '--'}",
        f"OCR matched: {ocr_debug_matched or '--'}",
        f"OCR confirmed: {ocr_debug_confirmed or confirmed_sign_text or '--'}",
        f"OCR votes: {vote_key or '--'} {vote_n}/{OCR_REQUIRE_STABLE_READS}",
    ]
    for ln in ocr_lines:
        screen.blit(dbg_font.render(ln, True, COLOR_YELLOW), (rect.x + 6, y))
        y += dbg_fs + 2
    if frame_age > CAMERA_FREEZE_SECONDS:
        screen.blit(font.render("CAMERA FRAME STALE", True, COLOR_ORANGE), (rect.x + 6, y))
        y += fs + 4
    if camera_error_message and camera_source.lower() == "none":
        screen.blit(font.render(camera_error_message[:52], True, COLOR_ORANGE), (rect.x + 6, y))
        y += fs + 4
    if focused_panel == 1:
        by = rect.bottom - 64
        for ln in (
            "0=quad  2=zones  3=LiDAR  4=floor-plan",
            "C=retry  X=cam  P=USB/PiCam  R=read sign  M=reset map",
        ):
            screen.blit(font.render(ln, True, COLOR_TEXT), (rect.x + 6, by))
            by += fs + 2


def zone_level(distance_m: Optional[float]) -> Tuple[str, Tuple[int, int, int]]:
    if distance_m is None:
        return "clear", COLOR_GREEN
    if distance_m <= VERY_CLOSE_DISTANCE_M:
        return "stop", COLOR_RED
    if distance_m <= STRONG_WARNING_DISTANCE_M:
        return "strong", COLOR_ORANGE
    if distance_m <= ALERT_DISTANCE_M:
        return "alert", COLOR_YELLOW
    return "clear", COLOR_GREEN


def draw_obstacle_zones(screen: pygame.Surface, rect: pygame.Rect) -> None:
    font_big = pygame.font.SysFont("monospace", 16, bold=True)
    font_small = pygame.font.SysFont("monospace", 12)
    cx, cy = rect.centerx, rect.centery
    w2, h2 = rect.width // 2, rect.height // 2
    zones = {
        "FRONT": pygame.Rect(cx - w2 // 2, rect.y + 8, w2, h2 - 12),
        "LEFT": pygame.Rect(rect.x + 8, cy - h2 // 2, w2 - 12, h2),
        "RIGHT": pygame.Rect(cx + 4, cy - h2 // 2, w2 - 12, h2),
        "BACK": pygame.Rect(cx - w2 // 2, cy + 4, w2, h2 - 12),
    }
    dists = {
        "FRONT": direction_distances["front"],
        "LEFT": direction_distances["left"],
        "RIGHT": direction_distances["right"],
        "BACK": direction_distances["back"],
    }
    counts = {
        "FRONT": last_zone_counts["front"],
        "LEFT": last_zone_counts["left"],
        "RIGHT": last_zone_counts["right"],
        "BACK": last_zone_counts["back"],
    }
    for name, zrect in zones.items():
        lvl, col = zone_level(dists[name])
        pygame.draw.rect(screen, tuple(min(255, c // 2 + 25) for c in col), zrect, border_radius=8)
        pygame.draw.rect(screen, col, zrect, 2, border_radius=8)
        title = f"{name}  {lvl.upper()}  {counts[name]} pts"
        dist_txt = "--" if dists[name] is None else f"{dists[name]:.2f} m"
        screen.blit(font_big.render(title, True, (8, 12, 15)), (zrect.x + 6, zrect.y + 6))
        screen.blit(font_small.render(dist_txt, True, (8, 12, 15)), (zrect.x + 6, zrect.y + 28))


def world_to_panel(x_m: float, y_m: float, rect: pygame.Rect) -> Tuple[int, int]:
    sx = int(rect.centerx + y_m * pixels_per_meter)
    sy = int(rect.centery - x_m * pixels_per_meter)
    return sx, sy


def world_to_slam_panel(x_m: float, y_m: float, rect: pygame.Rect) -> Tuple[int, int]:
    """Slight isometric skew so the 2D scan reads more like a SLAM point cloud."""
    skew = SLAM_LIDAR_ISO_SKEW
    sx = int(rect.centerx + (y_m + x_m * skew * 0.35) * pixels_per_meter)
    sy = int(rect.centery - (x_m - abs(y_m) * skew * 0.15) * pixels_per_meter)
    return sx, sy


def slam_distance_color(d_m: float) -> Tuple[int, int, int]:
    """Rainbow heatmap: violet/blue near → cyan/green mid → yellow/orange/red far."""
    t = max(0.0, min(1.0, d_m / SLAM_LIDAR_MAX_COLOR_DIST_M))
    # 5-stop gradient matching typical LiDAR SLAM visuals
    stops = [
        (0.00, (120, 40, 200)),   # violet
        (0.20, (40, 80, 255)),    # blue
        (0.40, (30, 220, 230)),   # cyan
        (0.60, (60, 220, 80)),    # green
        (0.80, (245, 210, 40)),   # yellow
        (1.00, (240, 60, 50)),    # red
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            u = (t - t0) / max(1e-6, t1 - t0)
            return (
                int(c0[0] + (c1[0] - c0[0]) * u),
                int(c0[1] + (c1[1] - c0[1]) * u),
                int(c0[2] + (c1[2] - c0[2]) * u),
            )
    return stops[-1][1]


def draw_lidar_distance_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    latest_scan_snapshot: List[Tuple[float, float, float, float]],
) -> None:
    """SLAM-style dense rainbow point cloud on black (visual style; sensor remains 2D)."""
    pygame.draw.rect(screen, (0, 0, 0), rect)

    # Soft range rings (scan rings like multi-line LiDAR visuals)
    for r in (1.0, 2.0, 3.0, 4.0, 5.0):
        rad = int(r * pixels_per_meter)
        if rad > 2:
            pygame.draw.circle(screen, (28, 32, 40), rect.center, rad, 1)

    # Dense history cloud first (older / dimmer)
    with data_lock:
        history_snap = [list(scan) for scan in slam_lidar_history]

    n_hist = max(1, len(history_snap))
    for hi, scan in enumerate(history_snap):
        age = hi / n_hist  # older = lower
        fade = 0.35 + 0.65 * age
        for x_m, y_m, d_m, _a in scan:
            px, py = world_to_slam_panel(x_m, y_m, rect)
            if not rect.collidepoint(px, py):
                continue
            r, g, b = slam_distance_color(d_m)
            col = (int(r * fade), int(g * fade), int(b * fade))
            pygame.draw.circle(screen, col, (px, py), 1)

    # Latest scan brighter and slightly larger
    for x_m, y_m, d_m, _a in latest_scan_snapshot:
        px, py = world_to_slam_panel(x_m, y_m, rect)
        if not rect.collidepoint(px, py):
            continue
        col = slam_distance_color(d_m)
        pygame.draw.circle(screen, col, (px, py), SLAM_LIDAR_POINT_RADIUS)

    # Sensor origin
    pygame.draw.circle(screen, (180, 255, 200), rect.center, 4)
    pygame.draw.circle(screen, (80, 200, 120), rect.center, 4, 1)

    # Caption / colour key
    font = pygame.font.SysFont("monospace", 11)
    screen.blit(
        font.render("SLAM-style LiDAR cloud  near=violet  far=red", True, (170, 180, 200)),
        (rect.x + 6, rect.bottom - 16),
    )
    # Mini colour bar
    bar_y = rect.y + 6
    bar_x = rect.right - 118
    for i in range(100):
        c = slam_distance_color(i / 100.0 * SLAM_LIDAR_MAX_COLOR_DIST_M)
        pygame.draw.line(screen, c, (bar_x + i, bar_y), (bar_x + i, bar_y + 8))
    screen.blit(font.render("0m", True, (150, 160, 180)), (bar_x - 14, bar_y - 1))
    screen.blit(font.render("5m", True, (150, 160, 180)), (bar_x + 102, bar_y - 1))


def connected_components(cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    return get_connected_wall_components(cells)


def get_connected_wall_components(occupied_cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """BFS wall components — filter noise blobs smaller than MIN_WALL_COMPONENT_SIZE."""
    cell_set = set(occupied_cells)
    components = []
    visited = set()
    neigh = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for c in cell_set:
        if c in visited:
            continue
        q = deque([c])
        visited.add(c)
        comp = [c]
        while q:
            x, y = q.popleft()
            for dx, dy in neigh:
                n = (x + dx, y + dy)
                if n in cell_set and n not in visited:
                    visited.add(n)
                    q.append(n)
                    comp.append(n)
        if len(comp) >= MIN_WALL_COMPONENT_SIZE:
            components.append(comp)
    return components


def world_to_floorplan_panel(
    x_m: float,
    y_m: float,
    rect: pygame.Rect,
    view_x: float,
    view_y: float,
    scale: float,
) -> Tuple[int, int]:
    """Map world metres to panel pixels; view centred on robot (or map focus)."""
    sx = int(rect.centerx + (y_m - view_y) * scale)
    sy = int(rect.centery - (x_m - view_x) * scale)
    return sx, sy


def draw_local_room_map(
    screen: pygame.Surface,
    rect: pygame.Rect,
    free_grid_snapshot: Dict[Tuple[int, int], int],
    occupied_grid_snapshot: Dict[Tuple[int, int], int],
    latest_scan_snapshot: List[Tuple[float, float, float, float]],
) -> None:
    """MathWorks-style occupancy floor plan + beginner SLAM trajectory."""
    # Unknown = mid grey (MATLAB occupancy unknown look)
    pygame.draw.rect(screen, (90, 90, 95), rect)

    with data_lock:
        pose = list(robot_pose)
        path = list(pose_history)
        status = slam_status_text
        scans_n = slam_scan_count
        match_err = slam_last_match_err

    view_x, view_y = pose[0], pose[1]
    scale = pixels_per_meter

    # Free space (light) — floor-plan walkable area
    for (ix, iy), fh in free_grid_snapshot.items():
        if fh < FREE_MIN_HITS:
            continue
        if occupied_grid_snapshot.get((ix, iy), 0) >= OCCUPIED_MIN_HITS:
            continue
        wx = ix * GRID_RESOLUTION_M
        wy = iy * GRID_RESOLUTION_M
        px, py = world_to_floorplan_panel(wx, wy, rect, view_x, view_y, scale)
        if rect.collidepoint(px, py):
            pygame.draw.rect(screen, (235, 235, 238), pygame.Rect(px - 1, py - 1, 3, 3))

    # Occupied walls (dark) — floor-plan walls
    for (ix, iy), hits in occupied_grid_snapshot.items():
        if hits < OCCUPIED_MIN_HITS:
            continue
        wx = ix * GRID_RESOLUTION_M
        wy = iy * GRID_RESOLUTION_M
        px, py = world_to_floorplan_panel(wx, wy, rect, view_x, view_y, scale)
        if not rect.collidepoint(px, py):
            continue
        if hits >= WALL_STRONG_HITS:
            col = (20, 20, 24)
            size = 4
        else:
            col = (50, 50, 58)
            size = 3
        pygame.draw.rect(screen, col, pygame.Rect(px - size // 2, py - size // 2, size, size))

    # Current scan in world frame (cyan dots)
    for x_r, y_r, d_m, _a in latest_scan_snapshot:
        wx, wy = robot_to_world(x_r, y_r, pose)
        px, py = world_to_floorplan_panel(wx, wy, rect, view_x, view_y, scale)
        if rect.collidepoint(px, py):
            col = COLOR_RED if d_m <= VERY_CLOSE_DISTANCE_M else (80, 180, 255)
            pygame.draw.circle(screen, col, (px, py), 2)

    # Trajectory (pose graph path)
    if len(path) >= 2:
        pts = []
        for x, y, _th in path:
            px, py = world_to_floorplan_panel(x, y, rect, view_x, view_y, scale)
            if rect.collidepoint(px, py):
                pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(screen, (40, 140, 255), False, pts, 2)

    # Robot pose marker (triangle: +x forward = up on screen)
    rx, ry = world_to_floorplan_panel(pose[0], pose[1], rect, view_x, view_y, scale)
    yaw = pose[2]
    tip = (rx + int(12 * math.sin(yaw)), ry - int(12 * math.cos(yaw)))
    left = (rx + int(7 * math.sin(yaw + 2.4)), ry - int(7 * math.cos(yaw + 2.4)))
    right = (rx + int(7 * math.sin(yaw - 2.4)), ry - int(7 * math.cos(yaw - 2.4)))
    pygame.draw.polygon(screen, (40, 200, 90), [tip, left, right])
    pygame.draw.circle(screen, (20, 120, 50), (rx, ry), 3)

    font = pygame.font.SysFont("monospace", 11)
    screen.blit(
        font.render(
            f"Floor plan SLAM  scans:{scans_n}  pose:({pose[0]:.1f},{pose[1]:.1f}) "
            f"yaw:{math.degrees(pose[2]):.0f}°  {status}",
            True,
            (30, 30, 35),
        ),
        (rect.x + 6, rect.y + 4),
    )
    err_txt = "--" if match_err < 0 else f"{match_err:.2f}m"
    screen.blit(
        font.render(f"match err:{err_txt}  M=reset map  white=free  black=wall  blue=path", True, (40, 40, 48)),
        (rect.x + 6, rect.bottom - 16),
    )


def draw_header(screen: pygame.Surface, fused_alert: str) -> None:
    pygame.draw.rect(screen, (12, 18, 26), pygame.Rect(0, 0, SCREEN_WIDTH, HEADER_HEIGHT))
    font = pygame.font.SysFont("monospace", 15, bold=True)
    lidar_state = "SIM" if (SIMULATED_MODE or not ENABLE_LIDAR) else ("LIVE" if lidar_enabled else "OFF")
    cam_state = camera_display_name()
    if ENABLE_AI_HAT:
        ai_state = AI_HAT_STATUS
    else:
        ai_state = "OFF"
    if ACTIVE_USB_CAMERA_INDEX is not None and camera_source == "usb":
        cam_state = f"USB{ACTIVE_USB_CAMERA_INDEX}"
    text = (
        f"LiDAR {lidar_state} | Camera {cam_state} | "
        f"AI HAT {ai_state} | OCR {'ON' if ocr_enabled else 'OFF'} | "
        f"Voice {'ON' if voice_enabled else 'OFF'} | Alert {fused_alert} | {view_status_text}"
    )
    screen.blit(font.render(text, True, COLOR_TEXT), (10, 16))


def handle_ui_button(action: str) -> None:
    if action == "voice_test":
        test_voice()
    elif action == "voice_toggle":
        toggle_voice_enabled()
    elif action == "voice_mute":
        toggle_temporary_mute()
    elif action == "voice_sign":
        test_voice_sign()
    elif action == "voice_obstacle":
        test_voice_obstacle()
    elif action == "voice_stop":
        test_voice_stop()


def draw_voice_ui_buttons(screen: pygame.Surface) -> None:
    global ui_button_rects
    mute_left = voice_mute_remaining_seconds()
    mute_label = f"Mute {mute_left:.0f}s" if mute_left > 0 else "Mute 30s"
    labels = [
        ("Voice Test", "voice_test"),
        ("Voice On/Off", "voice_toggle"),
        (mute_label, "voice_mute"),
        ("Test Sign", "voice_sign"),
        ("Test Alert", "voice_obstacle"),
        ("Test STOP", "voice_stop"),
    ]
    ui_button_rects = []
    btn_w, btn_h, gap = 100, 30, 5
    total_w = len(labels) * btn_w + (len(labels) - 1) * gap
    x0 = max(8, (SCREEN_WIDTH - total_w) // 2)
    y0 = SCREEN_HEIGHT - FOOTER_HEIGHT + 6
    font = pygame.font.SysFont("monospace", 11, bold=True)
    for i, (label, action) in enumerate(labels):
        rect = pygame.Rect(x0 + i * (btn_w + gap), y0, btn_w, btn_h)
        if action == "voice_toggle" and not voice_enabled:
            bg = (80, 45, 45)
        elif action == "voice_mute" and mute_left > 0:
            bg = (140, 90, 30)
        else:
            bg = (35, 95, 140)
        pygame.draw.rect(screen, bg, rect)
        pygame.draw.rect(screen, (120, 200, 255), rect, 2)
        txt = font.render(label, True, (235, 245, 255))
        screen.blit(txt, txt.get_rect(center=rect.center))
        ui_button_rects.append((rect, action))


def draw_footer(screen: pygame.Surface) -> None:
    pygame.draw.rect(screen, (12, 18, 26), pygame.Rect(0, SCREEN_HEIGHT - FOOTER_HEIGHT, SCREEN_WIDTH, FOOTER_HEIGHT))
    draw_voice_ui_buttons(screen)
    controls = (
        "0 Quad | 1 Camera | 2 Zones | 3 LiDAR | 4 Floor Plan | "
        "C Retry | X Cam | P USB/PiCam | R Read Sign | M Reset Map | N Mute 30s | "
        "O OCR | V Voice | D Debug | S Save | Q Quit"
    )
    font = pygame.font.SysFont("monospace", 11)
    screen.blit(font.render(controls, True, COLOR_MUTED), (10, SCREEN_HEIGHT - 14))


def draw_debug_panel(
    screen: pygame.Surface,
    fused_alert: str,
    detections_count: int,
    scan_points_count: int,
) -> None:
    if not debug_enabled:
        return
    rect = pygame.Rect(12, HEADER_HEIGHT + 10, 540, 500)
    overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 170))
    screen.blit(overlay, rect.topleft)
    pygame.draw.rect(screen, (100, 170, 220), rect, 2)
    font = pygame.font.SysFont("monospace", 12)
    since_voice = time.time() - last_voice_time if last_voice_time > 0 else -1.0
    frame_age = time.time() - latest_frame_time if latest_frame_time else -1.0
    model_exists = os.path.isfile(AI_MODEL_PATH)
    lines = [
        "--- Camera ---",
        f"CAMERA_BACKEND={CAMERA_BACKEND} PREFER_USB_CAMERA={prefer_usb_camera}",
        f"ACTIVE_USB_CAMERA_INDEX={ACTIVE_USB_CAMERA_INDEX} active_usb={active_usb_camera_index}",
        f"camera_enabled={camera_enabled} camera_available={camera_available} source={camera_source}",
        f"display={camera_display_name()} camera_error={camera_error_message or '-'}",
        f"latest_frame_age={frame_age:.2f}s  capture_fps={camera_capture_fps:.1f} "
        f"display_fps={camera_fps:.1f} ai_fps={ai_inference_fps:.1f}",
        f"camera_drop_count={camera_drop_count} frame_id={latest_frame_id}",
        f"ocr_enabled={ocr_enabled} ocr_text={last_ocr_text or '-'} "
        + (f"ocr_age={(time.time() - last_ocr_update_time):.1f}s" if last_ocr_update_time else "ocr_age=-"),
        f"detections={detections_count} overlay={ai_overlay_enabled} det_src={latest_detection_source}",
        f"HAILO_AVAILABLE={HAILO_AVAILABLE} model_exists={model_exists}",
        f"ai_hat_active={ai_hat_active} AI_HAT_STATUS={AI_HAT_STATUS}",
        f"display_alert={fused_alert} last_spoken={last_spoken_message or '-'}",
        f"tts_backend={tts_backend} voice_enabled={voice_enabled} "
        f"mute_left={voice_mute_remaining_seconds():.0f}s speaking={is_voice_speaking()}",
        f"seconds_since_last_voice={since_voice:.1f}",
        "--- OCR sign (v10) ---",
        f"ocr_raw={ocr_debug_raw or '-'}",
        f"ocr_cleaned={ocr_debug_cleaned or '-'}",
        f"ocr_matched={ocr_debug_matched or '-'}",
        f"ocr_confirmed={ocr_debug_confirmed or confirmed_sign_text or '-'}",
        f"ocr_votes={ocr_vote_counts}",
        f"ocr_candidates={', '.join(ocr_last_candidates[:6]) or '-'}",
        f"OCR_USE_SIGN_VOCABULARY={OCR_USE_SIGN_VOCABULARY} "
        f"OCR_VOCAB_CORRECT={OCR_VOCABULARY_CORRECTION} "
        f"stable={OCR_REQUIRE_STABLE_READS}/{OCR_STABLE_WINDOW_SECONDS:.0f}s",
        "--- Sign / OCR voice ---",
        f"sign_candidate={sign_candidate_text or '-'} ({sign_candidate_count}/{SIGN_CONFIRM_DETECTIONS})",
        f"confirmed_sign={confirmed_sign_text or '-'} last_spoken_sign={last_spoken_sign_text or '-'}",
        "--- Camera object voice ---",
        f"object_candidate={object_candidate_label or '-'} {object_candidate_direction or ''} "
        f"({object_candidate_count}/{CAMERA_OBJECT_CONFIRM_DETECTIONS})",
        f"confirmed_object={confirmed_object_label or '-'} {confirmed_object_direction or ''}",
        f"surroundings={confirmed_surroundings or '-'}",
        f"last_spoken_object={last_spoken_object_alert or '-'}",
        "--- LiDAR obstacle voice ---",
        f"lidar_raw={raw_lidar_alert} candidate={lidar_candidate_alert} "
        f"({lidar_candidate_count}/{LIDAR_CONFIRM_SCANS})",
        f"confirmed_lidar={confirmed_lidar_alert} last_spoken_lidar={last_spoken_lidar_alert or '-'}",
        f"clear_streak={lidar_clear_streak}/{CLEAR_CONFIRM_SCANS} danger_was={last_spoken_was_danger}",
        f"lidar_pkts={scan_points_count}",
        f"zones F/L/R/B={last_zone_counts['front']}/{last_zone_counts['left']}/"
        f"{last_zone_counts['right']}/{last_zone_counts['back']}",
        "--- Beginner SLAM ---",
        f"pose=({robot_pose[0]:.2f},{robot_pose[1]:.2f}) "
        f"yaw={math.degrees(robot_pose[2]):.1f}° scans={slam_scan_count}",
        f"match_err={slam_last_match_err:.2f} status={slam_status_text}",
        f"path_pts={len(pose_history)} grid_occ={len(occupied_grid)} free={len(free_grid)}",
    ]
    y = rect.y + 8
    for ln in lines:
        screen.blit(font.render(ln, True, (200, 230, 250)), (rect.x + 10, y))
        y += 20


def save_dashboard_and_csv(screen: pygame.Surface) -> None:
    pygame.image.save(screen, DASHBOARD_PNG)
    with open(LIDAR_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "angle_deg", "distance_cm", "x_m", "y_m", "distance_m"])
        w.writerows(lidar_log_rows[-20000:])
    with open(OCCUPANCY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ix", "iy", "occupied_hits", "free_hits"])
        with data_lock:
            occ_snap = dict(occupied_grid)
            free_snap = dict(free_grid)
        keys = set(occ_snap.keys()) | set(free_snap.keys())
        for k in sorted(keys):
            w.writerow([k[0], k[1], occ_snap.get(k, 0), free_snap.get(k, 0)])
    with open(CAMERA_DETECTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "label", "kind", "confidence", "distance_m", "x", "y", "w", "h", "source", "text"])
        for row in camera_log_rows[-20000:]:
            if len(row) >= 9:
                w.writerow([row[0], row[1], "object", row[2], row[7], row[3], row[4],
                            row[5] - row[3] if isinstance(row[5], int) else 0,
                            row[6] - row[4] if isinstance(row[6], int) else 0, row[8], ""])
    with open(OCR_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "text", "confidence", "x1", "y1", "x2", "y2"])
        w.writerows(ocr_log_rows[-20000:])
    print(f"Saved: {DASHBOARD_PNG}, {LIDAR_CSV}, {OCCUPANCY_CSV}, {CAMERA_DETECTIONS_CSV}, {OCR_CSV}")


def handle_keydown(key: int, screen: pygame.Surface) -> Optional[pygame.Surface]:
    global running, camera_enabled, voice_enabled, ocr_enabled, ai_overlay_enabled, lidar_enabled
    global zones_fullscreen, debug_enabled, simulation_paused, fusion_enabled, pixels_per_meter, fullscreen
    if key == pygame.K_q or key == pygame.K_ESCAPE:
        running = False
    elif key == pygame.K_c:
        retry_camera_initialisation()
    elif key == pygame.K_x:
        camera_enabled = not camera_enabled
        print(f"Camera {'ON' if camera_enabled else 'OFF'}")
        if camera_enabled:
            retry_camera_initialisation()
    elif key == pygame.K_p:
        toggle_camera_priority()
    elif key == pygame.K_r:
        request_manual_ocr_read()
    elif key == pygame.K_m:
        reset_slam_map()
    elif key == pygame.K_n:
        toggle_temporary_mute()
    elif key == pygame.K_s:
        save_dashboard_and_csv(screen)
    elif key == pygame.K_v:
        voice_enabled = not voice_enabled
        if not voice_enabled:
            stop_current_voice()
    elif key == pygame.K_t:
        test_voice()
    elif key == pygame.K_o:
        ocr_enabled = not ocr_enabled
        print(f"OCR {'ON' if ocr_enabled else 'OFF'}")
    elif key == pygame.K_i:
        ai_overlay_enabled = not ai_overlay_enabled
    elif key == pygame.K_l:
        lidar_enabled = not lidar_enabled
    elif key == pygame.K_z:
        zones_fullscreen = not zones_fullscreen
        print(f"Zones overlay {'ON' if zones_fullscreen else 'OFF'}")
    elif key == pygame.K_d:
        debug_enabled = not debug_enabled
    elif key == pygame.K_u:
        fusion_enabled = not fusion_enabled
        print(f"Camera-LiDAR fusion {'ON' if fusion_enabled else 'OFF'}")
    elif key == pygame.K_f:
        fullscreen = not fullscreen
        if fullscreen:
            return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        return pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    elif key == pygame.K_0 or key == pygame.K_KP0:
        set_focused_panel(0)
    elif key == pygame.K_1 or key == pygame.K_KP1:
        set_focused_panel(1)
    elif key == pygame.K_2 or key == pygame.K_KP2:
        set_focused_panel(2)
    elif key == pygame.K_3 or key == pygame.K_KP3:
        set_focused_panel(3)
    elif key == pygame.K_4 or key == pygame.K_KP4:
        set_focused_panel(4)
    elif key == pygame.K_SPACE:
        simulation_paused = not simulation_paused
    elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        pixels_per_meter = min(180.0, pixels_per_meter + 8.0)
    elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        pixels_per_meter = max(40.0, pixels_per_meter - 8.0)
    return None


def main() -> None:
    global ai_labels
    resolve_ai_model_paths()
    ai_labels = load_labels(AI_LABELS_PATH)
    print_voice_settings()
    print_camera_settings()
    init_voice_system()
    init_ocr_system()
    init_ai_hat()

    pygame.init()
    pygame.display.set_caption("Team Bravo Vision Assistant v10 beginner SLAM floor plan")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    print("Pi 5 v10: beginner SLAM | numbers+streets OCR | objects around you | obstacle 20s | Mute 30s / N")

    if voice_enabled:
        pygame.time.wait(300)
        test_voice()
    clock = pygame.time.Clock()

    lt = threading.Thread(target=lidar_thread_fn, daemon=True)
    lt.start()
    start_camera_workers()

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    new_screen = handle_keydown(event.key, screen)
                    if new_screen is not None:
                        screen = new_screen
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    handle_mouse_click(event.pos)

            fused_alert = process_voice_alerts()

            # Brief snapshots only — never hold locks while drawing
            with data_lock:
                free_grid_snapshot = dict(free_grid)
                occupied_grid_snapshot = dict(occupied_grid)
                latest_scan_snapshot = list(latest_scan_points)

            with camera_lock:
                camera_frame_snapshot = (
                    latest_camera_rgb.copy() if latest_camera_rgb is not None else None
                )
                camera_dets_snapshot = list(latest_camera_detections)
                ocr_snapshot = list(latest_ocr_results)

            screen.fill(COLOR_BG)

            if focused_panel == 0:
                p00 = panel_rect(0, 0)
                p10 = panel_rect(1, 0)
                p01 = panel_rect(0, 1)
                p11 = panel_rect(1, 1)

                r = draw_panel_frame(screen, p00, "AI Camera Detection + Sign Reading")
                draw_ai_camera_panel(screen, r, camera_frame_snapshot, camera_dets_snapshot, ocr_snapshot)

                r = draw_panel_frame(screen, p10, "Obstacle Zones")
                draw_obstacle_zones(screen, r)

                r = draw_panel_frame(screen, p01, "SLAM LiDAR Point Cloud")
                draw_lidar_distance_panel(screen, r, latest_scan_snapshot)

                r = draw_panel_frame(screen, p11, "SLAM Floor Plan")
                draw_local_room_map(screen, r, free_grid_snapshot, occupied_grid_snapshot, latest_scan_snapshot)
            else:
                full = full_content_rect()
                if focused_panel == 1:
                    r = draw_panel_frame(screen, full, "AI Camera Detection + Sign Reading")
                    draw_ai_camera_panel(screen, r, camera_frame_snapshot, camera_dets_snapshot, ocr_snapshot)
                elif focused_panel == 2:
                    r = draw_panel_frame(screen, full, "Obstacle Zones")
                    draw_obstacle_zones(screen, r)
                elif focused_panel == 3:
                    r = draw_panel_frame(screen, full, "SLAM LiDAR Point Cloud")
                    draw_lidar_distance_panel(screen, r, latest_scan_snapshot)
                elif focused_panel == 4:
                    r = draw_panel_frame(screen, full, "SLAM Floor Plan")
                    draw_local_room_map(screen, r, free_grid_snapshot, occupied_grid_snapshot, latest_scan_snapshot)

            if zones_fullscreen:
                full = pygame.Rect(8, HEADER_HEIGHT + 8, SCREEN_WIDTH - 16, SCREEN_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT - 16)
                overlay = pygame.Surface((full.width, full.height), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 120))
                screen.blit(overlay, full.topleft)
                draw_obstacle_zones(screen, full)

            draw_header(screen, fused_alert)
            draw_footer(screen)
            draw_debug_panel(screen, fused_alert, len(camera_dets_snapshot), len(latest_scan_snapshot))
            pygame.display.flip()
            clock.tick(FPS)
    finally:
        globals()["running"] = False
        stop_current_voice()
        release_camera_source()
        time.sleep(0.05)
        pygame.quit()


if __name__ == "__main__":
    main()

    main()
