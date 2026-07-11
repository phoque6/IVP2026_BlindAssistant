#!/usr/bin/env python3
"""
Team Bravo Vision Assistant v7 — better sign OCR, vocabulary matching, stable reads
==================================================================================

v7: Sign-aware OCR with vocabulary fuzzy match, scoring, voting, centre OCR box.

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
    python3 team_bravo_aihat_camera_lidar_vision_assistant_v7_better_sign_ocr.py

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
USB_CAMERA_INDEXES = (0, 1, 2, 3, 4, 5)
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

AI_MODEL_PATH = "models/yolov8n.hef"
AI_LABELS_PATH = "models/coco_labels.txt"
AI_DNN_MODEL_PATH = "models/yolov8n.onnx"
AI_CONFIDENCE_THRESHOLD = 0.45
AI_NMS_THRESHOLD = 0.40
AI_PROCESS_WIDTH = 416
AI_PROCESS_HEIGHT = 312
DNN_EVERY_N_FRAMES = 8

# Time-based camera processing (decoupled from capture rate)
OCR_INTERVAL_SECONDS = 5.0
OCR_VOICE_REPEAT_SECONDS = 5.0
OCR_PERSIST_SECONDS = 8.0
OCR_FRAME_SCALE = 1.5
AI_DETECTION_INTERVAL_SECONDS = 0.5
CAMERA_SLEEP_SECONDS = 0.001

OCR_MIN_TEXT_LENGTH = 2
OCR_MAX_TEXT_LENGTH = 20
OCR_WHOLE_FRAME_FALLBACK = False
OCR_USE_SIGN_VOCABULARY = True
OCR_MIN_CONFIDENCE_SCORE = 0.55
OCR_REQUIRE_STABLE_READS = 2
OCR_STABLE_WINDOW_SECONDS = 10.0
OCR_FUZZY_MATCH_THRESHOLD = 0.65

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

# Sign / OCR voice — require 2 stable reads before speaking
SIGN_CONFIRM_DETECTIONS = 2
SIGN_REPEAT_SECONDS = 5.0

# Camera object voice
CAMERA_OBJECT_CONFIRM_DETECTIONS = 10
CAMERA_OBJECT_REPEAT_SECONDS = 10.0

# LiDAR obstacle voice
LIDAR_CONFIRM_SCANS = 10
OBSTACLE_REPEAT_SECONDS = 10.0
VERY_CLOSE_REPEAT_SECONDS = 5.0

# Path clear voice
CLEAR_CONFIRM_SCANS = 15
CLEAR_REPEAT_SECONDS = 20.0

ESPEAK_SPEED = 155
ESPEAK_AMPLITUDE = 180
ESPEAK_WORD_GAP_MS = 6

USEFUL_OBJECT_LABELS = (
    "person", "chair", "table", "door", "backpack", "handbag", "bottle", "bag", "obstacle", "sign",
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
ocr_enabled = ENABLE_OCR
ai_overlay_enabled = True
lidar_enabled = ENABLE_LIDAR
camera_enabled = ENABLE_CAMERA
pixels_per_meter = PIXELS_PER_METER_DEFAULT

data_lock = threading.Lock()
camera_lock = threading.Lock()

latest_scan_points: List[Tuple[float, float, float, float]] = []
latest_polar_points: List[Tuple[float, float]] = []
occupied_grid: Dict[Tuple[int, int], int] = {}
free_grid: Dict[Tuple[int, int], int] = {}
last_zone_counts = {"front": 0, "left": 0, "right": 0, "back": 0}
direction_distances = {"front": None, "left": None, "right": None, "back": None}

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

# OCR debug + voting (v7 sign-aware)
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

# Camera object voice state
object_candidate_label = ""
object_candidate_direction = ""
object_candidate_count = 0
confirmed_object_label = ""
confirmed_object_direction = ""
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
        return []
    labels = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                txt = line.strip()
                if txt:
                    labels.append(txt)
    except OSError:
        return []
    return labels


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
    end_ix, end_iy = grid_index(x_m, y_m)
    line = bresenham_line_cells(0, 0, end_ix, end_iy)
    for i, cell in enumerate(line):
        if i == len(line) - 1:
            occupied_grid[cell] = occupied_grid.get(cell, 0) + 1
        elif occupied_grid.get(cell, 0) < OCCUPIED_MIN_HITS:
            free_grid[cell] = free_grid.get(cell, 0) + 1


def process_scan(scan_polar: List[Tuple[float, float]]) -> None:
    global latest_polar_points, latest_scan_points
    smoothed = smooth_scan_polar(scan_polar)
    xy_points = []
    ts = round(time.time(), 3)
    with data_lock:
        for a, d_cm in smoothed:
            x, y, d_m, a_deg = polar_to_xy(a, d_cm)
            xy_points.append((x, y, d_m, a_deg))
            carve_ray_to_obstacle(x, y)
            lidar_log_rows.append([ts, f"{a_deg:.2f}", f"{d_cm:.2f}", f"{x:.3f}", f"{y:.3f}", f"{d_m:.3f}"])
        latest_polar_points = smoothed
        latest_scan_points = xy_points


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
            safe = text.replace("'", "''")
            ps = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Rate = 0; "
                f"$s.Speak('{safe}')"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
            )
        except Exception as exc:
            print(f"Windows TTS error: {exc}")
        finally:
            with _tts_lock:
                _tts_busy = False

    threading.Thread(target=_worker, daemon=True).start()
    return True


def run_tts(text: str, force: bool = False) -> bool:
    """Speak text. force=True bypasses voice_enabled (for test buttons)."""
    global current_voice_process
    if not force and not voice_enabled:
        return False
    if not tts_checked:
        init_voice_system()

    if tts_backend == "espeak" and tts_executable:
        args = [
            tts_executable,
            "-s", str(ESPEAK_SPEED),
            "-a", str(ESPEAK_AMPLITUDE),
            "-g", str(ESPEAK_WORD_GAP_MS),
            text,
        ]
        try:
            current_voice_process = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return True
        except Exception:
            print("\a", end="", flush=True)
            return False

    if tts_backend == "windows":
        return _run_tts_windows(text)

    print(f"[VOICE] {text}")
    print("\a", end="", flush=True)
    return True


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
    global voice_enabled
    voice_enabled = not voice_enabled
    if not voice_enabled:
        stop_current_voice()
    print(f"Voice {'ON' if voice_enabled else 'OFF'}")
    run_tts(f"Voice {'on' if voice_enabled else 'off'}", force=True)


def print_voice_settings() -> None:
    print("Voice settings:")
    print(f"- Sign confirm detections: {SIGN_CONFIRM_DETECTIONS}")
    print(f"- Sign repeat: {SIGN_REPEAT_SECONDS:.0f} seconds")
    print(f"- LiDAR confirm scans: {LIDAR_CONFIRM_SCANS}")
    print(f"- Obstacle repeat: {OBSTACLE_REPEAT_SECONDS:.0f} seconds")
    print(f"- Very close repeat: {VERY_CLOSE_REPEAT_SECONDS:.0f} seconds")
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
    if "person" in low:
        return "Person"
    if "chair" in low:
        return "Chair"
    if "table" in low:
        return "Table"
    if "door" in low:
        return "Door"
    if any(k in low for k in ("backpack", "handbag", "bag")):
        return "Bag"
    if "bottle" in low:
        return "Bottle"
    if "sign" in low:
        return "Sign"
    if "obstacle" in low:
        return "Obstacle"
    return "Object"


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
    if not detections:
        return "", ""
    for priority in USEFUL_OBJECT_LABELS:
        matches = [d for d in detections if priority in d.label.lower()]
        if matches:
            best = max(matches, key=lambda d: d.confidence)
            return best.label, bbox_direction(best.bbox, frame_w)
    return "", ""


def update_sign_voice_state_machine(ocr_raw_text: str) -> None:
    global sign_candidate_text, sign_candidate_count, confirmed_sign_text, last_ocr_text
    cleaned = match_known_sign_text(ocr_raw_text) or clean_ocr_text(ocr_raw_text)
    if not cleaned or len(cleaned) < OCR_MIN_TEXT_LENGTH:
        return
    if len(cleaned) > OCR_MAX_TEXT_LENGTH and not is_known_sign_word(cleaned):
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
    global object_candidate_label, object_candidate_direction, object_candidate_count
    global confirmed_object_label, confirmed_object_direction
    label, direction = pick_best_camera_object(detections, frame_w)
    if not label:
        if object_candidate_label:
            object_candidate_label = ""
            object_candidate_direction = ""
            object_candidate_count = 0
        return
    if label == object_candidate_label and direction == object_candidate_direction:
        object_candidate_count += 1
    else:
        object_candidate_label = label
        object_candidate_direction = direction
        object_candidate_count = 1
    if object_candidate_count >= CAMERA_OBJECT_CONFIRM_DETECTIONS:
        confirmed_object_label = object_candidate_label
        confirmed_object_direction = object_candidate_direction


def update_lidar_voice_state_machine(raw_alert: str) -> None:
    global raw_lidar_alert, lidar_candidate_alert, lidar_candidate_count
    global confirmed_lidar_alert, lidar_clear_streak
    raw_lidar_alert = raw_alert
    if raw_alert == lidar_candidate_alert:
        lidar_candidate_count += 1
    else:
        lidar_candidate_alert = raw_alert
        lidar_candidate_count = 1
    if lidar_candidate_count >= LIDAR_CONFIRM_SCANS:
        confirmed_lidar_alert = lidar_candidate_alert
    elif raw_alert == "CLEAR":
        confirmed_lidar_alert = "CLEAR"
    lidar_clear_streak = lidar_clear_streak + 1 if raw_alert == "CLEAR" else 0


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
    words = text.title().split()
    return f"Sign says {' '.join(words)}"


def choose_voice_message(
    detections: List[Detection], frame_w: int
) -> Optional[Tuple[str, str, str, bool]]:
    """
    Return one voice message: (spoken_text, category_key, voice_track_key, interrupt).
    Priority: VERY_CLOSE > sign > STRONG > lidar normal > camera object > path clear.
    """
    now = time.time()

    if confirmed_lidar_alert.startswith("VERY_CLOSE_"):
        if (
            confirmed_lidar_alert != last_spoken_lidar_alert
            or (now - last_lidar_voice_time) >= VERY_CLOSE_REPEAT_SECONDS
        ):
            obj = matching_camera_object(confirmed_lidar_alert, detections, frame_w)
            msg = build_lidar_speech(confirmed_lidar_alert, obj)
            return msg, "lidar_very_close", confirmed_lidar_alert, True

    if confirmed_sign_text and not confirmed_lidar_alert.startswith("VERY_CLOSE_"):
        if (
            confirmed_sign_text != last_spoken_sign_text
            or (now - last_sign_voice_time) >= OCR_VOICE_REPEAT_SECONDS
        ):
            msg = build_sign_speech(confirmed_sign_text)
            return msg, "sign", confirmed_sign_text, False

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

    if confirmed_object_label and confirmed_lidar_alert == "CLEAR":
        alert_key = f"{confirmed_object_label}:{confirmed_object_direction}"
        if (
            alert_key != last_spoken_object_alert
            or (now - last_object_voice_time) >= CAMERA_OBJECT_REPEAT_SECONDS
        ):
            msg = build_object_speech(confirmed_object_label, confirmed_object_direction)
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

    if not voice_enabled:
        return False
    if interrupt or category == "lidar_very_close":
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
    if confirmed_lidar_alert != "CLEAR":
        return confirmed_lidar_alert
    if confirmed_sign_text:
        return f"SIGN:{confirmed_sign_text}"
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
        speak_chosen_message(spoken_text, category, track_key, interrupt)

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
    print("View: 0=quad | 1=camera | 2=zones | 3=LiDAR | 4=map | Z=zones overlay | F=fullscreen")
    print("Camera: C=retry | X=on/off | P=USB/PiCam priority | R=read sign now")
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
        f"vocabulary={len(KNOWN_SIGN_WORDS)} words"
    )
    return True


def get_centre_ocr_box(frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """Centre sign-reading region shown as yellow OCR SIGN AREA."""
    x1 = int(frame_w * 0.12)
    y1 = int(frame_h * 0.18)
    x2 = int(frame_w * 0.88)
    y2 = int(frame_h * 0.82)
    return x1, y1, x2, y2


def get_upper_ocr_box(frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """Upper sign band — signs are often mounted high."""
    return 0, 0, frame_w, max(60, int(frame_h * 0.5))


def is_known_sign_word(text: str) -> bool:
    return text in KNOWN_SIGN_WORDS


def _fix_ocr_chars_in_word(word: str) -> str:
    """Replace common OCR digit mistakes inside words."""
    if not word:
        return word
    out: List[str] = []
    upper = word.upper()
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
    """Clean OCR text and reject noisy random strings."""
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
        for k, v in corrections.items():
            if k in t and len(t) <= len(k) + 4:
                t = v
                break

    compact = t.replace(" ", "")
    if len(compact) < OCR_MIN_TEXT_LENGTH:
        return ""
    if len(compact) > OCR_MAX_TEXT_LENGTH and not is_known_sign_word(t):
        return ""

    if len(compact) >= 3 and len(set(compact)) == 1:
        return ""
    if len(compact) >= 4 and compact.count(compact[0]) / len(compact) > 0.85:
        return ""

    digit_count = sum(1 for c in compact if c.isdigit())
    if digit_count > 0 and digit_count / len(compact) > 0.35:
        return ""

    if not any(c in "AEIOU" for c in compact):
        if t not in VOWELLESS_OK and not any(t == w or t.startswith(w + " ") for w in VOWELLESS_OK):
            return ""

    if OCR_USE_SIGN_VOCABULARY and len(t) > OCR_MAX_TEXT_LENGTH and not is_known_sign_word(t):
        return ""

    return t


def match_known_sign_text(text: str) -> str:
    """Fuzzy-match OCR text to KNOWN_SIGN_WORDS. Returns best match or empty if rejected."""
    cleaned = clean_ocr_text(text)
    candidates: List[str] = []
    if cleaned:
        candidates.append(cleaned)
    rough = re.sub(r"[^A-Z0-9 ]+", " ", text.upper())
    rough = re.sub(r"\s+", " ", rough).strip()
    if rough and rough not in candidates:
        candidates.append(rough)
    for cand in candidates:
        if is_known_sign_word(cand):
            return cand
    best_word = ""
    best_ratio = 0.0
    for cand in candidates:
        for known in KNOWN_SIGN_WORDS:
            ratio = difflib.SequenceMatcher(None, cand, known).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_word = known
    if best_ratio >= OCR_FUZZY_MATCH_THRESHOLD:
        return best_word
    if OCR_USE_SIGN_VOCABULARY:
        return ""
    return cleaned or rough


def score_ocr_result(
    text: str,
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
) -> float:
    """Score OCR candidate — prefer known signs in centre/upper areas, not whole frame."""
    if not text:
        return 0.0
    fh, fw = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    box_area = bw * bh
    frame_area = max(1, fw * fh)
    area_ratio = box_area / frame_area

    score = 0.35
    if is_known_sign_word(text):
        score += 0.45
    elif OCR_USE_SIGN_VOCABULARY:
        score -= 0.25

    text_len = len(text.replace(" ", ""))
    if 3 <= text_len <= 12:
        score += 0.15
    elif text_len > 16:
        score -= 0.20

    digit_count = sum(1 for c in text if c.isdigit())
    if digit_count > 0:
        score -= 0.10 * digit_count

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


def _ocr_tesseract_configs() -> List[str]:
    whitelist = " -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return [f"--oem 3 --psm {psm}{whitelist}" for psm in (6, 7, 11, 13)]


def _ocr_single_roi(
    gray: np.ndarray,
    bbox: Tuple[int, int, int, int],
    results: List[OCRResult],
    frame_shape: Tuple[int, int],
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
    roi = cv2.convertScaleAbs(roi, alpha=1.45, beta=12)
    roi = cv2.bilateralFilter(roi, 5, 50, 50)
    variants = []
    _, otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.bitwise_not(otsu))
    variants.append(
        cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
    )
    seen: set = set()
    for img in variants:
        for cfg in _ocr_tesseract_configs():
            try:
                raw_txt = pytesseract.image_to_string(img, config=cfg)
            except Exception:
                continue
            raw_stripped = raw_txt.strip().upper()
            cleaned = clean_ocr_text(raw_txt)
            matched = match_known_sign_text(raw_txt)
            final_text = matched or cleaned
            if not final_text:
                continue
            if final_text in seen:
                continue
            seen.add(final_text)
            score = score_ocr_result(final_text, (x1, y1, x2, y2), frame_shape)
            if score < OCR_MIN_CONFIDENCE_SCORE:
                continue
            results.append(OCRResult(
                text=final_text,
                confidence=score,
                bbox=(x1, y1, x2, y2),
                timestamp=time.time(),
                raw_text=raw_stripped[:40],
                cleaned_text=cleaned,
                matched_text=matched or cleaned,
                score=score,
            ))
            ocr_log_rows.append([
                round(time.time(), 3), final_text, f"{score:.3f}", x1, y1, x2, y2,
            ])


def find_color_sign_rois(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Find likely sign regions by colour (red STOP signs, bright white boards)."""
    if cv2 is None:
        return []
    h, w = frame_bgr.shape[:2]
    upper = frame_bgr[0 : max(1, int(h * 0.55)), :]
    rois: List[Tuple[int, int, int, int]] = []
    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 80, 70), (12, 255, 255))
    red2 = cv2.inRange(hsv, (165, 80, 70), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    for mask in (red_mask, bright):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < 400 or bw < 20 or bh < 12:
                continue
            pad = 6
            rois.append((
                max(0, x - pad),
                max(0, y - pad),
                min(w, x + bw + pad),
                min(h, y + bh + pad),
            ))
    return rois[:4]


def run_ocr_on_signs(frame_bgr: np.ndarray, detections: List[Detection]) -> List[OCRResult]:
    if not ocr_enabled or pytesseract is None or cv2 is None:
        return []
    h, w = frame_bgr.shape[:2]
    frame_shape = (h, w)
    gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rois: List[Tuple[int, int, int, int]] = []

    sign_like = ("sign", "stop", "exit", "text", "poster")
    for d in detections:
        if any(s in d.label.lower() for s in sign_like):
            rois.append(d.bbox)

    rois.extend(find_color_sign_rois(frame_bgr))
    rois.append(get_centre_ocr_box(w, h))
    rois.append(get_upper_ocr_box(w, h))

    if OCR_WHOLE_FRAME_FALLBACK:
        rois.append((0, 0, w, h))

    results: List[OCRResult] = []
    seen_boxes: set = set()
    for box in rois:
        key = (box[0] // 20, box[1] // 20, box[2] // 20, box[3] // 20)
        if key in seen_boxes:
            continue
        seen_boxes.add(key)
        _ocr_single_roi(gray_full, box, results, frame_shape)

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
        ocr_debug_confirmed = vote_text
        confirmed_sign_text = vote_text
        sign_candidate_text = vote_text
        sign_candidate_count = SIGN_CONFIRM_DETECTIONS
        print(f"OCR CONFIRMED: {vote_text}")
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
    global dnn_net
    if cv2 is None:
        return []
    if dnn_net is None and os.path.isfile(AI_DNN_MODEL_PATH):
        try:
            dnn_net = cv2.dnn.readNet(AI_DNN_MODEL_PATH)
        except Exception:
            dnn_net = None
    if dnn_net is None:
        return []
    h, w = frame_bgr.shape[:2]
    try:
        blob = cv2.dnn.blobFromImage(frame_bgr, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        dnn_net.setInput(blob)
        outs = dnn_net.forward()
    except Exception:
        return []
    dets = []
    out = outs[0] if isinstance(outs, (list, tuple)) and len(outs) else outs
    if out is None or not hasattr(out, "shape"):
        return []
    if len(out.shape) == 3:
        out = out[0]
    for row in out:
        if len(row) < 6:
            continue
        conf = float(row[4])
        if conf < AI_CONFIDENCE_THRESHOLD:
            continue
        scores = row[5:]
        class_id = int(np.argmax(scores)) if len(scores) else 0
        cls_conf = float(scores[class_id]) if len(scores) else conf
        final_conf = conf * cls_conf
        if final_conf < AI_CONFIDENCE_THRESHOLD:
            continue
        cx, cy, bw, bh = row[0:4]
        x1 = int((cx - bw / 2) * w / 640)
        y1 = int((cy - bh / 2) * h / 640)
        x2 = int((cx + bw / 2) * w / 640)
        y2 = int((cy + bh / 2) * h / 640)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        label = ai_labels[class_id] if class_id < len(ai_labels) else f"class_{class_id}"
        dets.append(Detection(label, final_conf, (x1, y1, x2, y2), approximate_distance_from_bbox(w, (x1, y1, x2, y2)), "opencv_dnn", time.time()))
    return dets


def detect_with_hog(frame_bgr: np.ndarray) -> List[Detection]:
    global hog_detector
    if cv2 is None:
        return []
    if hog_detector is None:
        hog_detector = cv2.HOGDescriptor()
        hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog_detector.detectMultiScale(frame_bgr, winStride=(8, 8), padding=(8, 8), scale=1.05)
    h, w = frame_bgr.shape[:2]
    dets = []
    for (x, y, rw, rh), wt in zip(rects, weights):
        conf = float(wt)
        if conf < 0.2:
            continue
        bbox = (int(x), int(y), int(x + rw), int(y + rh))
        dets.append(Detection("person", min(0.9, 0.4 + conf / 2.0), bbox, approximate_distance_from_bbox(w, bbox), "hog", time.time()))
    return dets


def detect_with_contours(frame_bgr: np.ndarray) -> List[Detection]:
    if cv2 is None:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 60, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame_bgr.shape[:2]
    dets = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1400:
            continue
        x, y, rw, rh = cv2.boundingRect(c)
        if rw < 18 or rh < 18:
            continue
        bbox = (x, y, x + rw, y + rh)
        dets.append(Detection("obstacle", 0.35, bbox, approximate_distance_from_bbox(w, bbox), "contour", time.time()))
    return dets[:5]


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


def run_camera_detections(frame_bgr: np.ndarray, frame_n: int) -> List[Detection]:
    """AI detection on downscaled frame — Hailo first, then OpenCV fallback."""
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
    if frame_n % max(1, DNN_EVERY_N_FRAMES) == 0:
        dets = detect_with_opencv_dnn(small)
        if dets:
            latest_detection_source = "opencv_dnn"
            _last_ai_fps_count += 1
            return scale_detections_to_frame(dets, AI_PROCESS_WIDTH, AI_PROCESS_HEIGHT, w, h)
    dets = detect_with_hog(small)
    if dets:
        latest_detection_source = "hog"
        _last_ai_fps_count += 1
        return scale_detections_to_frame(dets, AI_PROCESS_WIDTH, AI_PROCESS_HEIGHT, w, h)
    if frame_n % 12 == 0:
        dets = detect_with_contours(small)
        if dets:
            latest_detection_source = "contour"
            _last_ai_fps_count += 1
            return scale_detections_to_frame(dets, AI_PROCESS_WIDTH, AI_PROCESS_HEIGHT, w, h)
    return []


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

    camera_error_message = "No working USB camera found after scanning indexes 0-5"
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
        out, "Hold sign inside yellow box for 5-10 seconds",
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
        3: "LiDAR Distance View (full)",
        4: "Local Room Map (full)",
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
            "0=quad  2=zones  3=LiDAR  4=map",
            "C=retry  X=cam  P=USB/PiCam  R=read sign now",
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


def draw_lidar_distance_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    latest_scan_snapshot: List[Tuple[float, float, float, float]],
) -> None:
    """LiDAR distance heatmap using a snapshot of the latest scan."""
    pygame.draw.circle(screen, (40, 46, 56), rect.center, min(rect.width, rect.height) // 2 - 4)
    for r in (0.5, 1.0, 1.5, 2.0):
        pygame.draw.circle(screen, (58, 68, 80), rect.center, int(r * pixels_per_meter), 1)
    for x_m, y_m, d_m, _a in latest_scan_snapshot:
        px, py = world_to_panel(x_m, y_m, rect)
        if not rect.collidepoint(px, py):
            continue
        col = COLOR_RED if d_m <= VERY_CLOSE_DISTANCE_M else COLOR_ORANGE if d_m <= STRONG_WARNING_DISTANCE_M else COLOR_YELLOW if d_m <= ALERT_DISTANCE_M else COLOR_CYAN
        screen.set_at((px, py), col)
    pygame.draw.circle(screen, COLOR_GREEN, rect.center, 5)


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


def draw_local_room_map(
    screen: pygame.Surface,
    rect: pygame.Rect,
    free_grid_snapshot: Dict[Tuple[int, int], int],
    occupied_grid_snapshot: Dict[Tuple[int, int], int],
    latest_scan_snapshot: List[Tuple[float, float, float, float]],
) -> None:
    """Local map using snapshots — safe while LiDAR thread updates grids."""
    pygame.draw.rect(screen, (15, 20, 26), rect)
    for r in (1, 2, 3, 4, 5):
        pygame.draw.circle(screen, (35, 90, 150), rect.center, int(r * pixels_per_meter), 1)
    # Free cells
    for (ix, iy), fh in free_grid_snapshot.items():
        if fh < FREE_MIN_HITS or occupied_grid_snapshot.get((ix, iy), 0) >= OCCUPIED_MIN_HITS:
            continue
        px, py = world_to_panel(ix * GRID_RESOLUTION_M, iy * GRID_RESOLUTION_M, rect)
        if rect.collidepoint(px, py):
            pygame.draw.rect(screen, (28, 48, 82), pygame.Rect(px - 1, py - 1, 3, 3))
    # Weak occupied
    for (ix, iy), hits in occupied_grid_snapshot.items():
        if OCCUPIED_MIN_HITS <= hits < WALL_STRONG_HITS:
            px, py = world_to_panel(ix * GRID_RESOLUTION_M, iy * GRID_RESOLUTION_M, rect)
            if rect.collidepoint(px, py):
                pygame.draw.rect(screen, (40, 160, 200), pygame.Rect(px - 1, py - 1, 3, 3))
    # Connected wall components
    cells = [c for c, hits in occupied_grid_snapshot.items() if hits >= OCCUPIED_MIN_HITS]
    comps = get_connected_wall_components(cells)
    for comp in comps:
        for ix, iy in comp:
            hits = occupied_grid_snapshot.get((ix, iy), 0)
            col = (140, 240, 180) if hits >= WALL_STRONG_HITS else (90, 220, 255)
            px, py = world_to_panel(ix * GRID_RESOLUTION_M, iy * GRID_RESOLUTION_M, rect)
            if rect.collidepoint(px, py):
                size = 4 if hits >= WALL_STRONG_HITS else 3
                pygame.draw.rect(screen, col, pygame.Rect(px - size // 2, py - size // 2, size, size))
    # Current scan
    for x_m, y_m, d_m, _a in latest_scan_snapshot:
        px, py = world_to_panel(x_m, y_m, rect)
        if rect.collidepoint(px, py):
            col = COLOR_RED if d_m <= VERY_CLOSE_DISTANCE_M else COLOR_YELLOW if d_m <= ALERT_DISTANCE_M else COLOR_CYAN
            pygame.draw.circle(screen, col, (px, py), 2)
    pygame.draw.circle(screen, COLOR_GREEN, rect.center, 4)
    pygame.draw.polygon(screen, COLOR_GREEN, [
        (rect.centerx, rect.centery - 10), (rect.centerx - 4, rect.centery + 2), (rect.centerx + 4, rect.centery + 2),
    ])


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
    elif action == "voice_sign":
        test_voice_sign()
    elif action == "voice_obstacle":
        test_voice_obstacle()
    elif action == "voice_stop":
        test_voice_stop()


def draw_voice_ui_buttons(screen: pygame.Surface) -> None:
    global ui_button_rects
    labels = [
        ("Voice Test", "voice_test"),
        ("Voice On/Off", "voice_toggle"),
        ("Test Sign", "voice_sign"),
        ("Test Alert", "voice_obstacle"),
        ("Test STOP", "voice_stop"),
    ]
    ui_button_rects = []
    btn_w, btn_h, gap = 118, 30, 6
    total_w = len(labels) * btn_w + (len(labels) - 1) * gap
    x0 = max(8, (SCREEN_WIDTH - total_w) // 2)
    y0 = SCREEN_HEIGHT - FOOTER_HEIGHT + 6
    font = pygame.font.SysFont("monospace", 12, bold=True)
    for i, (label, action) in enumerate(labels):
        rect = pygame.Rect(x0 + i * (btn_w + gap), y0, btn_w, btn_h)
        bg = (35, 95, 140) if voice_enabled or action != "voice_toggle" else (80, 45, 45)
        pygame.draw.rect(screen, bg, rect)
        pygame.draw.rect(screen, (120, 200, 255), rect, 2)
        txt = font.render(label, True, (235, 245, 255))
        screen.blit(txt, txt.get_rect(center=rect.center))
        ui_button_rects.append((rect, action))


def draw_footer(screen: pygame.Surface) -> None:
    pygame.draw.rect(screen, (12, 18, 26), pygame.Rect(0, SCREEN_HEIGHT - FOOTER_HEIGHT, SCREEN_WIDTH, FOOTER_HEIGHT))
    draw_voice_ui_buttons(screen)
    controls = (
        "0 Quad | 1 Camera | 2 Zones | 3 LiDAR | 4 Map | "
        "C Retry Camera | X Cam On/Off | P USB/PiCam | R Read Sign Now | "
        "O OCR | I AI Overlay | L LiDAR | V Voice | D Debug | S Save | Q Quit"
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
        f"tts_backend={tts_backend} voice_enabled={voice_enabled} speaking={is_voice_speaking()}",
        f"seconds_since_last_voice={since_voice:.1f}",
        "--- OCR sign (v7) ---",
        f"ocr_raw={ocr_debug_raw or '-'}",
        f"ocr_cleaned={ocr_debug_cleaned or '-'}",
        f"ocr_matched={ocr_debug_matched or '-'}",
        f"ocr_confirmed={ocr_debug_confirmed or confirmed_sign_text or '-'}",
        f"ocr_votes={ocr_vote_counts}",
        f"ocr_candidates={', '.join(ocr_last_candidates[:6]) or '-'}",
        f"OCR_USE_SIGN_VOCABULARY={OCR_USE_SIGN_VOCABULARY} "
        f"stable={OCR_REQUIRE_STABLE_READS}/{OCR_STABLE_WINDOW_SECONDS:.0f}s",
        "--- Sign / OCR voice ---",
        f"sign_candidate={sign_candidate_text or '-'} ({sign_candidate_count}/{SIGN_CONFIRM_DETECTIONS})",
        f"confirmed_sign={confirmed_sign_text or '-'} last_spoken_sign={last_spoken_sign_text or '-'}",
        "--- Camera object voice ---",
        f"object_candidate={object_candidate_label or '-'} {object_candidate_direction or ''} "
        f"({object_candidate_count}/{CAMERA_OBJECT_CONFIRM_DETECTIONS})",
        f"confirmed_object={confirmed_object_label or '-'} {confirmed_object_direction or ''}",
        f"last_spoken_object={last_spoken_object_alert or '-'}",
        "--- LiDAR obstacle voice ---",
        f"lidar_raw={raw_lidar_alert} candidate={lidar_candidate_alert} "
        f"({lidar_candidate_count}/{LIDAR_CONFIRM_SCANS})",
        f"confirmed_lidar={confirmed_lidar_alert} last_spoken_lidar={last_spoken_lidar_alert or '-'}",
        f"clear_streak={lidar_clear_streak}/{CLEAR_CONFIRM_SCANS} danger_was={last_spoken_was_danger}",
        f"lidar_pkts={scan_points_count}",
        f"zones F/L/R/B={last_zone_counts['front']}/{last_zone_counts['left']}/"
        f"{last_zone_counts['right']}/{last_zone_counts['back']}",
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
    ai_labels = load_labels(AI_LABELS_PATH)
    print_voice_settings()
    print_camera_settings()
    init_voice_system()
    init_ocr_system()
    init_ai_hat()

    pygame.init()
    pygame.display.set_caption("Team Bravo Vision Assistant v7 better sign OCR")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    print("Pi 5 v7: sign vocabulary OCR | R=read sign now | C=retry | P=USB/PiCam | D=debug")

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

                r = draw_panel_frame(screen, p01, "LiDAR Distance View")
                draw_lidar_distance_panel(screen, r, latest_scan_snapshot)

                r = draw_panel_frame(screen, p11, "Local Room Map")
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
                    r = draw_panel_frame(screen, full, "LiDAR Distance View")
                    draw_lidar_distance_panel(screen, r, latest_scan_snapshot)
                elif focused_panel == 4:
                    r = draw_panel_frame(screen, full, "Local Room Map")
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
