#!/usr/bin/env python3
"""
Team Bravo AI HAT Camera + LiDAR Vision Assistant for Blind Navigation Support
================================================================================

Real-time assistive navigation dashboard for Raspberry Pi 5.

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
    python3 team_bravo_aihat_camera_lidar_vision_assistant.py

Safety:
    Prototype assistive navigation aid only — NOT the sole safety device for a blind
    person. LiDAR may miss glass, shiny surfaces, low objects, soft materials.
    Camera AI may misclassify. OCR may misread signs. AI HAT supports but does not
    replace LiDAR distance safety. Test with human supervision.
"""

from __future__ import annotations

import csv
import math
import os
import random
import re
import shutil
import struct
import subprocess
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

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_DISPLAY_WIDTH = 320
CAMERA_DISPLAY_HEIGHT = 240

AI_MODEL_PATH = "models/yolov8n.hef"
AI_LABELS_PATH = "models/coco_labels.txt"
AI_DNN_MODEL_PATH = "models/yolov8n.onnx"
AI_CONFIDENCE_THRESHOLD = 0.45
AI_NMS_THRESHOLD = 0.40
AI_INFERENCE_EVERY_N_FRAMES = 1
OCR_EVERY_N_FRAMES = 5
OCR_MIN_TEXT_LENGTH = 2
OCR_REPEAT_SECONDS = 15.0
CAUTION_DISTANCE_M = 1.2
ALERT_DISTANCE_M = 1.0
STRONG_WARNING_DISTANCE_M = 0.75
VERY_CLOSE_DISTANCE_M = 0.40

VOICE_COOLDOWN_SECONDS = 2.0
VOICE_REPEAT_SECONDS = 6.0
VOICE_MIN_DETECTIONS = 8
ZONE_CLEAR_SCANS = 12
CLEAR_VOICE_MIN_GAP = 4.0
CAMERA_VOICE_MIN_DETECTIONS = 4
CAMERA_VOICE_COOLDOWN_SECONDS = 3.0
CAMERA_VOICE_REPEAT_SECONDS = 10.0
ESPEAK_SPEED = 155
ESPEAK_AMPLITUDE = 180
ESPEAK_WORD_GAP_MS = 6

# Additional tuned constants.
ZONE_MIN_POINTS = 3

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
HEADER_HEIGHT = 54
FOOTER_HEIGHT = 40
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


running = True
simulation_paused = False
zones_fullscreen = False
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
camera_source = "SIM"
camera_available = False
latest_camera_detections: List[Detection] = []
latest_ocr_results: List[OCRResult] = []
last_ocr_text = ""
last_ocr_spoken_at = 0.0
last_camera_banner = "No detections"

raw_lidar_alert = "CLEAR"
candidate_lidar_alert = "CLEAR"
candidate_lidar_count = 0
confirmed_lidar_alert = "CLEAR"
lidar_clear_streak = 0

raw_camera_alert = "CLEAR"
candidate_camera_alert = "CLEAR"
candidate_camera_count = 0
confirmed_camera_alert = "CLEAR"

last_spoken_alert = ""
last_voice_time = 0.0
tts_checked = False
tts_executable: Optional[str] = None
current_voice_process: Optional[subprocess.Popen] = None

camera_frame_counter = 0
sim_phase = 0.0

lidar_log_rows: List[List[object]] = []
camera_log_rows: List[List[object]] = []
ocr_log_rows: List[List[object]] = []

AI_HAT_RUNTIME_AVAILABLE = HAILO_AVAILABLE
ai_hat_active = False
ai_hat_device = None
ai_inference_fps = 0.0
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


def check_tts() -> bool:
    global tts_checked, tts_executable
    if not tts_checked:
        tts_executable = find_tts_executable()
        tts_checked = True
    return tts_executable is not None


def stop_current_voice() -> None:
    global current_voice_process
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


def run_tts(text: str) -> bool:
    global current_voice_process
    if not check_tts():
        print("\a", end="", flush=True)
        return False
    args = [tts_executable, "-s", "165", "-a", "180", "-g", "2", text]
    try:
        current_voice_process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        print("\a", end="", flush=True)
        return False


def lidar_to_speech(alert_state: str) -> str:
    mapping = {
        "CLEAR": "Path clear",
        "NORMAL_FRONT": "Obstacle ahead",
        "STRONG_FRONT": "Careful, obstacle close ahead",
        "VERY_CLOSE_FRONT": "Stop, obstacle very close ahead",
        "NORMAL_LEFT": "Obstacle on your left",
        "STRONG_LEFT": "Careful, obstacle close on your left",
        "VERY_CLOSE_LEFT": "Stop, obstacle very close on your left",
        "NORMAL_RIGHT": "Obstacle on your right",
        "STRONG_RIGHT": "Careful, obstacle close on your right",
        "VERY_CLOSE_RIGHT": "Stop, obstacle very close on your right",
        "BACK": "Obstacle behind you",
        "VERY_CLOSE_BACK": "Stop, obstacle very close behind",
    }
    return mapping.get(alert_state, "Obstacle nearby")


def camera_to_speech(camera_alert: str, ocr_text: str = "") -> str:
    if camera_alert.startswith("OCR_SIGN_"):
        return f"Sign says {camera_alert.replace('OCR_SIGN_', '').replace('_', ' ').title()}"
    if camera_alert.startswith("CAMERA_"):
        label = camera_alert.replace("CAMERA_", "").replace("_", " ").lower()
        return f"{label} detected ahead"
    if camera_alert.startswith("FUSED_"):
        msg = camera_alert.replace("FUSED_", "").replace("_", " ").lower()
        return f"Careful {msg}"
    if ocr_text:
        return f"Sign says {ocr_text.title()}"
    return ""


def update_lidar_voice_state_machine(raw_alert: str) -> None:
    global raw_lidar_alert, candidate_lidar_alert, candidate_lidar_count, confirmed_lidar_alert, lidar_clear_streak
    raw_lidar_alert = raw_alert
    if raw_alert == candidate_lidar_alert:
        candidate_lidar_count += 1
    else:
        candidate_lidar_alert = raw_alert
        candidate_lidar_count = 1
    if candidate_lidar_count >= VOICE_MIN_DETECTIONS:
        confirmed_lidar_alert = candidate_lidar_alert
    elif raw_alert == "CLEAR":
        confirmed_lidar_alert = "CLEAR"
    lidar_clear_streak = lidar_clear_streak + 1 if raw_alert == "CLEAR" else 0


def update_camera_voice_state_machine(raw_alert: str, _banner: str) -> None:
    global raw_camera_alert, candidate_camera_alert, candidate_camera_count, confirmed_camera_alert
    raw_camera_alert = raw_alert
    if raw_alert == candidate_camera_alert:
        candidate_camera_count += 1
    else:
        candidate_camera_alert = raw_alert
        candidate_camera_count = 1
    if candidate_camera_count >= CAMERA_VOICE_MIN_DETECTIONS:
        confirmed_camera_alert = candidate_camera_alert
    elif raw_alert == "CLEAR":
        confirmed_camera_alert = "CLEAR"


def clean_ocr_text(text: str) -> str:
    t = text.strip().upper()
    t = re.sub(r"[^A-Z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def run_ocr_on_signs(frame_bgr: np.ndarray, detections: List[Detection]) -> List[OCRResult]:
    if not ocr_enabled or pytesseract is None:
        return []
    h, w = frame_bgr.shape[:2]
    rois = []
    sign_like = ("sign", "stop", "exit", "text", "poster")
    for d in detections:
        if any(s in d.label.lower() for s in sign_like):
            rois.append(d.bbox)
    if not rois:
        rois.append((0, 0, w, h))
    results: List[OCRResult] = []
    for (x1, y1, x2, y2) in rois[:2]:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        roi = frame_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if cv2 is not None else roi
        if cv2 is not None:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        txt = pytesseract.image_to_string(gray, config="--oem 3 --psm 6")
        cleaned = clean_ocr_text(txt)
        if len(cleaned) >= OCR_MIN_TEXT_LENGTH:
            results.append(OCRResult(cleaned, 0.7, (x1, y1, x2, y2), time.time()))
            ocr_log_rows.append([round(time.time(), 3), cleaned, "0.70", x1, y1, x2, y2])
    return results


def approximate_distance_from_bbox(frame_w: int, bbox: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    rel = box_w / max(1, frame_w)
    return max(0.3, min(5.0, 1.9 / (rel + 1e-3)))


def try_hailo_inference_placeholder(frame_bgr: np.ndarray) -> Optional[List[Detection]]:
    """Internal Hailo inference hook — returns None until SDK code is inserted."""
    return run_ai_hat_inference(frame_bgr)


def init_ai_hat() -> bool:
    """
    Initialise Raspberry Pi AI HAT / Hailo runtime.
    Insert Hailo SDK setup here when available.
    """
    global ai_hat_active, ai_hat_device, AI_HAT_RUNTIME_AVAILABLE
    if not ENABLE_AI_HAT:
        print("AI HAT disabled in settings.")
        return False
    if not HAILO_AVAILABLE:
        print("AI HAT not available — using OpenCV fallback.")
        AI_HAT_RUNTIME_AVAILABLE = False
        return False
    try:
        # PLACEHOLDER — Raspberry Pi AI Kit / Hailo integration:
        # from hailo_platform import HEF, VDevice, HailoStreamInterface
        # ai_hat_device = VDevice()
        # load_ai_model()
        # ai_hat_active = True
        AI_HAT_RUNTIME_AVAILABLE = True
        ai_hat_active = load_ai_model()
        if ai_hat_active:
            print("AI HAT: active")
        else:
            print("AI HAT runtime found but model not loaded — using OpenCV fallback.")
        return ai_hat_active
    except Exception as exc:
        print(f"AI HAT init failed: {exc}")
        AI_HAT_RUNTIME_AVAILABLE = False
        ai_hat_active = False
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
    """Run AI HAT inference on BGR frame. Returns detections or None to fallback."""
    global _last_ai_fps_count
    if not ai_hat_active or not ENABLE_AI_HAT:
        return None
    try:
        # PLACEHOLDER:
        # preprocessed = preprocess_for_hailo(frame_bgr)
        # raw = hailo_network.run(preprocessed)
        # dets = parse_ai_hat_results(raw)
        # _last_ai_fps_count += 1
        # return dets
        return None
    except Exception as exc:
        print(f"AI HAT inference error: {exc}")
        return None


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


def camera_thread_fn() -> None:
    global camera_available, camera_source, latest_camera_rgb, latest_camera_detections, latest_ocr_results
    global last_camera_banner, camera_frame_counter, last_ocr_text, last_ocr_spoken_at
    global camera_fps, ai_inference_fps, _last_camera_fps_count, _last_camera_fps_time
    global _last_ai_fps_count, _last_ai_fps_time

    picam = None
    cap = None
    if ENABLE_CAMERA and not SIMULATED_MODE:
        if Picamera2 is not None:
            try:
                picam = Picamera2()
                cfg = picam.create_video_configuration(main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"})
                picam.configure(cfg)
                picam.start()
                camera_source = "PiCam"
                camera_available = True
            except Exception:
                picam = None
        if picam is None and cv2 is not None:
            cap = cv2.VideoCapture(0)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                camera_source = "USB"
                camera_available = True
    if not camera_available:
        camera_source = "SIM"

    while running:
        if not camera_enabled:
            time.sleep(0.05)
            continue

        frame_bgr = None
        if camera_source == "PiCam" and picam is not None:
            try:
                arr = picam.capture_array()
                frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if cv2 is not None else arr
            except Exception:
                frame_bgr = None
        elif camera_source == "USB" and cap is not None:
            ok, frame = cap.read()
            frame_bgr = frame if ok else None

        if frame_bgr is None:
            frame_bgr, dets, ocr_items = simulated_camera_frame_and_detections()
            cam_alert, banner = choose_camera_alert(dets, ocr_items)
        else:
            dets: List[Detection] = []
            if camera_frame_counter % max(1, AI_INFERENCE_EVERY_N_FRAMES) == 0:
                ai_out = run_ai_hat_inference(frame_bgr)
                if ai_out:
                    dets = ai_out
                    _last_ai_fps_count += 1
                else:
                    dets = detect_with_opencv_dnn(frame_bgr)
                    if not dets:
                        dets = detect_with_hog(frame_bgr)
                    if not dets:
                        dets = detect_with_contours(frame_bgr)
            ocr_items = []
            if camera_frame_counter % max(1, OCR_EVERY_N_FRAMES) == 0:
                ocr_items = run_ocr_on_signs(frame_bgr, dets)
            cam_alert, banner = choose_camera_alert(dets, ocr_items)

        for d in dets:
            camera_log_rows.append([
                round(d.timestamp, 3), d.label, f"{d.confidence:.3f}",
                d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
                "" if d.distance_m is None else f"{d.distance_m:.3f}", d.source
            ])

        update_camera_voice_state_machine(cam_alert, banner)
        last_camera_banner = banner

        # OCR repeat gate for spoken sign updates
        for item in ocr_items:
            if item.text != last_ocr_text or (time.time() - last_ocr_spoken_at) >= OCR_REPEAT_SECONDS:
                last_ocr_text = item.text
                last_ocr_spoken_at = time.time()
                break

        if cv2 is not None and ai_overlay_enabled:
            frame_bgr = draw_ai_detections(frame_bgr, dets)
            for o in ocr_items:
                x1, y1, x2, y2 = o.bbox
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 220, 40), 2)
                cv2.putText(frame_bgr, f"Text: {o.text}", (x1, min(frame_bgr.shape[0] - 10, y2 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 40), 2, cv2.LINE_AA)

        _last_camera_fps_count += 1
        now_fps = time.time()
        if now_fps - _last_camera_fps_time >= 1.0:
            camera_fps = _last_camera_fps_count / (now_fps - _last_camera_fps_time)
            ai_inference_fps = _last_ai_fps_count / (now_fps - _last_ai_fps_time)
            _last_camera_fps_count = 0
            _last_ai_fps_count = 0
            _last_camera_fps_time = now_fps
            _last_ai_fps_time = now_fps

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if cv2 is not None else frame_bgr
        with camera_lock:
            latest_camera_rgb = rgb
            latest_camera_detections = dets
            latest_ocr_results = ocr_items
        camera_frame_counter += 1
        time.sleep(0.01)

    if cap is not None:
        cap.release()
    if picam is not None:
        try:
            picam.stop()
        except Exception:
            pass


def fuse_camera_lidar_alerts(lidar_alert: str, camera_alert: str, ocr_items: List[OCRResult], detections: List[Detection]) -> str:
    # Priority 1: LiDAR VERY_CLOSE emergency
    if lidar_alert.startswith("VERY_CLOSE_"):
        return lidar_alert

    # Priority 2: LiDAR STRONG + AI label
    if lidar_alert.startswith("STRONG_") and camera_alert.startswith("CAMERA_"):
        return f"FUSED_{lidar_alert}_{camera_alert.replace('CAMERA_', '')}"

    # Priority 3: OCR
    if ocr_items:
        txt = ocr_items[0].text
        if txt:
            return f"OCR_SIGN_{txt.replace(' ', '_')}"

    # Priority 4: AI object
    if camera_alert.startswith("CAMERA_"):
        return camera_alert

    # Priority 5: normal LiDAR
    if lidar_alert != "CLEAR":
        return lidar_alert

    # Priority 6: clear
    return "CLEAR"


def speak_fused_alert(alert_state: str, ocr_text: str = "") -> bool:
    global last_spoken_alert, last_voice_time
    if not voice_enabled:
        return False
    now = time.time()
    if now - last_voice_time < VOICE_COOLDOWN_SECONDS and alert_state == last_spoken_alert:
        return False

    if alert_state.startswith("VERY_CLOSE_"):
        # Emergency STOP interrupt.
        stop_current_voice()

    text = camera_to_speech(alert_state, ocr_text)
    if not text:
        text = lidar_to_speech(alert_state)
    ok = run_tts(text)
    if ok:
        last_spoken_alert = alert_state
        last_voice_time = now
    return ok


def process_voice_alerts() -> str:
    with data_lock:
        pts = list(latest_scan_points)
    with camera_lock:
        dets = list(latest_camera_detections)
        ocr_items = list(latest_ocr_results)

    lidar_alert, nearest, zc = detect_obstacles_for_blind_user(pts)
    update_lidar_voice_state_machine(lidar_alert)
    cam_alert = confirmed_camera_alert
    fused = fuse_camera_lidar_alerts(confirmed_lidar_alert, cam_alert, ocr_items, dets) if fusion_enabled else confirmed_lidar_alert

    last_zone_counts["front"] = zc["front"]
    last_zone_counts["left"] = zc["left"]
    last_zone_counts["right"] = zc["right"]
    last_zone_counts["back"] = zc["back"]
    direction_distances["front"] = _zone_nearest(pts, lambda x, y, d: x > 0 and abs(y) <= 0.45 and d <= CAUTION_DISTANCE_M)
    direction_distances["left"] = _zone_nearest(pts, lambda x, y, d: y < -0.35 and d <= CAUTION_DISTANCE_M)
    direction_distances["right"] = _zone_nearest(pts, lambda x, y, d: y > 0.35 and d <= CAUTION_DISTANCE_M)
    direction_distances["back"] = _zone_nearest(pts, lambda x, y, d: x < 0 and abs(y) <= 0.45 and d <= 0.8)

    now = time.time()
    elapsed = now - last_voice_time
    if fused.startswith("VERY_CLOSE_"):
        if fused != last_spoken_alert or elapsed >= VOICE_REPEAT_SECONDS:
            speak_fused_alert(fused)
    elif fused != "CLEAR":
        if fused != last_spoken_alert:
            speak_fused_alert(fused)
        elif elapsed >= VOICE_REPEAT_SECONDS:
            speak_fused_alert(fused)
    else:
        if last_spoken_alert and last_spoken_alert != "CLEAR" and lidar_clear_streak >= VOICE_MIN_DETECTIONS and elapsed >= CLEAR_VOICE_MIN_GAP:
            speak_fused_alert("CLEAR")
    return fused


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


def draw_panel_frame(screen: pygame.Surface, rect: pygame.Rect, title: str) -> pygame.Rect:
    pygame.draw.rect(screen, COLOR_PANEL, rect)
    pygame.draw.rect(screen, COLOR_PANEL_BORDER, rect, 2)
    title_surf = pygame.font.SysFont("monospace", 14, bold=True).render(title, True, COLOR_TITLE)
    screen.blit(title_surf, (rect.x + 8, rect.y + 6))
    return pygame.Rect(rect.x + 4, rect.y + 24, rect.width - 8, rect.height - 28)


def draw_ai_camera_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    frame_snapshot: Optional[np.ndarray],
    detections_snapshot: List[Detection],
    ocr_snapshot: List[OCRResult],
) -> None:
    """Draw AI camera panel using pre-copied snapshots (no lock during draw)."""
    if frame_snapshot is not None:
        surf = pygame.surfarray.make_surface(frame_snapshot.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (rect.width, rect.height))
        screen.blit(surf, rect.topleft)
    else:
        pygame.draw.rect(screen, (30, 35, 45), rect)
        txt = pygame.font.SysFont("monospace", 14).render("No camera frame (using simulation fallback)", True, COLOR_MUTED)
        screen.blit(txt, (rect.x + 10, rect.y + 20))
    font = pygame.font.SysFont("monospace", 12)
    y = rect.y + 4
    ai_status = "ACTIVE" if ai_hat_active else ("FALLBACK" if ENABLE_AI_HAT else "OFF")
    screen.blit(font.render(f"AI HAT: {ai_status}  Cam: {camera_fps:.0f} fps  AI: {ai_inference_fps:.0f} fps", True, COLOR_CYAN), (rect.x + 6, y))
    y += 16
    screen.blit(font.render(f"OCR: {'ON' if ocr_enabled and pytesseract else 'OFF'}  Source: {camera_source}", True, COLOR_TEXT), (rect.x + 6, y))
    y += 16
    screen.blit(font.render(last_camera_banner[:48], True, COLOR_TEXT), (rect.x + 6, y))
    y += 16
    if ocr_snapshot:
        screen.blit(font.render(f"Sign: {ocr_snapshot[0].text}", True, COLOR_YELLOW), (rect.x + 6, y))
        y += 16
    for d in detections_snapshot[:3]:
        dist = f" {d.distance_m:.1f}m" if d.distance_m else ""
        screen.blit(font.render(f"{d.label}{dist} {int(d.confidence*100)}%", True, COLOR_GREEN), (rect.x + 6, y))
        y += 14


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
    cam_state = "SIM" if camera_source == "SIM" else ("PiCam" if camera_source == "PiCam" else "USB")
    if ENABLE_AI_HAT and ai_hat_active:
        ai_state = "ACTIVE"
    elif ENABLE_AI_HAT:
        ai_state = "FALLBACK"
    else:
        ai_state = "OFF"
    text = (
        f"LiDAR {lidar_state} | Camera {cam_state} | "
        f"AI HAT {ai_state} | OCR {'ON' if ocr_enabled else 'OFF'} | "
        f"Voice {'ON' if voice_enabled else 'OFF'} | Alert {fused_alert}"
    )
    screen.blit(font.render(text, True, COLOR_TEXT), (10, 16))


def draw_footer(screen: pygame.Surface) -> None:
    pygame.draw.rect(screen, (12, 18, 26), pygame.Rect(0, SCREEN_HEIGHT - FOOTER_HEIGHT, SCREEN_WIDTH, FOOTER_HEIGHT))
    controls = "Q Quit | C Camera | S Save | V Voice | T Test | O OCR | I Overlay | L LiDAR | Z Zones | D Debug | F Fusion | Space Pause | +/- Zoom"
    font = pygame.font.SysFont("monospace", 13)
    screen.blit(font.render(controls, True, COLOR_MUTED), (10, SCREEN_HEIGHT - 26))


def draw_debug_panel(
    screen: pygame.Surface,
    fused_alert: str,
    detections_count: int,
    scan_points_count: int,
) -> None:
    if not debug_enabled:
        return
    rect = pygame.Rect(12, HEADER_HEIGHT + 10, 390, 190)
    overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 170))
    screen.blit(overlay, rect.topleft)
    pygame.draw.rect(screen, (100, 170, 220), rect, 2)
    font = pygame.font.SysFont("monospace", 13)
    lines = [
        f"AI_HAT_RUNTIME={AI_HAT_RUNTIME_AVAILABLE} AI_HAT_ACTIVE={ai_hat_active}",
        f"model={AI_MODEL_PATH}",
        f"camera_fps={camera_fps:.1f} ai_fps={ai_inference_fps:.1f}",
        f"ocr_available={pytesseract is not None} ocr_text={last_ocr_text or '-'}",
        f"detections={detections_count} lidar_pkts={scan_points_count}",
        f"lidar_raw={raw_lidar_alert} lidar_conf={confirmed_lidar_alert}",
        f"fused_alert={fused_alert} voice_last={last_spoken_alert}",
        f"zones F/L/R/B={last_zone_counts['front']}/{last_zone_counts['left']}/"
        f"{last_zone_counts['right']}/{last_zone_counts['back']}",
    ]
    y = rect.y + 10
    for ln in lines:
        screen.blit(font.render(ln, True, (200, 230, 250)), (rect.x + 10, y))
        y += 24


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


def handle_keydown(key: int, screen: pygame.Surface) -> None:
    global running, camera_enabled, voice_enabled, ocr_enabled, ai_overlay_enabled, lidar_enabled
    global zones_fullscreen, debug_enabled, simulation_paused, fusion_enabled, pixels_per_meter
    if key == pygame.K_q or key == pygame.K_ESCAPE:
        running = False
    elif key == pygame.K_c:
        camera_enabled = not camera_enabled
    elif key == pygame.K_s:
        save_dashboard_and_csv(screen)
    elif key == pygame.K_v:
        voice_enabled = not voice_enabled
        if not voice_enabled:
            stop_current_voice()
    elif key == pygame.K_t:
        stop_current_voice()
        run_tts("Team Bravo AI HAT vision assistant ready")
    elif key == pygame.K_o:
        ocr_enabled = not ocr_enabled
    elif key == pygame.K_i:
        ai_overlay_enabled = not ai_overlay_enabled
    elif key == pygame.K_l:
        lidar_enabled = not lidar_enabled
    elif key == pygame.K_z:
        zones_fullscreen = not zones_fullscreen
    elif key == pygame.K_d:
        debug_enabled = not debug_enabled
    elif key == pygame.K_f:
        fusion_enabled = not fusion_enabled
    elif key == pygame.K_SPACE:
        simulation_paused = not simulation_paused
    elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        pixels_per_meter = min(180.0, pixels_per_meter + 8.0)
    elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        pixels_per_meter = max(40.0, pixels_per_meter - 8.0)


def main() -> None:
    global ai_labels
    ai_labels = load_labels(AI_LABELS_PATH)
    init_ai_hat()

    pygame.init()
    pygame.display.set_caption("Team Bravo AI HAT Camera + LiDAR Vision Assistant")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    lt = threading.Thread(target=lidar_thread_fn, daemon=True)
    ct = threading.Thread(target=camera_thread_fn, daemon=True)
    lt.start()
    ct.start()

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    handle_keydown(event.key, screen)

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
        time.sleep(0.05)
        pygame.quit()


if __name__ == "__main__":
    main()
