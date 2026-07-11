"""
Team Bravo Multi-Sensor Perception Dashboard for Blind Navigation Support
=========================================================================

Real-time 4-panel robotics perception dashboard for Raspberry Pi 5.

Hardware:
  - D6 AA55 2D LiDAR (USB serial)
  - Optional Raspberry Pi Camera (picamera2) or USB webcam (OpenCV)
  - Future: IMU, ultrasonic sensors

Install:
    sudo apt update
    sudo apt install python3-serial python3-pygame python3-opencv python3-numpy espeak-ng
    # Raspberry Pi Camera (Bookworm / Pi 5):
    sudo apt install python3-picamera2

Run:
    python3 team_bravo_multisensor_dashboard.py

Classroom demo (default):
    SIMULATED_MODE = True — runs without LiDAR hardware.
    Set SIMULATED_MODE = False when using a real D6 AA55 LiDAR.

Camera selection:
    CAMERA_BACKEND = "auto" tries Pi Camera (picamera2) first, then USB webcam.
    Use "picamera2", "usb", or "none" to force a specific source.

Safety:
    Prototype assistive navigation system only — NOT the sole safety device
    for a blind person. LiDAR can miss glass, shiny black surfaces, very low
    objects, and soft materials. A production system should combine LiDAR,
    camera, ultrasonic sensors, IMU, and extensive human testing.

This is NOT full SLAM:
    2D LiDAR only. True SLAM needs odometry, IMU, pose estimation, loop closure.
    The local map is centred on the sensor. Trajectory trail is simulated unless
    wheel encoders / IMU are added later.
"""

import csv
import math
import os
import random
import shutil
import struct
import subprocess
import threading
import time

import pygame
import serial
from serial.tools import list_ports

# OpenCV is optional — dashboard runs without camera
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    np = None

# Raspberry Pi Camera via libcamera stack (preferred on Pi 5)
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    Picamera2 = None

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
WIDTH = 1280
HEIGHT = 720
FPS_TARGET = 30
HEADER_H = 42
FOOTER_H = 26
PANEL_GAP = 4
PIXELS_PER_METER = 95
PIXELS_PER_METER_MIN = 50
PIXELS_PER_METER_MAX = 160

# ---------------------------------------------------------------------------
# Serial / LiDAR
# ---------------------------------------------------------------------------
PORT = "/dev/ttyUSB0"
BAUD = 230400
TIMEOUT = 0.5
SIMULATED_MODE = True   # True = classroom demo without LiDAR; False = real serial LiDAR

# Camera: "auto" | "picamera2" | "usb" | "none"
CAMERA_BACKEND = "auto"
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

MIN_RANGE_CM = 8
MAX_RANGE_M = 6.0
MAX_RANGE_CM = int(MAX_RANGE_M * 100)

# ---------------------------------------------------------------------------
# Occupancy grid map
# ---------------------------------------------------------------------------
GRID_RESOLUTION_M = 0.05
OCCUPIED_MIN_HITS = 3
WALL_STRONG_HITS = 6
FREE_MIN_HITS = 2
MAX_POINTS = 8000
POLAR_BIN_DEG = 1.0
ZONE_MIN_POINTS = 3
TRAJECTORY_MAX = 400

# LiDAR voice filter — when testing at a desk, your body in front triggers LiDAR
# (expected). This ignores very-close FRONT hits for voice only (panel still shows).
SELF_FILTER_VOICE = True
SELF_FILTER_FRONT_M = 0.50

# Camera vision alerts (objects / signs within 1 m)
CAMERA_ALERT_DISTANCE_M = 1.0
CAMERA_VISION_INTERVAL = 3          # run heavy detection every N camera frames
CAMERA_VOICE_MIN_DETECTIONS = 6
REF_PERSON_AREA_1M = 14000          # tune bbox area at ~1 m for 320x240
REF_SIGN_AREA_1M = 3200

# ---------------------------------------------------------------------------
# Blind navigation voice
# ---------------------------------------------------------------------------
ALERT_DISTANCE_M = 1.0
STRONG_WARNING_DISTANCE_M = 0.75
VERY_CLOSE_DISTANCE_M = 0.40
ENABLE_VOICE_ALERTS = True
VOICE_COOLDOWN_SECONDS = 2.0
VOICE_REPEAT_SECONDS = 6.0
VOICE_MIN_DETECTIONS = 8
ZONE_CLEAR_SCANS = 12
CLEAR_VOICE_MIN_GAP = 4.0
ESPEAK_SPEED = 155
ESPEAK_AMPLITUDE = 180
ESPEAK_WORD_GAP_MS = 6

# ---------------------------------------------------------------------------
# Save filenames
# ---------------------------------------------------------------------------
DASHBOARD_PNG = "team_bravo_dashboard.png"
LIDAR_POINTS_CSV = "team_bravo_lidar_points.csv"
OCCUPANCY_CSV = "team_bravo_occupancy_grid.csv"

# ---------------------------------------------------------------------------
# Theme colours (dark robotics HUD)
# ---------------------------------------------------------------------------
COLOR_BG = (4, 8, 18)
COLOR_HEADER = (8, 14, 30)
COLOR_PANEL = (10, 16, 32)
COLOR_PANEL_BORDER = (40, 120, 180)
COLOR_TITLE = (230, 240, 255)
COLOR_HUD = (160, 190, 220)
COLOR_CYAN = (0, 220, 255)
COLOR_GREEN = (60, 220, 120)
COLOR_YELLOW = (255, 220, 60)
COLOR_ORANGE = (255, 150, 40)
COLOR_RED = (255, 60, 60)
COLOR_BLUE = (60, 140, 255)
COLOR_FREE = (18, 32, 58)
COLOR_FREE_BRIGHT = (28, 50, 88)
COLOR_RING = (35, 90, 150)
COLOR_SENSOR = (230, 240, 255)

# Visual modes (M key)
VIS_MODE_STANDARD = 0
VIS_MODE_HIGH_CONTRAST = 1
VIS_MODE_MINIMAL = 2

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
data_lock = threading.Lock()
running = True
mapping_paused = False
visual_mode = VIS_MODE_STANDARD
show_debug = False
fullscreen = False
zones_fullscreen = False   # Z key: large zones overlay ON TOP of all 4 panels (panels stay visible)
voice_enabled = ENABLE_VOICE_ALERTS

points_xy = []
latest_scan_points = []
free_grid = {}
occupied_grid = {}
packet_count = 0

# Zone distances for panels
nearest_left_m = None
nearest_front_m = None
nearest_right_m = None
nearest_back_m = None
display_banner_text = "CLEAR"
raw_alert_state = "CLEAR"
candidate_alert_state = "CLEAR"
candidate_count = 0
confirmed_alert_state = "CLEAR"
last_spoken_alert_state = ""
clear_streak = 0
zone_counts = {}
last_voice_time = 0.0

# Camera vision (separate from LiDAR — was NOT triggering voice before)
camera_detections = []
camera_raw_alert = "CLEAR"
camera_candidate_alert = "CLEAR"
camera_candidate_count = 0
camera_confirmed_alert = "CLEAR"
camera_banner_text = ""
camera_frame_count = 0
hog_detector = None

current_voice_process = None
tts_executable = None
tts_checked = False
ser = None

# Simulated room + moving obstacles
SIM_WALLS = {"x_min": -2.0, "x_max": 4.0, "y_min": -3.0, "y_max": 3.0}
SIM_OBSTACLES = [
    {"x_min": 1.2, "x_max": 1.9, "y_min": 0.4, "y_max": 1.2},
    {"x_min": 1.9, "x_max": 2.5, "y_min": -1.3, "y_max": -0.5},
]
sim_mover_x = 0.9
sim_mover_y = 0.0
sim_mover_dx = 0.018
sim_left_x = 0.5
sim_left_y = -0.8
sim_right_x = 0.6
sim_right_y = 0.9

# Simulated trajectory (no odometry — demo only)
trajectory_trail = [(0.0, 0.0)]
sim_heading = 0.0

# Camera / optical flow
camera_cap = None
picam2 = None
camera_available = False
camera_source = "none"   # "none", "picamera2", "usb"
camera_lock = threading.Lock()
latest_camera_frame = None
prev_gray = None
flow_points = None
sim_flow_phase = 0.0


def full_zones_content_rect():
    """Full-screen obstacle zone area (between header and footer)."""
    return pygame.Rect(
        PANEL_GAP, HEADER_H + PANEL_GAP,
        WIDTH - 2 * PANEL_GAP,
        HEIGHT - HEADER_H - FOOTER_H - 2 * PANEL_GAP,
    )


# =============================================================================
# PANEL LAYOUT
# =============================================================================

def panel_content_rect(col, row):
    """Return pygame.Rect for panel interior (col,row) in 2x2 grid."""
    usable_w = WIDTH - PANEL_GAP * 3
    usable_h = HEIGHT - HEADER_H - FOOTER_H - PANEL_GAP * 3
    pw = usable_w // 2
    ph = usable_h // 2
    x = PANEL_GAP + col * (pw + PANEL_GAP)
    y = HEADER_H + PANEL_GAP + row * (ph + PANEL_GAP)
    return pygame.Rect(x, y, pw, ph)


PANEL_TITLES = [
    ("Optical Flow", 0, 0),
    ("Obstacle Zones", 1, 0),
    ("Metric Distance", 0, 1),
    ("Local Map", 1, 1),
]


def draw_panel_frame(screen, rect, title):
    """Draw panel border and title bar."""
    border_col = (80, 200, 255) if visual_mode == VIS_MODE_HIGH_CONTRAST else COLOR_PANEL_BORDER
    border_w = 3 if visual_mode == VIS_MODE_HIGH_CONTRAST else 2
    pygame.draw.rect(screen, COLOR_PANEL, rect)
    pygame.draw.rect(screen, border_col, rect, border_w)
    title_surf = pygame.font.SysFont("monospace", 14, bold=True).render(title, True, COLOR_TITLE)
    screen.blit(title_surf, (rect.x + 8, rect.y + 6))
    return pygame.Rect(rect.x + 4, rect.y + 24, rect.width - 8, rect.height - 28)


# =============================================================================
# LIDAR — D6 AA55 PACKET PARSING
# =============================================================================

def list_serial_ports():
    ports = [p.device for p in list_ports.comports()]
    if not ports:
        try:
            for name in sorted(os.listdir("/dev")):
                if name.startswith("ttyUSB") or name.startswith("ttyACM"):
                    ports.append("/dev/" + name)
        except OSError:
            pass
    return ports


def open_serial_port(port, baud, timeout):
    try:
        conn = serial.Serial(port, baud, timeout=timeout)
        print(f"LiDAR connected: {port} @ {baud}")
        return conn
    except serial.SerialException as exc:
        print(f"ERROR: Cannot open '{port}': {exc}")
        for p in list_serial_ports():
            print(f"  - {p}")
        raise SystemExit(1) from exc


def read_packet(connection):
    """Search for AA55 header and return one complete packet."""
    while running:
        try:
            b = connection.read(1)
        except serial.SerialException as exc:
            print(f"WARNING: Serial error: {exc}")
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


def parse_packet(packet):
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
        if lsn > 1:
            angle_deg = start_angle + angle_diff * i / (lsn - 1)
        else:
            angle_deg = start_angle
        angle_deg = angle_deg % 360.0
        if MIN_RANGE_CM <= distance_cm <= MAX_RANGE_CM:
            points.append((angle_deg, distance_cm))
    return points


def polar_to_xy(angle_deg, distance_cm):
    """Polar -> metres. 0° = forward (+X), 90° = right (+Y)."""
    distance_m = distance_cm / 100.0
    rad = math.radians(angle_deg)
    x = distance_m * math.cos(rad)
    y = distance_m * math.sin(rad)
    return x, y, distance_m, angle_deg


def grid_index(x_m, y_m):
    return round(x_m / GRID_RESOLUTION_M), round(y_m / GRID_RESOLUTION_M)


def grid_to_world(ix, iy):
    return ix * GRID_RESOLUTION_M, iy * GRID_RESOLUTION_M


def world_to_panel(x_m, y_m, rect):
    """Map coordinates to panel pixel coords (sensor at centre, forward up)."""
    cx = rect.centerx
    cy = rect.centery
    sx = int(cx + y_m * PIXELS_PER_METER)
    sy = int(cy - x_m * PIXELS_PER_METER)
    return sx, sy


def bresenham_line_cells(x0, y0, x1, y1):
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


def carve_ray_to_obstacle(x_m, y_m):
    end_ix, end_iy = grid_index(x_m, y_m)
    line = bresenham_line_cells(0, 0, end_ix, end_iy)
    for i, (ix, iy) in enumerate(line):
        key = (ix, iy)
        if i == len(line) - 1:
            occupied_grid[key] = occupied_grid.get(key, 0) + 1
        else:
            if occupied_grid.get(key, 0) >= OCCUPIED_MIN_HITS:
                continue
            free_grid[key] = free_grid.get(key, 0) + 1


def smooth_scan_polar(scan_polar):
    bins = {}
    for angle_deg, distance_cm in scan_polar:
        bin_key = int(round(angle_deg / POLAR_BIN_DEG))
        bins.setdefault(bin_key, []).append(distance_cm)
    smoothed = []
    for bin_key, distances in bins.items():
        distances.sort()
        median_cm = distances[len(distances) // 2]
        angle_deg = (bin_key * POLAR_BIN_DEG) % 360.0
        smoothed.append((angle_deg, median_cm))
    return smoothed


def update_trajectory_demo():
    """
    Simulated local path trail — NOT real odometry.
    Accurate trajectory needs IMU / wheel encoders in a future version.
    """
    global sim_heading, trajectory_trail
    t = time.time()
    sim_heading = 0.15 * math.sin(t * 0.3)
    dx = 0.012 * math.cos(sim_heading)
    dy = 0.012 * math.sin(sim_heading)
    if trajectory_trail:
        lx, ly = trajectory_trail[-1]
        trajectory_trail.append((lx + dx, ly + dy))
    else:
        trajectory_trail.append((0.0, 0.0))
    if len(trajectory_trail) > TRAJECTORY_MAX:
        del trajectory_trail[:-TRAJECTORY_MAX]


def process_scan(scan_polar):
    global packet_count, latest_scan_points
    scan_polar = smooth_scan_polar(scan_polar)
    now = time.time()
    batch = []
    with data_lock:
        for angle_deg, distance_cm in scan_polar:
            x_m, y_m, distance_m, angle = polar_to_xy(angle_deg, distance_cm)
            batch.append((x_m, y_m, distance_m, angle, now))
            carve_ray_to_obstacle(x_m, y_m)
        points_xy.extend(batch)
        if len(points_xy) > MAX_POINTS:
            del points_xy[:-MAX_POINTS]
        latest_scan_points = [(p[0], p[1], p[2], p[3]) for p in batch]
        packet_count += 1
    update_trajectory_demo()


# =============================================================================
# SIMULATION MODE
# =============================================================================

def sim_moving_obstacle_distance(angle_deg):
    global sim_mover_x, sim_mover_y, sim_mover_dx
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    best = None

    def check_point(px, py, radius=0.22):
        nonlocal best
        t = px * dx + py * dy
        if t <= 0.05:
            return
        cx, cy = t * dx, t * dy
        if math.hypot(cx - px, cy - py) <= radius:
            if best is None or t < best:
                best = t

    check_point(sim_mover_x, sim_mover_y, 0.25)
    check_point(sim_left_x, sim_left_y, 0.20)
    check_point(sim_right_x, sim_right_y, 0.20)
    return None if best is None else best * 100.0


def update_sim_movers():
    global sim_mover_x, sim_mover_dx
    sim_mover_x += sim_mover_dx
    if sim_mover_x > 1.15 or sim_mover_x < 0.45:
        sim_mover_dx *= -1


def sim_ray_distance(angle_deg):
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    best = MAX_RANGE_M

    def try_wall_x(xw):
        nonlocal best
        if abs(dx) < 1e-9:
            return
        t = xw / dx
        if t > 0:
            yh = t * dy
            if SIM_WALLS["y_min"] <= yh <= SIM_WALLS["y_max"]:
                best = min(best, t)

    def try_wall_y(yw):
        nonlocal best
        if abs(dy) < 1e-9:
            return
        t = yw / dy
        if t > 0:
            xh = t * dx
            if SIM_WALLS["x_min"] <= xh <= SIM_WALLS["x_max"]:
                best = min(best, t)

    try_wall_x(SIM_WALLS["x_min"])
    try_wall_x(SIM_WALLS["x_max"])
    try_wall_y(SIM_WALLS["y_min"])
    try_wall_y(SIM_WALLS["y_max"])

    for box in SIM_OBSTACLES:
        for xw in (box["x_min"], box["x_max"]):
            if abs(dx) > 1e-9:
                t = xw / dx
                if t > 0:
                    yh = t * dy
                    if box["y_min"] <= yh <= box["y_max"]:
                        best = min(best, t)
        for yw in (box["y_min"], box["y_max"]):
            if abs(dy) > 1e-9:
                t = yw / dy
                if t > 0:
                    xh = t * dx
                    if box["x_min"] <= xh <= box["x_max"]:
                        best = min(best, t)

    mover_cm = sim_moving_obstacle_distance(angle_deg)
    if mover_cm is not None:
        best = min(best, mover_cm / 100.0)

    best += random.uniform(-0.012, 0.012)
    return max(MIN_RANGE_CM / 100.0, min(best, MAX_RANGE_M)) * 100.0


def generate_simulated_scan():
    update_sim_movers()
    base = random.uniform(0, 360)
    return [
        ((base + i * 3.0) % 360.0, sim_ray_distance((base + i * 3.0) % 360.0))
        for i in range(28)
    ]


def serial_reader_loop(connection):
    global running
    print(f"LiDAR reader started ({'SIM' if SIMULATED_MODE else 'SERIAL'}).")
    while running:
        if mapping_paused:
            time.sleep(0.05)
            continue
        if SIMULATED_MODE:
            process_scan(generate_simulated_scan())
            time.sleep(0.035)
        else:
            packet = read_packet(connection)
            if packet:
                process_scan(parse_packet(packet))
            time.sleep(0.001)
    print("LiDAR reader stopped.")


# =============================================================================
# CAMERA + OPTICAL FLOW (OPTIONAL)
# =============================================================================

def _apply_optical_flow_arrows(frame_bgr):
    """Draw sparse optical-flow arrows on a BGR frame. Returns RGB numpy array."""
    global prev_gray, flow_points
    if not CV2_AVAILABLE or np is None:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if CV2_AVAILABLE else frame_bgr

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        if flow_points is None or len(flow_points) < 20:
            flow_points = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=80, qualityLevel=0.3,
                minDistance=12, blockSize=7)
        if flow_points is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, flow_points, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            if next_pts is not None and status is not None:
                good_old = flow_points[status.flatten() == 1]
                good_new = next_pts[status.flatten() == 1]
                flow_points = good_new.reshape(-1, 1, 2)
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    if i % 3 != 0:
                        continue
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.arrowedLine(
                        frame_bgr, (int(c), int(d)), (int(a), int(b)),
                        (0, 255, 255), 1, tipLength=0.35)
    prev_gray = gray
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def init_picamera2():
    """Start Raspberry Pi Camera via picamera2."""
    global picam2, camera_available, camera_source
    if not PICAMERA2_AVAILABLE:
        return False
    try:
        cam = Picamera2()
        config = cam.create_preview_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"})
        cam.configure(config)
        cam.start()
        time.sleep(0.3)
        picam2 = cam
        camera_available = True
        camera_source = "picamera2"
        print("Camera OK: Raspberry Pi Camera (picamera2)")
        return True
    except Exception as exc:
        print(f"picamera2 unavailable: {exc}")
        picam2 = None
        return False


def init_usb_camera():
    """Start USB webcam via OpenCV VideoCapture."""
    global camera_cap, camera_available, camera_source
    if not CV2_AVAILABLE:
        return False
    for idx in (0, 1, 2):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            camera_cap = cap
            camera_available = True
            camera_source = "usb"
            print(f"Camera OK: USB webcam (index {idx})")
            return True
    return False


def init_camera():
    """
    Initialise camera for optical-flow panel.
    auto: picamera2 first (Pi Camera), then USB webcam.
    """
    global camera_available, camera_source
    camera_available = False
    camera_source = "none"

    if CAMERA_BACKEND == "none":
        print("Camera disabled — optical flow panel uses simulation.")
        return

    if CAMERA_BACKEND == "auto":
        order = ["picamera2", "usb"]
    elif CAMERA_BACKEND == "picamera2":
        order = ["picamera2"]
    elif CAMERA_BACKEND == "usb":
        order = ["usb"]
    else:
        print(f"Unknown CAMERA_BACKEND '{CAMERA_BACKEND}' — using auto.")
        order = ["picamera2", "usb"]

    for backend in order:
        if backend == "picamera2" and init_picamera2():
            return
        if backend == "usb" and init_usb_camera():
            return

    print("No camera found — optical flow panel uses simulation.")


def grab_camera_frame_bgr():
    """Return a BGR numpy frame from active camera, or None."""
    if not camera_available:
        return None
    if camera_source == "picamera2" and picam2 is not None:
        try:
            rgb = picam2.capture_array()
            if rgb is None:
                return None
            if CV2_AVAILABLE:
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return rgb
        except Exception as exc:
            print(f"WARNING: picamera2 capture failed: {exc}")
            return None
    if camera_source == "usb" and camera_cap is not None:
        ret, frame = camera_cap.read()
        return frame if ret else None
    return None


def camera_reader_loop():
    """Background thread: grab frames, optical flow, and object/sign vision."""
    global latest_camera_frame, camera_detections, camera_frame_count
    if not camera_available:
        return

    init_hog_detector()
    last_dets = []

    while running:
        frame = grab_camera_frame_bgr()
        if frame is None:
            time.sleep(0.05)
            continue

        if CV2_AVAILABLE:
            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            camera_frame_count += 1
            run_hog = (camera_frame_count % CAMERA_VISION_INTERVAL == 0)
            dets = analyze_camera_frame(frame, run_hog=run_hog)
            if dets or run_hog:
                last_dets = dets
                alert, banner = pick_camera_alert(dets)
                update_camera_voice_state_machine(alert, banner)
            frame = draw_vision_overlay(frame, last_dets)
            rgb = _apply_optical_flow_arrows(frame)
        else:
            rgb = frame

        with camera_lock:
            latest_camera_frame = rgb
            camera_detections = list(last_dets)
        time.sleep(0.02)


def release_camera():
    """Clean shutdown for picamera2 and USB webcam."""
    global camera_cap, picam2, camera_available, camera_source
    if picam2 is not None:
        try:
            picam2.stop()
            picam2.close()
        except Exception:
            pass
        picam2 = None
    if camera_cap is not None:
        try:
            camera_cap.release()
        except Exception:
            pass
        camera_cap = None
    camera_available = False
    camera_source = "none"


def camera_status_label():
    """Short label for HUD header."""
    if camera_source == "picamera2":
        return "PI-CAM"
    if camera_source == "usb":
        return "USB"
    return "SIM"


# =============================================================================
# CAMERA VISION — OBJECTS & SIGNS (≤ 1 m)
# =============================================================================

def init_hog_detector():
    global hog_detector
    if not CV2_AVAILABLE or hog_detector is not None:
        return hog_detector is not None
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        hog_detector = hog
        print("Camera vision: HOG person detector ready.")
        return True
    except Exception as exc:
        print(f"Camera vision HOG init failed: {exc}")
        return False


def estimate_distance_from_bbox(area_px, ref_area_at_1m):
    """Monocular distance estimate from bounding-box size (prototype)."""
    if area_px <= 0:
        return 9.9
    return max(0.15, math.sqrt(ref_area_at_1m / area_px))


def _signs_from_color_mask(frame_bgr, lower, upper, color_name):
    """Find coloured sign-like blobs in HSV range."""
    if not CV2_AVAILABLE:
        return []
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    h_img, w_img = frame_bgr.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 350:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 18 or bh < 18:
            continue
        cx = x + bw // 2
        # Signs ahead are usually in centre-lower field of view
        if cx < w_img * 0.15 or cx > w_img * 0.85:
            continue
        dist_m = estimate_distance_from_bbox(bw * bh, REF_SIGN_AREA_1M)
        if dist_m > CAMERA_ALERT_DISTANCE_M:
            continue
        aspect = bw / max(bh, 1)
        if color_name == "red":
            label = "STOP sign"
        elif color_name == "yellow":
            label = "Caution sign" if aspect < 1.6 else "Warning sign"
        elif color_name == "blue":
            label = "Information sign"
        else:
            label = "Sign"
        found.append({
            "label": label, "distance_m": dist_m,
            "bbox": (x, y, bw, bh), "kind": "sign",
        })
    return found


def detect_signs_in_frame(frame_bgr):
    """Colour-based sign detection (prototype — not OCR)."""
    if not CV2_AVAILABLE or np is None:
        return []
    detections = []
    # Red (two HSV wraps)
    detections.extend(_signs_from_color_mask(frame_bgr, (0, 90, 70), (10, 255, 255), "red"))
    detections.extend(_signs_from_color_mask(frame_bgr, (160, 90, 70), (179, 255, 255), "red"))
    detections.extend(_signs_from_color_mask(frame_bgr, (18, 90, 90), (38, 255, 255), "yellow"))
    detections.extend(_signs_from_color_mask(frame_bgr, (95, 80, 60), (125, 255, 255), "blue"))
    return detections


def detect_people_in_frame(frame_bgr):
    """HOG person detector — only report if estimated ≤ 1 m."""
    if hog_detector is None or not CV2_AVAILABLE:
        return []
    found = []
    try:
        boxes, weights = hog_detector.detectMultiScale(
            frame_bgr, winStride=(8, 8), padding=(8, 8), scale=1.05)
        for (x, y, bw, bh), w in zip(boxes, weights):
            if w < 0.4:
                continue
            dist_m = estimate_distance_from_bbox(bw * bh, REF_PERSON_AREA_1M)
            if dist_m > CAMERA_ALERT_DISTANCE_M:
                continue
            found.append({
                "label": "Person", "distance_m": dist_m,
                "bbox": (int(x), int(y), int(bw), int(bh)), "kind": "person",
            })
    except Exception:
        pass
    return found


def detect_objects_in_frame(frame_bgr):
    """Large foreground objects in centre view (simple contour heuristic)."""
    if not CV2_AVAILABLE:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = frame_bgr.shape[:2]
    found = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2500:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cx = x + bw // 2
        if cx < w_img * 0.25 or cx > w_img * 0.75 or y < h_img * 0.15:
            continue
        dist_m = estimate_distance_from_bbox(bw * bh, REF_SIGN_AREA_1M * 2.5)
        if dist_m > CAMERA_ALERT_DISTANCE_M:
            continue
        found.append({
            "label": "Object", "distance_m": dist_m,
            "bbox": (x, y, bw, bh), "kind": "object",
        })
        if len(found) >= 2:
            break
    return found


def analyze_camera_frame(frame_bgr, run_hog=True):
    """
    Detect people, signs, and objects within CAMERA_ALERT_DISTANCE_M.
    Returns list of detection dicts with label, distance_m, bbox, kind.
    """
    if not CV2_AVAILABLE:
        return []
    if run_hog:
        init_hog_detector()
    detections = []
    detections.extend(detect_signs_in_frame(frame_bgr))
    if run_hog:
        detections.extend(detect_people_in_frame(frame_bgr))
    if not detections and run_hog:
        detections.extend(detect_objects_in_frame(frame_bgr))
    detections.sort(key=lambda d: d["distance_m"])
    return detections[:5]


def pick_camera_alert(detections):
    """Choose highest-priority camera alert from detections ≤ 1 m."""
    if not detections:
        return "CLEAR", ""
    best = detections[0]
    label = best["label"]
    if "STOP" in label:
        return "CAMERA_STOP_SIGN", f"STOP SIGN {best['distance_m']:.1f}m"
    if "Caution" in label or "Warning" in label:
        return "CAMERA_CAUTION_SIGN", f"CAUTION SIGN {best['distance_m']:.1f}m"
    if best["kind"] == "person":
        return "CAMERA_PERSON", f"PERSON {best['distance_m']:.1f}m"
    if best["kind"] == "sign":
        return "CAMERA_SIGN", f"SIGN {best['distance_m']:.1f}m"
    return "CAMERA_OBJECT", f"OBJECT {best['distance_m']:.1f}m"


def draw_vision_overlay(frame_bgr, detections):
    """Draw bounding boxes on camera frame."""
    out = frame_bgr.copy()
    for det in detections:
        x, y, bw, bh = det["bbox"]
        col = (0, 255, 255)
        if det["kind"] == "person":
            col = (0, 200, 255)
        elif det["kind"] == "sign":
            col = (0, 80, 255)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), col, 2)
        txt = f"{det['label']} {det['distance_m']:.1f}m"
        cv2.putText(out, txt, (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return out


def update_camera_voice_state_machine(raw_alert, banner):
    global camera_raw_alert, camera_candidate_alert, camera_candidate_count
    global camera_confirmed_alert, camera_banner_text
    camera_raw_alert = raw_alert
    camera_banner_text = banner
    if raw_alert == camera_candidate_alert:
        camera_candidate_count += 1
    else:
        camera_candidate_alert = raw_alert
        camera_candidate_count = 1
    if camera_candidate_count >= CAMERA_VOICE_MIN_DETECTIONS:
        camera_confirmed_alert = camera_candidate_alert
    elif raw_alert == "CLEAR":
        camera_confirmed_alert = "CLEAR"


def camera_warning_to_speech(alert_state):
    mapping = {
        "CAMERA_PERSON": "Person close ahead",
        "CAMERA_STOP_SIGN": "Stop sign ahead",
        "CAMERA_CAUTION_SIGN": "Caution sign ahead",
        "CAMERA_SIGN": "Sign ahead",
        "CAMERA_OBJECT": "Object close ahead",
    }
    return mapping.get(alert_state, "")


def lidar_suppressed_by_camera(lidar_alert, camera_alert):
    """Avoid duplicate generic LiDAR front alert when camera has semantic detection."""
    if camera_alert == "CLEAR":
        return False
    if lidar_alert in ("NORMAL_FRONT", "STRONG_FRONT") and camera_alert.startswith("CAMERA_"):
        return True
    return False


def distance_to_color(distance_m):
    """Heatmap colour: red=close, green/cyan=safe, blue=far."""
    if distance_m <= VERY_CLOSE_DISTANCE_M:
        return COLOR_RED
    if distance_m <= STRONG_WARNING_DISTANCE_M:
        return COLOR_ORANGE
    if distance_m <= ALERT_DISTANCE_M:
        return COLOR_YELLOW
    if distance_m <= 2.5:
        return COLOR_GREEN
    if distance_m <= 4.0:
        return COLOR_CYAN
    return COLOR_BLUE


def zone_level(distance_m):
    """Zone colour level for obstacle panel."""
    if distance_m is None:
        return "clear", COLOR_GREEN
    if distance_m <= VERY_CLOSE_DISTANCE_M:
        return "stop", COLOR_RED
    if distance_m <= STRONG_WARNING_DISTANCE_M:
        return "strong", COLOR_ORANGE
    if distance_m <= ALERT_DISTANCE_M:
        return "alert", COLOR_YELLOW
    return "clear", COLOR_GREEN


# =============================================================================
# BLIND NAVIGATION — ZONE DETECTION + VOICE
# =============================================================================

def _zone_nearest(points, pred):
    nearest = None
    for x_m, y_m, distance_m, _a in points:
        if pred(x_m, y_m, distance_m):
            if nearest is None or distance_m < nearest:
                nearest = distance_m
    return nearest


def _zone_count(points, pred):
    return sum(1 for x_m, y_m, d, _a in points if pred(x_m, y_m, d))


def _lidar_point_counts_for_voice(x_m, y_m, distance_m):
    """Skip very-close front hits when desk self-filter is on (voice only)."""
    if (SELF_FILTER_VOICE and x_m > 0 and abs(y_m) <= 0.45
            and distance_m < SELF_FILTER_FRONT_M):
        return False
    return True


def detect_obstacles_for_blind_user(latest_points, for_voice=False):
    """
    Cluster-based zone detection.
    for_voice=True applies desk self-filter so leaning over the sensor does not spam alerts.
    """
    zc = {
        "front": 0, "back": 0, "left": 0, "right": 0,
        "vc_front": 0, "vc_back": 0, "vc_left": 0, "vc_right": 0,
    }

    for x_m, y_m, distance_m, _a in latest_points:
        if for_voice and not _lidar_point_counts_for_voice(x_m, y_m, distance_m):
            continue
        if x_m > 0 and abs(y_m) <= 0.45 and distance_m <= ALERT_DISTANCE_M:
            zc["front"] += 1
        if x_m < 0 and abs(y_m) <= 0.45 and distance_m <= 0.75:
            zc["back"] += 1
        if y_m < -0.35 and -0.3 <= x_m <= 1.2 and distance_m <= ALERT_DISTANCE_M:
            zc["left"] += 1
        if y_m > 0.35 and -0.3 <= x_m <= 1.2 and distance_m <= ALERT_DISTANCE_M:
            zc["right"] += 1
        if distance_m <= VERY_CLOSE_DISTANCE_M:
            if x_m > 0 and abs(y_m) <= 0.45:
                zc["vc_front"] += 1
            elif x_m < 0 and abs(y_m) <= 0.45:
                zc["vc_back"] += 1
            elif y_m < -0.25:
                zc["vc_left"] += 1
            elif y_m > 0.25:
                zc["vc_right"] += 1

    nf = _zone_nearest(latest_points, lambda x, y, d: x > 0 and abs(y) <= 0.45 and d <= ALERT_DISTANCE_M)
    nb = _zone_nearest(latest_points, lambda x, y, d: x < 0 and abs(y) <= 0.45 and d <= 0.75)
    nl = _zone_nearest(latest_points, lambda x, y, d: y < -0.35 and -0.3 <= x <= 1.2 and d <= ALERT_DISTANCE_M)
    nr = _zone_nearest(latest_points, lambda x, y, d: y > 0.35 and -0.3 <= x <= 1.2 and d <= ALERT_DISTANCE_M)

    if zc["vc_front"] >= ZONE_MIN_POINTS:
        return "VERY_CLOSE_FRONT", nf, zc
    if zc["vc_back"] >= ZONE_MIN_POINTS:
        return "VERY_CLOSE_BACK", nb, zc
    if zc["vc_left"] >= ZONE_MIN_POINTS:
        return "VERY_CLOSE_LEFT", nl, zc
    if zc["vc_right"] >= ZONE_MIN_POINTS:
        return "VERY_CLOSE_RIGHT", nr, zc
    if zc["front"] >= ZONE_MIN_POINTS and nf is not None and nf <= STRONG_WARNING_DISTANCE_M:
        return "STRONG_FRONT", nf, zc
    if zc["left"] >= ZONE_MIN_POINTS and nl is not None and nl <= STRONG_WARNING_DISTANCE_M:
        return "STRONG_LEFT", nl, zc
    if zc["right"] >= ZONE_MIN_POINTS and nr is not None and nr <= STRONG_WARNING_DISTANCE_M:
        return "STRONG_RIGHT", nr, zc
    if zc["left"] >= ZONE_MIN_POINTS and zc["right"] >= ZONE_MIN_POINTS:
        return "BOTH_SIDES", min(nl, nr), zc
    if zc["front"] >= ZONE_MIN_POINTS:
        return "NORMAL_FRONT", nf, zc
    if zc["left"] >= ZONE_MIN_POINTS:
        return "NORMAL_LEFT", nl, zc
    if zc["right"] >= ZONE_MIN_POINTS:
        return "NORMAL_RIGHT", nr, zc
    if zc["back"] >= ZONE_MIN_POINTS:
        return "BACK", nb, zc
    overall = min([d for d in (nf, nb, nl, nr) if d is not None], default=None)
    return "CLEAR", overall, zc


def compute_direction_distances(latest_points):
    back_d = _zone_nearest(latest_points, lambda x, y, d: x < 0 and abs(y) <= 0.45 and d <= 0.75)
    left_d = _zone_nearest(latest_points, lambda x, y, d: y < -0.35 and -0.3 <= x <= 1.2 and d <= ALERT_DISTANCE_M)
    front_d = _zone_nearest(latest_points, lambda x, y, d: x > 0 and abs(y) <= 0.45 and d <= ALERT_DISTANCE_M)
    right_d = _zone_nearest(latest_points, lambda x, y, d: y > 0.35 and -0.3 <= x <= 1.2 and d <= ALERT_DISTANCE_M)
    return back_d, left_d, front_d, right_d


def alert_to_banner(alert_state):
    mapping = {
        "CLEAR": "CLEAR",
        "NORMAL_FRONT": "OBSTACLE AHEAD",
        "STRONG_FRONT": "CAREFUL: OBSTACLE AHEAD",
        "VERY_CLOSE_FRONT": "STOP: VERY CLOSE",
        "NORMAL_LEFT": "OBSTACLE LEFT",
        "STRONG_LEFT": "CAREFUL: CLOSE ON LEFT",
        "VERY_CLOSE_LEFT": "STOP: VERY CLOSE ON LEFT",
        "NORMAL_RIGHT": "OBSTACLE RIGHT",
        "STRONG_RIGHT": "CAREFUL: CLOSE ON RIGHT",
        "VERY_CLOSE_RIGHT": "STOP: VERY CLOSE ON RIGHT",
        "BOTH_SIDES": "OBSTACLES BOTH SIDES",
        "BACK": "OBSTACLE BEHIND",
        "VERY_CLOSE_BACK": "STOP: VERY CLOSE BEHIND",
    }
    return mapping.get(alert_state, alert_state)


def is_very_close_alert(alert_state):
    return alert_state.startswith("VERY_CLOSE_")


def update_voice_state_machine(raw_alert):
    global raw_alert_state, candidate_alert_state, candidate_count
    global confirmed_alert_state, clear_streak

    raw_alert_state = raw_alert
    if raw_alert == candidate_alert_state:
        candidate_count += 1
    else:
        candidate_alert_state = raw_alert
        candidate_count = 1

    if candidate_count >= VOICE_MIN_DETECTIONS:
        confirmed_alert_state = candidate_alert_state
    elif raw_alert == "CLEAR":
        confirmed_alert_state = "CLEAR"

    if raw_alert == "CLEAR":
        clear_streak += 1
    else:
        clear_streak = 0


def warning_to_speech(alert_state):
    mapping = {
        "CLEAR": "Path clear",
        "NORMAL_FRONT": "Obstacle ahead",
        "STRONG_FRONT": "Careful. Obstacle ahead",
        "VERY_CLOSE_FRONT": "Stop. Obstacle very close ahead",
        "NORMAL_LEFT": "Obstacle on your left",
        "STRONG_LEFT": "Careful. Obstacle close on your left",
        "VERY_CLOSE_LEFT": "Stop. Obstacle very close on your left",
        "NORMAL_RIGHT": "Obstacle on your right",
        "STRONG_RIGHT": "Careful. Obstacle close on your right",
        "VERY_CLOSE_RIGHT": "Stop. Obstacle very close on your right",
        "BOTH_SIDES": "Obstacles on both sides",
        "BACK": "Obstacle behind you",
        "VERY_CLOSE_BACK": "Stop. Obstacle very close behind",
    }
    return mapping.get(alert_state, "Obstacle nearby")


def find_tts_executable():
    for name in ("espeak-ng", "espeak"):
        path = shutil.which(name)
        if path:
            return path
    for path in ("/usr/bin/espeak-ng", "/usr/bin/espeak"):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def init_voice_system():
    global tts_executable, tts_checked
    tts_executable = find_tts_executable()
    tts_checked = True
    if tts_executable:
        print(f"Voice OK: {tts_executable}")
    else:
        print("Install: sudo apt install espeak-ng")


def check_tts():
    global tts_executable, tts_checked
    if not tts_checked:
        init_voice_system()
    return tts_executable is not None


def stop_current_voice():
    global current_voice_process
    if current_voice_process is not None and current_voice_process.poll() is None:
        try:
            current_voice_process.terminate()
            current_voice_process.wait(timeout=0.5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                current_voice_process.kill()
            except OSError:
                pass
    current_voice_process = None


def run_tts(text):
    global current_voice_process
    if not check_tts():
        return False
    args = [tts_executable, "-s", str(ESPEAK_SPEED), "-a", str(ESPEAK_AMPLITUDE),
            "-g", str(ESPEAK_WORD_GAP_MS), text]
    try:
        current_voice_process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except OSError:
        pass
    try:
        safe = text.replace('"', "'")
        subprocess.Popen(
            f'{tts_executable} -s {ESPEAK_SPEED} -a {ESPEAK_AMPLITUDE} -g {ESPEAK_WORD_GAP_MS} "{safe}" &',
            shell=True)
        return True
    except OSError:
        return False


def speak_voice_alert(alert_state, force=False):
    global last_spoken_alert_state, last_voice_time
    if not voice_enabled or alert_state is None:
        return False
    now = time.time()
    if not force and now - last_voice_time < VOICE_COOLDOWN_SECONDS:
        return False
    if force:
        stop_current_voice()
    text = camera_warning_to_speech(alert_state) or warning_to_speech(alert_state)
    if not text:
        return False
    print(f">>> VOICE: {text}")
    if not run_tts(text):
        print("\a", end="", flush=True)
        return False
    last_spoken_alert_state = alert_state
    last_voice_time = now
    return True


def effective_lidar_alert():
    """Suppress generic LiDAR front alert when camera has a semantic detection."""
    if lidar_suppressed_by_camera(confirmed_alert_state, camera_confirmed_alert):
        return "CLEAR"
    return confirmed_alert_state


def process_voice_alerts():
    global last_spoken_alert_state, last_voice_time
    if not voice_enabled:
        return
    now = time.time()
    lidar = effective_lidar_alert()
    camera = camera_confirmed_alert
    elapsed = now - last_voice_time

    # LiDAR emergency STOP always highest priority
    if is_very_close_alert(lidar):
        if lidar != last_spoken_alert_state or elapsed >= VOICE_REPEAT_SECONDS:
            speak_voice_alert(lidar, force=True)
        return

    # Camera semantic alert (person / sign / object within 1 m)
    if camera != "CLEAR":
        if camera != last_spoken_alert_state:
            if elapsed >= VOICE_COOLDOWN_SECONDS:
                speak_voice_alert(camera, force=False)
        elif elapsed >= VOICE_REPEAT_SECONDS:
            speak_voice_alert(camera, force=False)
        return

    if lidar == "CLEAR":
        if last_spoken_alert_state and last_spoken_alert_state != "CLEAR":
            if clear_streak >= ZONE_CLEAR_SCANS and elapsed >= CLEAR_VOICE_MIN_GAP:
                speak_voice_alert("CLEAR", force=False)
        return

    if lidar == last_spoken_alert_state:
        if elapsed >= VOICE_REPEAT_SECONDS:
            speak_voice_alert(lidar, force=False)
        return

    if elapsed >= VOICE_COOLDOWN_SECONDS:
        speak_voice_alert(lidar, force=False)


def test_voice():
    stop_current_voice()
    global last_voice_time
    last_voice_time = 0.0
    run_tts("Path clear. Team Bravo dashboard ready.")


def update_alerts():
    global nearest_left_m, nearest_front_m, nearest_right_m, nearest_back_m, zone_counts
    with data_lock:
        latest = list(latest_scan_points)
    # Display uses all LiDAR points; voice uses self-filter for desk testing
    alert_display, _nearest, zc = detect_obstacles_for_blind_user(latest, for_voice=False)
    alert_voice, _, _ = detect_obstacles_for_blind_user(latest, for_voice=True)
    zone_counts = zc
    back_d, left_d, front_d, right_d = compute_direction_distances(latest)
    nearest_back_m, nearest_left_m, nearest_front_m, nearest_right_m = back_d, left_d, front_d, right_d
    update_voice_state_machine(alert_voice)
    # Banner shows LiDAR + camera info
    if camera_banner_text and camera_raw_alert != "CLEAR":
        display_banner_text_local = f"{alert_to_banner(alert_display)} | {camera_banner_text}"
    else:
        display_banner_text_local = alert_to_banner(alert_display)
    global display_banner_text
    display_banner_text = display_banner_text_local
    process_voice_alerts()


# =============================================================================
# PANEL DRAWING
# =============================================================================

def draw_simulated_optical_flow(screen, rect):
    """Animated arrows when no camera is available."""
    global sim_flow_phase
    sim_flow_phase += 0.04
    surf = pygame.Surface((rect.width, rect.height))
    for y in range(0, rect.height, 28):
        for x in range(0, rect.width, 28):
            angle = math.sin(sim_flow_phase + x * 0.02 + y * 0.015) * 1.2
            length = 14 + 6 * math.sin(sim_flow_phase * 2 + x * 0.01)
            ex = x + length * math.cos(angle)
            ey = y + length * math.sin(angle)
            hue = int(128 + 127 * math.sin(sim_flow_phase + x * 0.03))
            col = (hue // 3, hue, 255)
            pygame.draw.line(surf, col, (x, y), (ex, ey), 2)
            pygame.draw.circle(surf, col, (int(ex), int(ey)), 2)
    # Gradient background
    bg = pygame.Surface((rect.width, rect.height))
    for i in range(rect.height):
        shade = 8 + int(20 * i / rect.height)
        pygame.draw.line(bg, (shade, shade + 10, shade + 25), (0, i), (rect.width, i))
    bg.blit(surf, (0, 0))
    screen.blit(bg, rect.topleft)
    lbl = pygame.font.SysFont("monospace", 11).render("SIMULATED OPTICAL FLOW", True, (120, 160, 200))
    screen.blit(lbl, (rect.x + 8, rect.bottom - 18))


def draw_optical_flow_panel(screen, rect):
    if camera_available:
        with camera_lock:
            frame = latest_camera_frame
        if frame is not None:
            img = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            img = pygame.transform.scale(img, (rect.width, rect.height))
            screen.blit(img, rect.topleft)
            src = "Pi Camera" if camera_source == "picamera2" else "USB Cam"
            lbl = pygame.font.SysFont("monospace", 10).render(src, True, (140, 200, 220))
            screen.blit(lbl, (rect.x + 8, rect.bottom - 16))
            return
    draw_simulated_optical_flow(screen, rect)


def obstacle_zone_rects(rect):
    """Cross layout: FRONT/BACK/LEFT/RIGHT fill the entire panel."""
    pad = 4
    gutter = 10
    r = rect.inflate(-pad * 2, -pad * 2)
    w, h = r.width, r.height
    cx = r.centerx
    top_h = int(h * 0.38)
    bot_h = int(h * 0.38)
    mid_h = h - top_h - bot_h - gutter
    arm_w = (w - gutter) // 2

    front = pygame.Rect(r.x, r.y, w, top_h)
    back = pygame.Rect(r.x, r.bottom - bot_h, w, bot_h)
    mid_y = r.y + top_h + gutter // 2
    left = pygame.Rect(r.x, mid_y, arm_w, mid_h)
    right = pygame.Rect(r.right - arm_w, mid_y, arm_w, mid_h)
    center = pygame.Rect(cx - 78, mid_y, 156, mid_h)
    return front, back, left, right, center


def draw_zone_block(screen, block, label, dist, font, font_sm):
    """Draw one directional zone block filling its rectangle."""
    _level, col = zone_level(dist)
    fill = pygame.Surface((block.width, block.height), pygame.SRCALPHA)
    fill.fill((*col, 100))
    screen.blit(fill, block.topleft)
    pygame.draw.rect(screen, col, block, 3)

    title = font.render(label, True, COLOR_TITLE)
    screen.blit(title, title.get_rect(center=(block.centerx, block.centery - 14)))
    dist_str = "CLEAR" if dist is None else f"{dist:.2f} m"
    dist_surf = font_sm.render(dist_str, True, COLOR_HUD)
    screen.blit(dist_surf, dist_surf.get_rect(center=(block.centerx, block.centery + 12)))


def draw_obstacle_zones_panel(screen, rect):
    """Full cross-layout obstacle zones — FRONT / LEFT / RIGHT / BACK fill the area."""
    scale = 1.6 if rect.height > 400 else 1.0
    font = pygame.font.SysFont("monospace", int(16 * scale), bold=True)
    font_sm = pygame.font.SysFont("monospace", int(13 * scale))
    font_banner = pygame.font.SysFont("monospace", int(14 * scale), bold=True)

    front, back, left, right, center = obstacle_zone_rects(rect)
    draw_zone_block(screen, front, "FRONT", nearest_front_m, font, font_sm)
    draw_zone_block(screen, back, "BACK", nearest_back_m, font, font_sm)
    draw_zone_block(screen, left, "LEFT", nearest_left_m, font, font_sm)
    draw_zone_block(screen, right, "RIGHT", nearest_right_m, font, font_sm)

    banner_col = COLOR_GREEN
    if "STOP" in display_banner_text:
        banner_col = COLOR_RED
    elif "CAREFUL" in display_banner_text:
        banner_col = COLOR_ORANGE
    elif "OBSTACLE" in display_banner_text or "SIGN" in display_banner_text or "PERSON" in display_banner_text:
        banner_col = COLOR_YELLOW

    centre_fill = pygame.Surface((center.width, center.height), pygame.SRCALPHA)
    centre_fill.fill((0, 0, 0, 220))
    screen.blit(centre_fill, center.topleft)
    pygame.draw.rect(screen, banner_col, center, 2)

    banner_lines = display_banner_text.split(" | ")
    y = center.y + 8
    for line in banner_lines[:2]:
        banner = font_banner.render(line[:28], True, banner_col)
        screen.blit(banner, banner.get_rect(centerx=center.centerx, y=y))
        y += banner.get_height() + 2

    if camera_banner_text and camera_raw_alert != "CLEAR":
        cam_txt = font_sm.render(camera_banner_text[:24], True, COLOR_CYAN)
        screen.blit(cam_txt, cam_txt.get_rect(centerx=center.centerx, y=center.bottom - 18))

    hint = font_sm.render("LiDAR zones + camera ≤1m", True, (100, 130, 160))
    screen.blit(hint, hint.get_rect(midbottom=(rect.centerx, rect.bottom - 2)))


def draw_zones_overlay(screen):
    """
    Large obstacle-zone overlay (Z key).
    Drawn on top of all 4 panels — LiDAR map, metric distance, and optical flow stay visible behind.
    Press Z again to close.
    """
    overlay_h = HEIGHT - HEADER_H - FOOTER_H
    backdrop = pygame.Surface((WIDTH, overlay_h), pygame.SRCALPHA)
    backdrop.fill((0, 0, 0, 140))
    screen.blit(backdrop, (0, HEADER_H))

    rect = full_zones_content_rect()
    pygame.draw.rect(screen, COLOR_PANEL, rect)
    pygame.draw.rect(screen, COLOR_PANEL_BORDER, rect, 3)

    font_title = pygame.font.SysFont("monospace", 16, bold=True)
    font_hint = pygame.font.SysFont("monospace", 11)
    title = font_title.render("OBSTACLE ZONES — EXPANDED (Z to close)", True, COLOR_TITLE)
    screen.blit(title, (rect.x + 12, rect.y + 6))
    hint = font_hint.render(
        "Map, LiDAR heatmap & camera panels remain visible behind this overlay", True, (120, 160, 200))
    screen.blit(hint, (rect.x + 12, rect.y + 26))

    inner = pygame.Rect(rect.x + 8, rect.y + 42, rect.width - 16, rect.height - 50)
    draw_obstacle_zones_panel(screen, inner)


def draw_metric_distance_panel(screen, rect, latest):
    """LiDAR distance heatmap with light rays and colour legend."""
    cx, cy = rect.centerx, rect.centery
    max_r = min(rect.width, rect.height) // 2 - 16

    # Faint range arcs
    for m in (0.5, 1.0, 2.0, 3.0, 4.0):
        r = int(m * PIXELS_PER_METER * rect.width / 400)
        if r < max_r:
            pygame.draw.circle(screen, (25, 55, 90), (cx, cy), r, 1)

    # Sensor marker
    pygame.draw.circle(screen, COLOR_SENSOR, (cx, cy), 4)

    ray_layer = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    local_cx, local_cy = cx - rect.x, cy - rect.y
    draw_rays = visual_mode != VIS_MODE_MINIMAL
    for x_m, y_m, dist_m, _angle_deg in latest:
        col = distance_to_color(dist_m)
        if visual_mode == VIS_MODE_HIGH_CONTRAST:
            col = tuple(min(255, c + 40) for c in col)
        sx = int(cx + y_m * PIXELS_PER_METER * rect.width / 400)
        sy = int(cy - x_m * PIXELS_PER_METER * rect.width / 400)
        if not rect.collidepoint(sx, sy):
            continue
        if draw_rays:
            local_sx, local_sy = sx - rect.x, sy - rect.y
            pygame.draw.line(ray_layer, (*col, 50), (local_cx, local_cy), (local_sx, local_sy), 1)
        radius = 5 if visual_mode == VIS_MODE_HIGH_CONTRAST else (4 if dist_m <= ALERT_DISTANCE_M else 3)
        pygame.draw.circle(screen, col, (sx, sy), radius)
    if draw_rays:
        screen.blit(ray_layer, rect.topleft)

    # Legend
    legend_x = rect.right - 110
    legend_y = rect.y + 8
    font = pygame.font.SysFont("monospace", 10)
    items = [
        ("danger", COLOR_RED, f"<{VERY_CLOSE_DISTANCE_M}m"),
        ("near", COLOR_YELLOW, f"<{ALERT_DISTANCE_M}m"),
        ("safe", COLOR_GREEN, "2m"),
        ("far", COLOR_BLUE, ">4m"),
    ]
    for i, (name, col, label) in enumerate(items):
        y = legend_y + i * 16
        pygame.draw.rect(screen, col, (legend_x, y, 12, 12))
        screen.blit(font.render(f"{name} {label}", True, COLOR_HUD), (legend_x + 16, y))


def draw_local_map_panel(screen, rect, free_snap, occ_snap, latest, trail):
    """Top-down occupancy map with trajectory trail."""
    cell_px = max(3, int(GRID_RESOLUTION_M * PIXELS_PER_METER))

    # Range rings
    for m in range(1, int(MAX_RANGE_M) + 1):
        r = int(m * PIXELS_PER_METER)
        if r < min(rect.width, rect.height) // 2:
            pygame.draw.circle(screen, COLOR_RING, rect.center, r, 1)

    # Free cells
    layer = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    for (ix, iy), count in free_snap.items():
        if count < FREE_MIN_HITS or occ_snap.get((ix, iy), 0) >= OCCUPIED_MIN_HITS:
            continue
        x_m, y_m = grid_to_world(ix, iy)
        sx, sy = world_to_panel(x_m, y_m, rect)
        if not rect.collidepoint(sx, sy):
            continue
        local = (sx - rect.x, sy - rect.y)
        shade = COLOR_FREE_BRIGHT if count >= 4 else COLOR_FREE
        pygame.draw.rect(layer, (*shade, 70),
                         pygame.Rect(local[0] - cell_px // 2, local[1] - cell_px // 2, cell_px, cell_px))
    screen.blit(layer, rect.topleft)

    # Occupied walls
    for (ix, iy), hits in occ_snap.items():
        if hits < OCCUPIED_MIN_HITS:
            continue
        x_m, y_m = grid_to_world(ix, iy)
        sx, sy = world_to_panel(x_m, y_m, rect)
        if not rect.collidepoint(sx, sy):
            continue
        if hits >= WALL_STRONG_HITS:
            col = (140, 240, 180)
            size = cell_px + 2
        else:
            col = (40, 160, 200)
            size = cell_px
        pygame.draw.rect(screen, col,
                         pygame.Rect(sx - size // 2, sy - size // 2, size, size))

    # Trajectory trail (simulated — needs IMU/encoders for real pose)
    if len(trail) > 1:
        pts = [world_to_panel(x, y, rect) for x, y in trail]
        pygame.draw.lines(screen, COLOR_CYAN, False, pts, 2)
        pygame.draw.circle(screen, COLOR_YELLOW, pts[-1], 4)

    # Current scan
    for x_m, y_m, dist_m, _a in latest:
        sx, sy = world_to_panel(x_m, y_m, rect)
        if rect.collidepoint(sx, sy):
            pygame.draw.circle(screen, distance_to_color(dist_m), (sx, sy), 2)

    # Sensor
    scx, scy = rect.center
    pygame.draw.circle(screen, COLOR_SENSOR, (scx, scy), 5)
    pygame.draw.polygon(screen, COLOR_SENSOR, [
        (scx, scy - 12), (scx - 5, scy + 2), (scx + 5, scy + 2),
    ])

    font = pygame.font.SysFont("monospace", 9)
    screen.blit(font.render("Trail: simulated (no odometry)", True, (100, 130, 160)),
                (rect.x + 6, rect.bottom - 14))


def draw_header(screen, font):
    pygame.draw.rect(screen, COLOR_HEADER, (0, 0, WIDTH, HEADER_H))
    title = font.render("Team Bravo Multi-Sensor Perception Dashboard", True, COLOR_TITLE)
    screen.blit(title, (12, 10))
    status = (f"LiDAR:{'SIM' if SIMULATED_MODE else 'LIVE'}  "
              f"Cam:{camera_status_label()}  Voice:{'ON' if voice_enabled else 'OFF'}  "
              f"Filter:{'ON' if SELF_FILTER_VOICE else 'OFF'}")
    screen.blit(pygame.font.SysFont("monospace", 12).render(status, True, COLOR_HUD),
                (WIDTH - 380, 14))


def draw_footer(screen, font, fps):
    pygame.draw.rect(screen, COLOR_HEADER, (0, HEIGHT - FOOTER_H, WIDTH, FOOTER_H))
    controls = ("Q=Quit  C=Clear  S=Save  V=Voice  T=Test  Z=Zones  B=Filter  "
                "M=Mode  F=Full  Space=Pause  +/-=Zoom  D=Debug")
    screen.blit(font.render(controls, True, (110, 130, 155)), (8, HEIGHT - FOOTER_H + 6))
    screen.blit(font.render(f"FPS:{fps:.0f}  Pkts:{packet_count}  Zoom:{PIXELS_PER_METER}", True, COLOR_HUD),
                (WIDTH - 220, HEIGHT - FOOTER_H + 6))


def draw_debug_overlay(screen, font):
    if not show_debug:
        return
    lines = [
        f"raw:{raw_alert_state}  cand:{candidate_alert_state} ({candidate_count}/{VOICE_MIN_DETECTIONS})",
        f"confirmed:{confirmed_alert_state}  spoken:{last_spoken_alert_state or '-'}",
        f"zones F/L/R/B: {zone_counts.get('front',0)}/{zone_counts.get('left',0)}/"
        f"{zone_counts.get('right',0)}/{zone_counts.get('back',0)}",
        f"camera: {camera_raw_alert} ({camera_candidate_count}/{CAMERA_VOICE_MIN_DETECTIONS})  "
        f"self-filter:{'ON' if SELF_FILTER_VOICE else 'OFF'}",
    ]
    y = HEADER_H + 4
    for line in lines:
        screen.blit(font.render(line, True, (255, 220, 100)), (WIDTH // 2 - 200, y))
        y += 14


# =============================================================================
# SAVE
# =============================================================================

def clear_map():
    global points_xy, free_grid, occupied_grid, latest_scan_points, packet_count
    global trajectory_trail, raw_alert_state, candidate_alert_state, candidate_count
    global confirmed_alert_state, last_spoken_alert_state, clear_streak, zone_counts
    global display_banner_text, nearest_left_m, nearest_front_m, nearest_right_m, nearest_back_m
    with data_lock:
        points_xy = []
        free_grid = {}
        occupied_grid = {}
        latest_scan_points = []
        packet_count = 0
        trajectory_trail = [(0.0, 0.0)]
        raw_alert_state = candidate_alert_state = confirmed_alert_state = "CLEAR"
        candidate_count = 0
        clear_streak = 0
        last_spoken_alert_state = ""
        zone_counts = {}
        display_banner_text = "CLEAR"
        nearest_left_m = nearest_front_m = nearest_right_m = nearest_back_m = None
    stop_current_voice()
    print("Map cleared.")


def save_dashboard(screen):
    with data_lock:
        pts = list(points_xy)
        free_snap = dict(free_grid)
        occ_snap = dict(occupied_grid)

    if pts:
        with open(LIDAR_POINTS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["x_m", "y_m", "distance_m", "angle_deg", "timestamp"])
            for x, y, d, a, ts in pts:
                w.writerow([f"{x:.5f}", f"{y:.5f}", f"{d:.5f}", f"{a:.2f}", f"{ts:.3f}"])

    all_keys = set(free_snap) | set(occ_snap)
    with open(OCCUPANCY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ix", "iy", "x_m", "y_m", "free_hits", "occupied_hits"])
        for key in sorted(all_keys):
            ix, iy = key
            x_m, y_m = grid_to_world(ix, iy)
            w.writerow([ix, iy, f"{x_m:.3f}", f"{y_m:.3f}",
                        free_snap.get(key, 0), occ_snap.get(key, 0)])

    pygame.image.save(screen, DASHBOARD_PNG)
    print(f"Saved {DASHBOARD_PNG}")
    print(f"Saved {LIDAR_POINTS_CSV}")
    print(f"Saved {OCCUPANCY_CSV}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    global running, ser, mapping_paused, voice_enabled, fullscreen
    global show_debug, visual_mode, PIXELS_PER_METER, camera_cap, camera_available
    global zones_fullscreen, SELF_FILTER_VOICE

    print("=" * 72)
    print("Team Bravo Multi-Sensor Perception Dashboard")
    print("=" * 72)
    print("Prototype assistive navigation — NOT full SLAM or autonomous driving.")
    print()

    init_voice_system()
    init_camera()

    connection = None
    if not SIMULATED_MODE:
        connection = open_serial_port(PORT, BAUD, TIMEOUT)
        ser = connection

    lidar_thread = threading.Thread(target=serial_reader_loop, args=(connection,), daemon=True)
    lidar_thread.start()

    if camera_available:
        cam_thread = threading.Thread(target=camera_reader_loop, daemon=True)
        cam_thread.start()

    pygame.init()
    pygame.display.set_caption("Team Bravo Multi-Sensor Dashboard")
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font_footer = pygame.font.SysFont("monospace", 11)
    font_header = pygame.font.SysFont("monospace", 16, bold=True)
    font_debug = pygame.font.SysFont("monospace", 11)

    if voice_enabled and check_tts():
        pygame.time.wait(500)
        test_voice()

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_c:
                        clear_map()
                    elif event.key == pygame.K_s:
                        save_dashboard(screen)
                    elif event.key == pygame.K_v:
                        voice_enabled = not voice_enabled
                        print(f"Voice {'ON' if voice_enabled else 'OFF'}")
                    elif event.key == pygame.K_t:
                        test_voice()
                    elif event.key == pygame.K_SPACE:
                        mapping_paused = not mapping_paused
                        print("PAUSED" if mapping_paused else "RESUMED")
                    elif event.key == pygame.K_f:
                        fullscreen = not fullscreen
                        screen = pygame.display.set_mode(
                            (0, 0), pygame.FULLSCREEN) if fullscreen else \
                            pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                    elif event.key == pygame.K_m:
                        visual_mode = (visual_mode + 1) % 3
                        names = ["STANDARD", "HIGH CONTRAST", "MINIMAL"]
                        print(f"Visual mode: {names[visual_mode]}")
                    elif event.key == pygame.K_d:
                        show_debug = not show_debug
                    elif event.key == pygame.K_z:
                        zones_fullscreen = not zones_fullscreen
                        print(f"Zones overlay {'ON' if zones_fullscreen else 'OFF'} "
                              "(all 4 panels still drawn underneath)")
                    elif event.key == pygame.K_b:
                        SELF_FILTER_VOICE = not SELF_FILTER_VOICE
                        print(f"LiDAR desk self-filter {'ON' if SELF_FILTER_VOICE else 'OFF'} "
                              f"(ignores front voice alerts < {SELF_FILTER_FRONT_M} m)")
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        PIXELS_PER_METER = min(PIXELS_PER_METER_MAX, PIXELS_PER_METER + 10)
                    elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                        PIXELS_PER_METER = max(PIXELS_PER_METER_MIN, PIXELS_PER_METER - 10)

            update_alerts()

            with data_lock:
                free_snap = dict(free_grid)
                occ_snap = dict(occupied_grid)
                latest = list(latest_scan_points)
                trail = list(trajectory_trail)

            screen.fill(COLOR_BG)
            draw_header(screen, font_header)

            # Always draw all 4 panels (optical flow, zones, metric distance, local map)
            for title, col, row in PANEL_TITLES:
                prect = panel_content_rect(col, row)
                inner = draw_panel_frame(screen, prect, title)
                if title == "Optical Flow":
                    draw_optical_flow_panel(screen, inner)
                elif title == "Obstacle Zones":
                    draw_obstacle_zones_panel(screen, inner)
                elif title == "Metric Distance":
                    draw_metric_distance_panel(screen, inner, latest)
                elif title == "Local Map":
                    draw_local_map_panel(screen, inner, free_snap, occ_snap, latest, trail)

            # Z = expanded zones overlay on top (does not hide map / LiDAR panels)
            if zones_fullscreen:
                draw_zones_overlay(screen)

            draw_footer(screen, font_footer, clock.get_fps())
            draw_debug_overlay(screen, font_debug)

            pygame.display.flip()
            clock.tick(FPS_TARGET)

    except KeyboardInterrupt:
        print("\nCtrl+C received.")
    finally:
        running = False
        lidar_thread.join(timeout=2.0)
        if connection and connection.is_open:
            connection.close()
        release_camera()
        stop_current_voice()
        pygame.quit()
        print("Stopped safely.")


if __name__ == "__main__":
    main()
