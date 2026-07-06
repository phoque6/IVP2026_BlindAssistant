"""
Clear 2D Room Mapping and Blind Navigation Alert System Using D6 AA55 LiDAR
===========================================================================

Improved Pygame room mapper with free-space carving, wall outlines, and
blind-navigation voice alerts.

Install:
    sudo apt update
    sudo apt install python3-serial python3-pygame python3-numpy espeak

Run:
    python3 d6_aa55_clear_room_mapper_blind_alert.py

How it works (beginner notes):
------------------------------
A 2D LiDAR spins and measures distance at many angles in one flat plane.
Each reading is angle + distance (polar coordinates). We convert those into
X/Y map coordinates (forward = +X, right = +Y).

A raw dot cloud can look messy. This program builds a clearer map using:
  - Free-space carving: cells along each laser ray are marked FREE
  - Occupied cells: the hit cell at the end of each ray is marked OCCUPIED
  - Hit counts: repeated hits make walls brighter; isolated noise is filtered

This helps obstacle detection and navigation support for a blind assistant.

This is NOT full SLAM:
  No wheel odometry, IMU, robot pose tracking, or loop closure.
  The map is centred on the current sensor position.

Test without LiDAR: set SIMULATED_MODE = True
"""

import csv
import math
import os
import random
import struct
import subprocess
import threading
import time

import pygame
import serial
from serial.tools import list_ports

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
WIDTH = 1280
HEIGHT = 720
PIXELS_PER_METER = 85
FPS_TARGET = 60

# Map cell states
UNKNOWN = 0
FREE = 1
OCCUPIED = 2

# View modes (M key cycles)
VIEW_RAW = 0
VIEW_MAP = 1
VIEW_COMBINED = 2

# ---------------------------------------------------------------------------
# Serial / LiDAR
# ---------------------------------------------------------------------------
PORT = "/dev/ttyUSB0"
BAUD = 230400
TIMEOUT = 0.5
SIMULATED_MODE = False

MIN_RANGE_CM = 8
MAX_RANGE_M = 6.0
MAX_RANGE_CM = int(MAX_RANGE_M * 100)

# ---------------------------------------------------------------------------
# Grid map
# ---------------------------------------------------------------------------
GRID_RESOLUTION_M = 0.05
GRID_SIZE_M = 12.0
OCCUPIED_MIN_HITS = 3          # raised — filters single-hit noise
WALL_STRONG_HITS = 5
FREE_MIN_HITS = 2
MAX_POINTS = 12000
RAY_DRAW_STEP = 6
POLAR_BIN_DEG = 1.0            # median distance per degree bin = smoother walls

# ---------------------------------------------------------------------------
# Blind navigation alerts
# ---------------------------------------------------------------------------
ALERT_DISTANCE_M = 1.0
VERY_CLOSE_DISTANCE_M = 0.45
ENABLE_VOICE_ALERTS = True
VOICE_COOLDOWN_SECONDS = 5.0   # minimum gap between ANY spoken messages
VOICE_REPEAT_SECONDS = 12.0    # re-announce same danger only after this long (if still present)
VOICE_MIN_DETECTIONS = 15      # ~15 consecutive scans before speaking (reduces false alarms)
VERY_CLOSE_VOICE_MIN = 8       # emergency stop — still faster than normal, but not instant chatter
ZONE_CLEAR_SCANS = 4           # need several clear scans before resetting detection streak
MOVING_DISTANCE_M = 0.15       # metres shift to count as moving object
MOVING_MIN_SCANS = 5           # scans needed to confirm movement for early voice

# espeak: speed (words/min, default ~175), amplitude (0-200), gap between words (ms)
ESPEAK_SPEED = 115             # slower = easier to understand
ESPEAK_AMPLITUDE = 200
ESPEAK_WORD_GAP_MS = 12

# ---------------------------------------------------------------------------
# Save filenames
# ---------------------------------------------------------------------------
POINTS_CSV = "clear_room_points.csv"
OCCUPANCY_CSV = "clear_room_occupancy.csv"
MAP_PNG = "clear_room_map.png"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLOR_BG = (6, 10, 22)
COLOR_FREE = (18, 32, 58)
COLOR_FREE_BRIGHT = (28, 48, 82)
COLOR_RING = (35, 90, 150)
COLOR_RING_FAINT = (22, 55, 100)
COLOR_SENSOR = (230, 240, 255)
COLOR_RAY = (40, 90, 140)
COLOR_SCAN_POINT = (120, 255, 240)
COLOR_HUD = (190, 205, 225)
COLOR_PANEL_BG = (12, 18, 35)
COLOR_WARN_CLEAR = (70, 220, 100)
COLOR_WARN_FRONT = (255, 220, 50)
COLOR_WARN_SIDE = (255, 150, 45)
COLOR_WARN_STOP = (255, 50, 50)
COLOR_DIST_OK = (80, 200, 100)
COLOR_DIST_WARN = (255, 200, 60)
COLOR_DIST_DANGER = (255, 70, 70)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
data_lock = threading.Lock()
running = True
mapping_paused = False

points_xy = []              # (x_m, y_m, distance_m, angle_deg, timestamp)
latest_scan_points = []     # same tuple without needing timestamp always
free_grid = {}              # (ix, iy) -> seen_count
occupied_grid = {}          # (ix, iy) -> hit_count
packet_count = 0
latest_angle_deg = 0.0

warning_state = "CLEAR"
warning_direction = ""
nearest_distance_m = None
nearest_left_m = None
nearest_front_m = None
nearest_right_m = None

last_spoken_warning = ""
last_voice_time = 0.0
last_confirmed_warning = ""
clear_streak = 0
current_voice_process = None
espeak_available = None

# Display toggles
show_rays = True
show_grid_map = True
show_help = False
fullscreen = False
view_mode = VIEW_MAP           # MAP view is clearest for room outline
voice_enabled = ENABLE_VOICE_ALERTS

# Voice confirmation tracking (reduces false spoken alerts)
zone_streak = {"FRONT": 0, "LEFT": 0, "RIGHT": 0, "BOTH": 0, "VERY_CLOSE": 0}
zone_position_history = {"FRONT": [], "LEFT": [], "RIGHT": [], "BOTH": []}
voice_confirmed = False        # True when voice is allowed to speak current warning
display_warning_state = "CLEAR"

ser = None

# Simulated room (metres: X forward, Y right)
SIM_WALLS = {"x_min": -2.0, "x_max": 4.0, "y_min": -3.0, "y_max": 3.0}
SIM_OBSTACLES = [
    {"x_min": 1.2, "x_max": 1.9, "y_min": 0.4, "y_max": 1.2},   # table
    {"x_min": 1.9, "x_max": 2.5, "y_min": -1.3, "y_max": -0.5},  # chair
]


# =============================================================================
# PART 1 — LIDAR SERIAL READING
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
        print(f"Connected: {port} @ {baud}")
        return conn
    except serial.SerialException as exc:
        print(f"ERROR: Cannot open '{port}': {exc}")
        for p in list_serial_ports():
            print(f"  - {p}")
        raise SystemExit(1) from exc


def read_packet(connection):
    """Search AA55 header and return one packet, or None on timeout."""
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
    """Return list of (angle_deg, distance_cm)."""
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
    """Polar -> map metres. 0°=forward, 90°=right."""
    distance_m = distance_cm / 100.0
    rad = math.radians(angle_deg)
    x = distance_m * math.cos(rad)
    y = distance_m * math.sin(rad)
    return x, y, distance_m, angle_deg


def world_to_screen(x_m, y_m, sw, sh):
    """Sensor at centre; forward (X) up; right (Y) right."""
    cx, cy = sw // 2, sh // 2
    sx = int(cx + y_m * PIXELS_PER_METER)
    sy = int(cy - x_m * PIXELS_PER_METER)
    return sx, sy


def grid_index(x_m, y_m):
    return round(x_m / GRID_RESOLUTION_M), round(y_m / GRID_RESOLUTION_M)


def grid_to_world(ix, iy):
    return ix * GRID_RESOLUTION_M, iy * GRID_RESOLUTION_M


# =============================================================================
# PART 2 & 3 — MAP BUILDING + FREE SPACE CARVING
# =============================================================================

def bresenham_line_cells(x0, y0, x1, y1):
    """Bresenham line in grid cell coordinates."""
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
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
    """
    Mark cells along ray from sensor (0,0) to hit as FREE.
    Mark final cell as OCCUPIED. Occupied overrides free.
    """
    end_ix, end_iy = grid_index(x_m, y_m)
    line = bresenham_line_cells(0, 0, end_ix, end_iy)

    for i, (ix, iy) in enumerate(line):
        key = (ix, iy)
        if i == len(line) - 1:
            occupied_grid[key] = occupied_grid.get(key, 0) + 1
        else:
            # Do not mark free through confirmed walls
            if occupied_grid.get(key, 0) >= OCCUPIED_MIN_HITS:
                continue
            free_grid[key] = free_grid.get(key, 0) + 1


def smooth_scan_polar(scan_polar):
    """
    Bin scan points by angle and use median distance per bin.
    This removes jitter so walls draw as clean straight segments.
    """
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


def process_scan(scan_polar):
    """Add scan to point cloud and update free/occupied grids."""
    global packet_count, latest_angle_deg, latest_scan_points

    scan_polar = smooth_scan_polar(scan_polar)
    now = time.time()
    batch = []

    with data_lock:
        for angle_deg, distance_cm in scan_polar:
            x_m, y_m, distance_m, angle = polar_to_xy(angle_deg, distance_cm)
            batch.append((x_m, y_m, distance_m, angle, now))
            carve_ray_to_obstacle(x_m, y_m)

        if scan_polar:
            latest_angle_deg = scan_polar[-1][0]

        points_xy.extend(batch)
        if len(points_xy) > MAX_POINTS:
            del points_xy[:-MAX_POINTS]
        latest_scan_points = [(p[0], p[1], p[2], p[3]) for p in batch]
        packet_count += 1


def clear_map():
    global points_xy, free_grid, occupied_grid, latest_scan_points
    global packet_count, warning_state, nearest_left_m, nearest_front_m, nearest_right_m
    global zone_streak, zone_position_history, voice_confirmed, display_warning_state
    global clear_streak, last_confirmed_warning, current_voice_process
    with data_lock:
        points_xy = []
        free_grid = {}
        occupied_grid = {}
        latest_scan_points = []
        packet_count = 0
        warning_state = "CLEAR"
        display_warning_state = "CLEAR"
        nearest_left_m = nearest_front_m = nearest_right_m = None
        for key in zone_streak:
            zone_streak[key] = 0
        for key in zone_position_history:
            zone_position_history[key] = []
        voice_confirmed = False
        clear_streak = 0
        last_confirmed_warning = ""
    stop_current_voice()
    print("Map cleared.")


def cell_state(ix, iy):
    """Return UNKNOWN, FREE, or OCCUPIED for a grid cell."""
    occ = occupied_grid.get((ix, iy), 0)
    if occ >= OCCUPIED_MIN_HITS:
        return OCCUPIED
    free = free_grid.get((ix, iy), 0)
    if free >= FREE_MIN_HITS:
        return FREE
    return UNKNOWN


# =============================================================================
# PART 12 — SIMULATED MODE
# =============================================================================

def sim_ray_distance(angle_deg):
    """Ray-cast against simulated room walls and obstacles."""
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

    best += random.uniform(-0.015, 0.015)
    return max(MIN_RANGE_CM / 100.0, min(best, MAX_RANGE_M)) * 100.0


def generate_simulated_scan():
    base = random.uniform(0, 360)
    return [
        ((base + i * 3.0) % 360.0, sim_ray_distance((base + i * 3.0) % 360.0))
        for i in range(28)
    ]


# =============================================================================
# PART 7 — BLIND NAVIGATION OBSTACLE ALERT
# =============================================================================

def detect_obstacles_for_blind_user(latest_points):
    """
    Detect obstacles within ALERT_DISTANCE_M for blind navigation.
    Returns (warning_state, nearest_distance, direction).
    """
    front_near = None
    left_near = None
    right_near = None
    very_close = False
    overall_near = None

    for x_m, y_m, distance_m, _angle in latest_points:
        if overall_near is None or distance_m < overall_near:
            overall_near = distance_m

        if distance_m <= VERY_CLOSE_DISTANCE_M:
            very_close = True

        if x_m > 0 and abs(y_m) <= 0.40 and distance_m <= ALERT_DISTANCE_M:
            if front_near is None or distance_m < front_near:
                front_near = distance_m

        if y_m < -0.30 and abs(x_m) <= 1.0 and distance_m <= ALERT_DISTANCE_M:
            if left_near is None or distance_m < left_near:
                left_near = distance_m

        if y_m > 0.30 and abs(x_m) <= 1.0 and distance_m <= ALERT_DISTANCE_M:
            if right_near is None or distance_m < right_near:
                right_near = distance_m

    if very_close:
        return "STOP! OBJECT VERY CLOSE", overall_near, "FRONT"

    if left_near is not None and right_near is not None:
        return "OBSTACLES LEFT AND RIGHT WITHIN 1 METRE", min(left_near, right_near), "BOTH"

    if front_near is not None:
        return "OBSTACLE AHEAD WITHIN 1 METRE", front_near, "FRONT"
    if left_near is not None:
        return "OBSTACLE LEFT WITHIN 1 METRE", left_near, "LEFT"
    if right_near is not None:
        return "OBSTACLE RIGHT WITHIN 1 METRE", right_near, "RIGHT"

    return "CLEAR", overall_near, ""


def compute_direction_distances(latest_points):
    """Nearest obstacle in left / front / right sectors."""
    left_d = front_d = right_d = None
    for x_m, y_m, distance_m, _a in latest_points:
        if x_m > 0 and abs(y_m) <= 0.45:
            if front_d is None or distance_m < front_d:
                front_d = distance_m
        if y_m < -0.25:
            if left_d is None or distance_m < left_d:
                left_d = distance_m
        if y_m > 0.25:
            if right_d is None or distance_m < right_d:
                right_d = distance_m
    return left_d, front_d, right_d


def state_to_zone_key(state):
    if "STOP" in state or "VERY CLOSE" in state:
        return "VERY_CLOSE"
    if "LEFT AND RIGHT" in state:
        return "BOTH"
    if "LEFT" in state:
        return "LEFT"
    if "RIGHT" in state:
        return "RIGHT"
    if "AHEAD" in state:
        return "FRONT"
    return None


def points_in_zone(zone, points):
    """Return points inside a warning zone."""
    matched = []
    for x_m, y_m, distance_m, _a in points:
        if zone == "VERY_CLOSE" and distance_m <= VERY_CLOSE_DISTANCE_M:
            matched.append((x_m, y_m, distance_m))
        elif zone == "FRONT" and x_m > 0 and abs(y_m) <= 0.40 and distance_m <= ALERT_DISTANCE_M:
            matched.append((x_m, y_m, distance_m))
        elif zone == "LEFT" and y_m < -0.30 and abs(x_m) <= 1.0 and distance_m <= ALERT_DISTANCE_M:
            matched.append((x_m, y_m, distance_m))
        elif zone == "RIGHT" and y_m > 0.30 and abs(x_m) <= 1.0 and distance_m <= ALERT_DISTANCE_M:
            matched.append((x_m, y_m, distance_m))
        elif zone == "BOTH":
            if (y_m < -0.30 or y_m > 0.30) and abs(x_m) <= 1.0 and distance_m <= ALERT_DISTANCE_M:
                matched.append((x_m, y_m, distance_m))
    return matched


def detect_moving_in_zone(zone, points):
    """
    Detect if obstacle position is shifting between scans (moving object).
    Moving objects may trigger voice sooner than VOICE_MIN_DETECTIONS.
    """
    matched = points_in_zone(zone, points)
    if len(matched) < 2:
        return False

    cx = sum(p[0] for p in matched) / len(matched)
    cy = sum(p[1] for p in matched) / len(matched)
    now = time.time()

    history = zone_position_history.setdefault(zone, [])
    history.append((now, cx, cy))
    # Keep last 2 seconds of samples
    zone_position_history[zone] = [(t, x, y) for t, x, y in history if now - t < 2.0]

    if len(zone_position_history[zone]) < MOVING_MIN_SCANS:
        return False

    old_t, old_x, old_y = zone_position_history[zone][0]
    shift = math.hypot(cx - old_x, cy - old_y)
    return shift >= MOVING_DISTANCE_M


def update_voice_confirmation(raw_state, latest_points):
    """
    Voice speaks only when:
      - obstacle detected in zone >= VOICE_MIN_DETECTIONS consecutive scans, OR
      - a moving object is detected (with at least 5 scans in zone).
    CLEAR requires ZONE_CLEAR_SCANS consecutive clear scans to avoid flicker.
    """
    global voice_confirmed, display_warning_state, zone_streak, clear_streak

    if raw_state == "CLEAR":
        clear_streak += 1
        if clear_streak >= ZONE_CLEAR_SCANS:
            for key in zone_streak:
                zone_streak[key] = 0
            voice_confirmed = True
            display_warning_state = "CLEAR"
        else:
            voice_confirmed = False
            display_warning_state = f"CLEARING... ({clear_streak}/{ZONE_CLEAR_SCANS})"
        return

    clear_streak = 0
    zone = state_to_zone_key(raw_state)
    if zone is None:
        voice_confirmed = False
        display_warning_state = raw_state
        return

    for key in zone_streak:
        if key != zone:
            zone_streak[key] = 0
    zone_streak[zone] = zone_streak.get(zone, 0) + 1

    moving = detect_moving_in_zone(zone, latest_points)
    needed = VERY_CLOSE_VOICE_MIN if zone == "VERY_CLOSE" else VOICE_MIN_DETECTIONS
    confirmed = zone_streak[zone] >= needed or (moving and zone_streak[zone] >= 5)
    voice_confirmed = confirmed

    if confirmed:
        display_warning_state = raw_state
    else:
        count = zone_streak[zone]
        display_warning_state = f"CHECKING... {raw_state} ({count}/{needed})"


def update_alerts():
    """Update warning state, voice confirmation, and direction distances."""
    global warning_state, warning_direction, nearest_distance_m
    global nearest_left_m, nearest_front_m, nearest_right_m

    with data_lock:
        latest = list(latest_scan_points)

    state, nearest, direction = detect_obstacles_for_blind_user(latest)
    left_d, front_d, right_d = compute_direction_distances(latest)

    warning_state = state
    warning_direction = direction
    nearest_distance_m = nearest
    nearest_left_m = left_d
    nearest_front_m = front_d
    nearest_right_m = right_d

    update_voice_confirmation(state, latest)


# =============================================================================
# PART 8 — AUDIO FEEDBACK
# =============================================================================

def check_espeak():
    global espeak_available
    if espeak_available is not None:
        return espeak_available
    try:
        subprocess.run(["which", "espeak"], capture_output=True, check=True)
        espeak_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        espeak_available = False
        print("NOTE: espeak not found. Install: sudo apt install espeak")
    return espeak_available


def warning_to_speech(state):
    """Short, clear phrases — easier to hear at slow speed."""
    mapping = {
        "CLEAR": "Path clear",
        "OBSTACLE AHEAD WITHIN 1 METRE": "Obstacle ahead",
        "OBSTACLE LEFT WITHIN 1 METRE": "Obstacle on left",
        "OBSTACLE RIGHT WITHIN 1 METRE": "Obstacle on right",
        "OBSTACLES LEFT AND RIGHT WITHIN 1 METRE": "Obstacles left and right",
        "STOP! OBJECT VERY CLOSE": "Stop. Object very close",
    }
    return mapping.get(state, state)


def stop_current_voice():
    """Stop any speech still playing so messages do not overlap."""
    global current_voice_process
    if current_voice_process is not None and current_voice_process.poll() is None:
        try:
            current_voice_process.terminate()
            current_voice_process.wait(timeout=0.5)
        except (OSError, subprocess.TimeoutExpired):
            pass
    current_voice_process = None


def speak_warning(state, allow_voice):
    """
    Speak one clear alert with slow espeak and a minimum 5 second gap.
    Same warning will not repeat until VOICE_REPEAT_SECONDS unless state changes.
    """
    global last_spoken_warning, last_voice_time, last_confirmed_warning, current_voice_process

    if not voice_enabled:
        return
    if not allow_voice and state != "CLEAR":
        return

    now = time.time()
    elapsed = now - last_voice_time

    # Always wait at least VOICE_COOLDOWN_SECONDS between any two messages
    if elapsed < VOICE_COOLDOWN_SECONDS:
        return

    # Same message: only repeat after VOICE_REPEAT_SECONDS (avoid fast loops)
    if state == last_spoken_warning and elapsed < VOICE_REPEAT_SECONDS:
        return

    # Do not speak CLEAR unless we previously announced a real danger
    if state == "CLEAR" and last_confirmed_warning == "":
        return

    stop_current_voice()

    last_spoken_warning = state
    last_voice_time = now
    if state != "CLEAR":
        last_confirmed_warning = state
    else:
        last_confirmed_warning = ""

    text = warning_to_speech(state)
    print(f"VOICE [{ESPEAK_SPEED} wpm]: {text}")

    if check_espeak():
        try:
            current_voice_process = subprocess.Popen(
                [
                    "espeak",
                    "-s", str(ESPEAK_SPEED),
                    "-a", str(ESPEAK_AMPLITUDE),
                    "-g", str(ESPEAK_WORD_GAP_MS),
                    text,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            print("\a", end="", flush=True)
    else:
        print("\a", end="", flush=True)


# =============================================================================
# PART 13 — BACKGROUND READER THREAD
# =============================================================================

def serial_reader_loop(connection):
    global running, latest_angle_deg
    print(f"Reader started ({'SIM' if SIMULATED_MODE else 'SERIAL'}).")
    sweep = 0.0

    while running:
        if mapping_paused:
            time.sleep(0.05)
            continue

        if SIMULATED_MODE:
            scan = generate_simulated_scan()
            sweep = (sweep + 5.0) % 360.0
            latest_angle_deg = sweep
            time.sleep(0.035)
        else:
            packet = read_packet(connection)
            scan = parse_packet(packet)

        if scan:
            process_scan(scan)

        time.sleep(0.001)

    print("Reader stopped.")


# =============================================================================
# PART 5 & 6 — PYGAME DRAWING
# =============================================================================

def occupied_color(hits):
    """Auto-contrast colour from hit count."""
    if hits >= WALL_STRONG_HITS:
        t = min((hits - WALL_STRONG_HITS) / 8.0, 1.0)
        return (int(100 + 155 * t), int(240 + 15 * t), int(120 + 135 * t))
    if hits >= OCCUPIED_MIN_HITS:
        t = (hits - OCCUPIED_MIN_HITS) / max(WALL_STRONG_HITS - OCCUPIED_MIN_HITS, 1)
        return (int(40 + 60 * t), int(160 + 80 * t), int(180 + 40 * t))
    return (30, 100, 130)


def warning_color(state):
    if "CHECKING" in state or "CLEARING" in state:
        return (160, 180, 200)
    if "STOP" in state:
        return COLOR_WARN_STOP
    if "AHEAD" in state:
        return COLOR_WARN_FRONT
    if "LEFT" in state or "RIGHT" in state:
        return COLOR_WARN_SIDE
    return COLOR_WARN_CLEAR


def dist_panel_color(d):
    if d is None:
        return COLOR_DIST_OK
    if d <= VERY_CLOSE_DISTANCE_M:
        return COLOR_DIST_DANGER
    if d <= ALERT_DISTANCE_M:
        return COLOR_DIST_WARN
    return COLOR_DIST_OK


def dist_panel_text(d):
    if d is None:
        return "clear"
    return f"{d:.1f} m"


def draw_range_rings(screen, sw, sh, font):
    cx, cy = sw // 2, sh // 2
    for m in range(1, int(MAX_RANGE_M) + 1):
        r = int(m * PIXELS_PER_METER)
        col = COLOR_RING if m % 2 == 0 else COLOR_RING_FAINT
        pygame.draw.circle(screen, col, (cx, cy), r, 1)
        lbl = font.render(f"{m}m", True, col)
        screen.blit(lbl, (cx + 5, cy - r - 14))


def draw_free_cells(screen, sw, sh, grid_free, grid_occ):
    """Draw explored free floor — larger, softer fill for readable room interior."""
    cell_px = max(3, int(GRID_RESOLUTION_M * PIXELS_PER_METER) + 1)
    layer = pygame.Surface((sw, sh), pygame.SRCALPHA)
    for (ix, iy), count in grid_free.items():
        if count < FREE_MIN_HITS:
            continue
        if grid_occ.get((ix, iy), 0) >= OCCUPIED_MIN_HITS:
            continue
        x_m, y_m = grid_to_world(ix, iy)
        sx, sy = world_to_screen(x_m, y_m, sw, sh)
        alpha = min(90, 35 + count * 8)
        shade = COLOR_FREE_BRIGHT if count >= 4 else COLOR_FREE
        rect = pygame.Rect(sx - cell_px // 2, sy - cell_px // 2, cell_px, cell_px)
        pygame.draw.rect(layer, (*shade, alpha), rect)
    screen.blit(layer, (0, 0))


def draw_occupied_cells(screen, sw, sh, grid_occ):
    """Draw walls as thick bright blocks with neighbour fill."""
    cell_px = max(4, int(GRID_RESOLUTION_M * PIXELS_PER_METER) + 2)
    wall_layer = pygame.Surface((sw, sh), pygame.SRCALPHA)

    strong_cells = {k for k, v in grid_occ.items() if v >= WALL_STRONG_HITS}
    drawn = set()

    for (ix, iy), hits in grid_occ.items():
        if hits < OCCUPIED_MIN_HITS:
            continue
        colour = occupied_color(hits)
        x_m, y_m = grid_to_world(ix, iy)
        sx, sy = world_to_screen(x_m, y_m, sw, sh)

        if hits >= WALL_STRONG_HITS:
            size = cell_px + 3
        else:
            size = cell_px

        rect = pygame.Rect(sx - size // 2, sy - size // 2, size, size)
        pygame.draw.rect(wall_layer, (*colour, 220), rect)
        drawn.add((ix, iy))

        # Fill gaps between neighbouring wall cells
        if hits >= WALL_STRONG_HITS:
            for dix, diy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
                nk = (ix + dix, iy + diy)
                if nk in strong_cells or grid_occ.get(nk, 0) >= OCCUPIED_MIN_HITS:
                    nx, ny = grid_to_world(nk[0], nk[1])
                    nsx, nsy = world_to_screen(nx, ny, sw, sh)
                    pygame.draw.line(wall_layer, (*colour, 180), (sx, sy), (nsx, nsy), 3)

    screen.blit(wall_layer, (0, 0))

    # Bright white outline on confirmed walls
    for (ix, iy) in strong_cells:
        x_m, y_m = grid_to_world(ix, iy)
        sx, sy = world_to_screen(x_m, y_m, sw, sh)
        size = cell_px + 3
        rect = pygame.Rect(sx - size // 2, sy - size // 2, size, size)
        pygame.draw.rect(screen, (220, 255, 230), rect, 1)


def draw_wall_outline_ring(screen, sw, sh, grid_occ):
    """
    Connect strong wall cells into a visible room outline polygon.
    Makes the room shape easier to understand at a glance.
    """
    cell_px = max(4, int(GRID_RESOLUTION_M * PIXELS_PER_METER) + 2)
    strong = [(ix, iy) for (ix, iy), h in grid_occ.items() if h >= WALL_STRONG_HITS]
    if len(strong) < 8:
        return

    # Convert to screen points and draw segments between close neighbours
    screen_pts = []
    for ix, iy in strong:
        x_m, y_m = grid_to_world(ix, iy)
        screen_pts.append(world_to_screen(x_m, y_m, sw, sh))

    outline_layer = pygame.Surface((sw, sh), pygame.SRCALPHA)
    for i, p1 in enumerate(screen_pts):
        for p2 in screen_pts[i + 1:]:
            dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            if 2 < dist < cell_px * 3.5:
                pygame.draw.line(outline_layer, (180, 255, 200, 140), p1, p2, 2)
    screen.blit(outline_layer, (0, 0))


def draw_scan_rays(screen, sw, sh, latest):
    cx, cy = sw // 2, sh // 2
    for i, (x_m, y_m, dist_m, _a) in enumerate(latest):
        if i % RAY_DRAW_STEP != 0:
            continue
        ex, ey = world_to_screen(x_m, y_m, sw, sh)
        col = COLOR_RAY
        if dist_m <= ALERT_DISTANCE_M:
            col = (120, 60, 40) if dist_m > VERY_CLOSE_DISTANCE_M else (180, 40, 40)
        pygame.draw.line(screen, col, (cx, cy), (ex, ey), 1)


def draw_latest_points(screen, sw, sh, latest):
    """Highlight only the current scan — small markers, not a noisy cloud."""
    for x_m, y_m, dist_m, _a in latest:
        sx, sy = world_to_screen(x_m, y_m, sw, sh)
        if dist_m <= VERY_CLOSE_DISTANCE_M:
            col = (255, 80, 80)
            radius = 4
        elif dist_m <= ALERT_DISTANCE_M:
            col = (255, 200, 80)
            radius = 3
        else:
            col = (80, 200, 220)
            radius = 2
        pygame.draw.circle(screen, col, (sx, sy), radius, 1)


def draw_raw_points(screen, sw, sh, points):
    for x_m, y_m, dist_m, _a, _ts in points[-800:]:
        sx, sy = world_to_screen(x_m, y_m, sw, sh)
        pygame.draw.circle(screen, (50, 120, 160), (sx, sy), 1)


def draw_sensor(screen, sw, sh):
    cx, cy = sw // 2, sh // 2
    pygame.draw.polygon(screen, COLOR_SENSOR, [
        (cx, cy - 18), (cx - 7, cy + 4), (cx + 7, cy + 4),
    ])
    pygame.draw.circle(screen, COLOR_SENSOR, (cx, cy), 6)
    pygame.draw.circle(screen, (100, 150, 220), (cx, cy), 6, 1)


def draw_warning_zones(screen, sw, sh, state):
    """Transparent front/left/right alert zones coloured by danger."""
    surf = pygame.Surface((sw, sh), pygame.SRCALPHA)

    def zone_rect(x0, y0, w, h, base_col, active):
        sx0, sy0 = world_to_screen(x0 + w, y0, sw, sh)
        sx1, sy1 = world_to_screen(x0, y0 + h, sw, sh)
        rect = pygame.Rect(min(sx0, sx1), min(sy0, sy1), abs(sx1 - sx0), abs(sy1 - sy0))
        alpha = 70 if active else 25
        pygame.draw.rect(surf, (*base_col, alpha), rect)
        pygame.draw.rect(surf, (*base_col, 120 if active else 40), rect, 1)

    front_on = "AHEAD" in state or "STOP" in state
    left_on = "LEFT" in state
    right_on = "RIGHT" in state

    zone_rect(0, -0.40, ALERT_DISTANCE_M, 0.80, (255, 220, 50), front_on)
    zone_rect(-1.0, -3.0, 1.0, 2.7, (255, 150, 50), left_on)
    zone_rect(-1.0, 0.30, 1.0, 2.7, (255, 150, 50), right_on)

    # Front cone up to 1 m
    cx, cy = sw // 2, sh // 2
    tip_x, tip_y = world_to_screen(ALERT_DISTANCE_M, 0, sw, sh)
    left_x, left_y = world_to_screen(0, -0.4, sw, sh)
    right_x, right_y = world_to_screen(0, 0.4, sw, sh)
    cone_col = (255, 60, 60, 50) if "STOP" in state else (255, 220, 50, 35)
    pygame.draw.polygon(surf, cone_col, [(cx, cy), (left_x, left_y), (tip_x, tip_y), (right_x, right_y)])

    screen.blit(surf, (0, 0))


def draw_distance_panel(screen, font, sw):
    """Side panel: nearest distance left / front / right."""
    pw, ph = 200, 110
    px, py = sw - pw - 14, 60
    panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
    panel.fill((*COLOR_PANEL_BG, 210))
    screen.blit(panel, (px, py))
    pygame.draw.rect(screen, (60, 90, 130), (px, py, pw, ph), 1)

    rows = [
        ("LEFT", nearest_left_m),
        ("FRONT", nearest_front_m),
        ("RIGHT", nearest_right_m),
    ]
    y = py + 10
    title = font.render("DISTANCES", True, COLOR_HUD)
    screen.blit(title, (px + 10, y))
    y += 22
    for label, dist in rows:
        col = dist_panel_color(dist)
        txt = font.render(f"{label}: {dist_panel_text(dist)}", True, col)
        screen.blit(txt, (px + 10, y))
        y += 24


def draw_hud(screen, fonts, sw, sh, fps):
    small, med, large = fonts
    warn_col = warning_color(display_warning_state)

    # Large warning banner top centre
    banner = large.render(display_warning_state, True, warn_col)
    bx = (sw - banner.get_width()) // 2
    screen.blit(banner, (bx, 8))

    with data_lock:
        n_pts = len(points_xy)
        n_pkt = packet_count
        n_occ = sum(1 for h in occupied_grid.values() if h >= OCCUPIED_MIN_HITS)
        n_free = sum(1 for k, v in free_grid.items()
                      if v >= FREE_MIN_HITS and occupied_grid.get(k, 0) < OCCUPIED_MIN_HITS)

    mode_names = {VIEW_RAW: "RAW", VIEW_MAP: "MAP", VIEW_COMBINED: "COMBINED"}
    lines = [
        "D6 AA55 Clear Room Mapper",
        f"{PORT if not SIMULATED_MODE else 'SIM'} @ {BAUD}  |  {mode_names[view_mode]}",
        f"Points:{n_pts}  Free:{n_free}  Walls:{n_occ}  Pkts:{n_pkt}  FPS:{fps:.0f}",
        f"Voice:{'ON' if voice_enabled else 'OFF'}  "
        f"Confirm:{'YES' if voice_confirmed else 'NO'}  "
        f"Rays:{'ON' if show_rays else 'OFF'}  "
        f"Grid:{'ON' if show_grid_map else 'OFF'}  {'PAUSED' if mapping_paused else 'LIVE'}",
    ]
    y = 42
    for i, line in enumerate(lines):
        fnt = med if i == 0 else small
        screen.blit(fnt.render(line, True, COLOR_HUD), (12, y))
        y += fnt.get_height() + 2

    controls = "Q=Quit C=Clear S=Save Space=Pause R=Rays G=Grid V=Voice M=View F=Full H=Help"
    screen.blit(small.render(controls, True, (110, 130, 155)), (12, sh - 22))

    draw_distance_panel(screen, small, sw)

    if show_help:
        draw_help_panel(screen, small, sw, sh)


def draw_help_panel(screen, font, sw, sh):
    pw, ph = 420, 280
    px, py = (sw - pw) // 2, (sh - ph) // 2
    panel = pygame.Surface((pw, ph))
    panel.fill((15, 22, 45))
    pygame.draw.rect(panel, (80, 120, 180), panel.get_rect(), 2)
    help_lines = [
        "HELP — Clear Room Mapper",
        "",
        "Map builds from laser rays:",
        "  FREE cells = explored floor",
        "  OCCUPIED cells = walls/objects",
        "",
        "Alerts within 1 metre:",
        "  Front / Left / Right zones",
        "",
        "M = cycle RAW / MAP / COMBINED view",
        "Voice: 5 s gap, slow speech, 15 detections",
        "V = toggle voice (espeak)",
        "Not full SLAM — no pose tracking",
    ]
    y = 12
    for line in help_lines:
        screen.blit(font.render(line, True, COLOR_HUD), (px + 14, py + y))
        y += 20
    screen.blit(panel, (px, py))


# =============================================================================
# PART 11 — SAVE MAP
# =============================================================================

def save_map(screen):
    with data_lock:
        pts = list(points_xy)
        free_snap = dict(free_grid)
        occ_snap = dict(occupied_grid)

    if not pts and not occ_snap:
        print("Nothing to save.")
        return

    with open(POINTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x_m", "y_m", "distance_m", "angle_deg", "timestamp"])
        for x, y, d, a, ts in pts:
            w.writerow([f"{x:.5f}", f"{y:.5f}", f"{d:.5f}", f"{a:.2f}", f"{ts:.3f}"])

    all_keys = set(free_snap) | set(occ_snap)
    with open(OCCUPANCY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ix", "iy", "x_m", "y_m", "free_hits", "occupied_hits", "state"])
        for key in sorted(all_keys):
            ix, iy = key
            fh = free_snap.get(key, 0)
            oh = occ_snap.get(key, 0)
            x_m, y_m = grid_to_world(ix, iy)
            st = {UNKNOWN: "UNKNOWN", FREE: "FREE", OCCUPIED: "OCCUPIED"}[
                OCCUPIED if oh >= OCCUPIED_MIN_HITS else (FREE if fh >= FREE_MIN_HITS else UNKNOWN)
            ]
            w.writerow([ix, iy, f"{x_m:.3f}", f"{y_m:.3f}", fh, oh, st])

    pygame.image.save(screen, MAP_PNG)
    print(f"Saved {POINTS_CSV}")
    print(f"Saved {OCCUPANCY_CSV}")
    print(f"Saved {MAP_PNG}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    global running, ser, mapping_paused, show_rays, show_grid_map
    global show_help, fullscreen, view_mode, voice_enabled, warning_state

    print("=" * 72)
    print("Clear 2D Room Mapper + Blind Navigation Alerts")
    print("=" * 72)
    print("NOT full SLAM — no odometry, IMU, or pose tracking.")
    print()

    connection = None
    if not SIMULATED_MODE:
        connection = open_serial_port(PORT, BAUD, TIMEOUT)
        ser = connection

    reader = threading.Thread(target=serial_reader_loop, args=(connection,), daemon=True)
    reader.start()

    pygame.init()
    pygame.display.set_caption("D6 AA55 Clear Room Mapper")
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font_sm = pygame.font.SysFont("monospace", 13)
    font_md = pygame.font.SysFont("monospace", 15, bold=True)
    font_lg = pygame.font.SysFont("monospace", 22, bold=True)
    font_ring = pygame.font.SysFont("monospace", 11)
    fonts = (font_sm, font_md, font_lg)

    prev_voice_state = None

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_c:
                        clear_map()
                    elif event.key == pygame.K_s:
                        save_map(screen)
                    elif event.key == pygame.K_SPACE:
                        mapping_paused = not mapping_paused
                        print("PAUSED" if mapping_paused else "RESUMED")
                    elif event.key == pygame.K_r:
                        show_rays = not show_rays
                    elif event.key == pygame.K_g:
                        show_grid_map = not show_grid_map
                    elif event.key == pygame.K_f:
                        fullscreen = not fullscreen
                        screen = pygame.display.set_mode(
                            (0, 0), pygame.FULLSCREEN) if fullscreen else \
                            pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                    elif event.key == pygame.K_v:
                        voice_enabled = not voice_enabled
                        print(f"Voice {'ON' if voice_enabled else 'OFF'}")
                    elif event.key == pygame.K_m:
                        view_mode = (view_mode + 1) % 3
                        names = ["RAW", "MAP", "COMBINED"]
                        print(f"View mode: {names[view_mode]}")
                    elif event.key == pygame.K_h:
                        show_help = not show_help

            sw, sh = screen.get_size()
            update_alerts()

            # Voice: only on confirmed state change, with 5 s minimum gap
            if voice_confirmed or (warning_state == "CLEAR" and clear_streak >= ZONE_CLEAR_SCANS):
                if warning_state != prev_voice_state:
                    speak_warning(
                        warning_state,
                        voice_confirmed or warning_state == "CLEAR",
                    )
                    prev_voice_state = warning_state

            with data_lock:
                free_snap = dict(free_grid)
                occ_snap = dict(occupied_grid)
                latest = list(latest_scan_points)
                pts = list(points_xy)

            # --- draw layers ---
            screen.fill(COLOR_BG)
            draw_range_rings(screen, sw, sh, font_ring)

            if view_mode == VIEW_RAW:
                draw_raw_points(screen, sw, sh, pts)

            if show_grid_map and view_mode in (VIEW_MAP, VIEW_COMBINED):
                draw_free_cells(screen, sw, sh, free_snap, occ_snap)
                draw_occupied_cells(screen, sw, sh, occ_snap)
                draw_wall_outline_ring(screen, sw, sh, occ_snap)

            if show_rays and latest and view_mode != VIEW_MAP:
                draw_scan_rays(screen, sw, sh, latest)

            if view_mode == VIEW_COMBINED:
                draw_latest_points(screen, sw, sh, latest)

            draw_warning_zones(screen, sw, sh, display_warning_state)
            draw_sensor(screen, sw, sh)

            # Direction labels
            fx, fy = world_to_screen(0.6, 0, sw, sh)
            screen.blit(font_sm.render("FRONT", True, (180, 200, 220)), (fx - 20, fy - 20))

            draw_hud(screen, fonts, sw, sh, clock.get_fps())

            pygame.display.flip()
            clock.tick(FPS_TARGET)

    except KeyboardInterrupt:
        print("\nCtrl+C received.")
    finally:
        running = False
        reader.join(timeout=2.0)
        if connection and connection.is_open:
            connection.close()
            print("Serial closed.")
        stop_current_voice()
        pygame.quit()
        print("Stopped safely.")


if __name__ == "__main__":
    main()
