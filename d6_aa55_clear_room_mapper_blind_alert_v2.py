"""
Clear 2D Room Mapping and Blind Navigation Alert System Using D6 AA55 LiDAR
===========================================================================

Improved Pygame room mapper with free-space carving, wall outlines, and
blind-navigation voice alerts.

Install:
    sudo apt update
    sudo apt install python3-serial python3-pygame python3-numpy espeak-ng

If no sound, test in terminal:
    espeak-ng "Voice test"
    amixer set Master 80%

Run:
    python3 d6_aa55_clear_room_mapper_blind_alert_v2.py

Safety note:
    Prototype navigation aid only — not the only safety device for a blind person.
    Test voice alerts in a safe controlled environment.
    LiDAR may miss glass, shiny surfaces, low objects, or soft materials.
    A real mobility system should combine LiDAR with ultrasonic sensors,
    camera, IMU, and extensive human testing.
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
import shutil
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
PIXELS_PER_METER = 95
PIXELS_PER_METER_MIN = 50
PIXELS_PER_METER_MAX = 160
FPS_TARGET = 60
ACCESSIBILITY_MODE = True

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
OCCUPIED_MIN_HITS = 3
WALL_STRONG_HITS = 5
FREE_MIN_HITS = 2
MIN_WALL_COMPONENT_SIZE = 4
MAX_POINTS = 8000
RAY_DRAW_STEP = 8
POLAR_BIN_DEG = 1.0
ZONE_MIN_POINTS = 3

# ---------------------------------------------------------------------------
# Blind navigation alerts (v2 tuned settings)
# ---------------------------------------------------------------------------
ALERT_DISTANCE_M = 1.0
CAUTION_DISTANCE_M = 1.2
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
COLOR_WARN_STRONG = (255, 160, 50)
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
nearest_back_m = None

raw_alert_state = "CLEAR"
current_alert_state = "CLEAR"
candidate_alert_state = "CLEAR"
candidate_count = 0
confirmed_alert_state = "CLEAR"
last_spoken_alert_state = ""
last_clear_voice_time = 0.0
clear_streak = 0
zone_counts = {}
display_warning_state = "CLEAR"
display_banner_text = "CLEAR"

current_voice_process = None
tts_executable = None
tts_checked = False

show_rays = False
show_debug = False
show_grid_map = True
show_help = False
fullscreen = False
view_mode = VIEW_MAP           # MAP view is clearest for room outline
voice_enabled = ENABLE_VOICE_ALERTS
last_voice_time = 0.0

ser = None

# Simulated room + moving test obstacles
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
    global packet_count, nearest_left_m, nearest_front_m, nearest_right_m, nearest_back_m
    global display_warning_state, display_banner_text, clear_streak
    global raw_alert_state, candidate_alert_state, candidate_count, confirmed_alert_state
    global current_alert_state, last_spoken_alert_state, zone_counts
    with data_lock:
        points_xy = []
        free_grid = {}
        occupied_grid = {}
        latest_scan_points = []
        packet_count = 0
        display_warning_state = "CLEAR"
        display_banner_text = "CLEAR"
        nearest_left_m = nearest_front_m = nearest_right_m = nearest_back_m = None
        raw_alert_state = candidate_alert_state = confirmed_alert_state = current_alert_state = "CLEAR"
        candidate_count = 0
        clear_streak = 0
        last_spoken_alert_state = ""
        zone_counts = {}
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

def sim_moving_obstacle_distance(angle_deg):
    """Extra simulated obstacles: mover in front, left, right."""
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
    if best is None:
        return None
    return best * 100.0


def update_sim_movers():
    global sim_mover_x, sim_mover_dx
    sim_mover_x += sim_mover_dx
    if sim_mover_x > 1.15 or sim_mover_x < 0.45:
        sim_mover_dx *= -1


def sim_ray_distance(angle_deg):
    """Ray-cast walls + static boxes + moving test obstacles."""
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


# =============================================================================
# BLIND NAVIGATION — ZONE DETECTION + PRIORITY ALERTS
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


def detect_obstacles_for_blind_user(latest_points):
    """
    Cluster-based zone detection. Returns:
    (alert_state, nearest_distance, direction, zone_counts)
    """
    zc = {
        "front": 0, "back": 0, "left": 0, "right": 0,
        "vc_front": 0, "vc_back": 0, "vc_left": 0, "vc_right": 0,
    }

    for x_m, y_m, distance_m, _a in latest_points:
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

    overall = min([d for d in (nf, nb, nl, nr) if d is not None], default=None)

    # Priority 1: very close
    if zc["vc_front"] >= ZONE_MIN_POINTS:
        return "VERY_CLOSE_FRONT", nf, "FRONT", zc
    if zc["vc_back"] >= ZONE_MIN_POINTS:
        return "VERY_CLOSE_BACK", nb, "BACK", zc
    if zc["vc_left"] >= ZONE_MIN_POINTS:
        return "VERY_CLOSE_LEFT", nl, "LEFT", zc
    if zc["vc_right"] >= ZONE_MIN_POINTS:
        return "VERY_CLOSE_RIGHT", nr, "RIGHT", zc

    # Priority 2-3: strong warnings
    if zc["front"] >= ZONE_MIN_POINTS and nf is not None and nf <= STRONG_WARNING_DISTANCE_M:
        return "STRONG_FRONT", nf, "FRONT", zc
    if zc["left"] >= ZONE_MIN_POINTS and nl is not None and nl <= STRONG_WARNING_DISTANCE_M:
        return "STRONG_LEFT", nl, "LEFT", zc
    if zc["right"] >= ZONE_MIN_POINTS and nr is not None and nr <= STRONG_WARNING_DISTANCE_M:
        return "STRONG_RIGHT", nr, "RIGHT", zc

    # Both sides
    if zc["left"] >= ZONE_MIN_POINTS and zc["right"] >= ZONE_MIN_POINTS:
        return "BOTH_SIDES", min(nl, nr), "BOTH", zc

    # Priority 4-5: normal within 1 m
    if zc["front"] >= ZONE_MIN_POINTS:
        return "NORMAL_FRONT", nf, "FRONT", zc
    if zc["left"] >= ZONE_MIN_POINTS:
        return "NORMAL_LEFT", nl, "LEFT", zc
    if zc["right"] >= ZONE_MIN_POINTS:
        return "NORMAL_RIGHT", nr, "RIGHT", zc

    # Back
    if zc["back"] >= ZONE_MIN_POINTS:
        return "BACK", nb, "BACK", zc

    return "CLEAR", overall, "", zc


def compute_direction_distances(latest_points):
    """Nearest distance per direction for on-screen panel (caution up to 1.2 m)."""
    back_d = _zone_nearest(latest_points,
                           lambda x, y, d: x < 0 and abs(y) <= 0.45 and d <= 0.75)
    left_d = _zone_nearest(latest_points,
                           lambda x, y, d: y < -0.35 and -0.3 <= x <= 1.2 and d <= CAUTION_DISTANCE_M)
    front_d = _zone_nearest(latest_points,
                            lambda x, y, d: x > 0 and abs(y) <= 0.45 and d <= CAUTION_DISTANCE_M)
    right_d = _zone_nearest(latest_points,
                            lambda x, y, d: y > 0.35 and -0.3 <= x <= 1.2 and d <= CAUTION_DISTANCE_M)
    return back_d, left_d, front_d, right_d


def alert_to_banner(alert_state):
    """Screen banner text (can show distances in panel, not in voice)."""
    mapping = {
        "CLEAR": "CLEAR",
        "NORMAL_FRONT": "OBSTACLE AHEAD",
        "STRONG_FRONT": "CAREFUL: OBSTACLE AHEAD",
        "VERY_CLOSE_FRONT": "STOP: VERY CLOSE AHEAD",
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
    """
    Stable voice state machine — display updates instantly, voice uses confirmed state only.
    """
    global raw_alert_state, candidate_alert_state, candidate_count
    global confirmed_alert_state, current_alert_state, clear_streak
    global display_banner_text, display_warning_state, zone_counts

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

    current_alert_state = confirmed_alert_state
    display_warning_state = raw_alert
    display_banner_text = alert_to_banner(raw_alert)

    if raw_alert == "CLEAR":
        clear_streak += 1
    else:
        clear_streak = 0


def update_alerts():
    global nearest_distance_m, nearest_left_m, nearest_front_m, nearest_right_m, nearest_back_m
    global zone_counts

    with data_lock:
        latest = list(latest_scan_points)

    alert, nearest, direction, zc = detect_obstacles_for_blind_user(latest)
    zone_counts = zc
    back_d, left_d, front_d, right_d = compute_direction_distances(latest)

    nearest_distance_m = nearest
    nearest_back_m = back_d
    nearest_left_m = left_d
    nearest_front_m = front_d
    nearest_right_m = right_d

    update_voice_state_machine(alert)
    process_voice_alerts()


def warning_to_speech(alert_state):
    """Natural short guidance phrases for blind navigation."""
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


def process_voice_alerts():
    """Speak confirmed alerts only — emergency VERY_CLOSE interrupts cooldown."""
    global last_spoken_alert_state, last_voice_time, last_clear_voice_time

    if not voice_enabled:
        return

    now = time.time()
    alert = confirmed_alert_state
    elapsed = now - last_voice_time

    # Emergency STOP interrupts and speaks immediately
    if is_very_close_alert(alert):
        if alert != last_spoken_alert_state or elapsed >= VOICE_REPEAT_SECONDS:
            speak_voice_alert(alert, force=True)
        return

    if alert == "CLEAR":
        if last_spoken_alert_state and last_spoken_alert_state != "CLEAR":
            if clear_streak >= ZONE_CLEAR_SCANS and elapsed >= CLEAR_VOICE_MIN_GAP:
                speak_voice_alert("CLEAR", force=False)
        return

    if alert == last_spoken_alert_state:
        if elapsed >= VOICE_REPEAT_SECONDS:
            speak_voice_alert(alert, force=False)
        return

    if elapsed >= VOICE_COOLDOWN_SECONDS:
        speak_voice_alert(alert, force=False)


def speak_voice_alert(alert_state, force=False):
    global last_spoken_alert_state, last_voice_time, last_clear_voice_time

    if not voice_enabled or alert_state is None:
        return False

    now = time.time()
    if not force and now - last_voice_time < VOICE_COOLDOWN_SECONDS:
        return False

    if force:
        stop_current_voice()

    text = warning_to_speech(alert_state)
    print(f">>> VOICE [{ESPEAK_SPEED} wpm]: {text}")
    if not run_tts(text):
        print("\a", end="", flush=True)
        return False

    last_spoken_alert_state = alert_state
    last_voice_time = now
    if alert_state == "CLEAR":
        last_clear_voice_time = now
    return True


# =============================================================================
# AUDIO / TTS
# =============================================================================

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
        print(f"Voice OK: using {tts_executable}")
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


def test_voice():
    stop_current_voice()
    global last_voice_time
    last_voice_time = 0.0
    run_tts("Path clear. Voice system ready.")


# =============================================================================
# BACKGROUND READER THREAD
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


def warning_color(banner_text):
    """Colour banner by danger level: green / yellow / orange / red."""
    if "STOP" in banner_text:
        return COLOR_WARN_STOP
    if "CAREFUL" in banner_text:
        return COLOR_WARN_STRONG
    if "OBSTACLE" in banner_text or "BOTH SIDES" in banner_text:
        return COLOR_WARN_FRONT
    return COLOR_WARN_CLEAR


def make_fonts():
    """Font sizes scale up in accessibility mode."""
    scale = 1.35 if ACCESSIBILITY_MODE else 1.0
    sm = max(11, int(13 * scale))
    md = max(13, int(15 * scale))
    lg = max(18, int(22 * scale))
    ring = max(10, int(11 * scale))
    return (
        pygame.font.SysFont("monospace", sm),
        pygame.font.SysFont("monospace", md, bold=True),
        pygame.font.SysFont("monospace", lg, bold=True),
        pygame.font.SysFont("monospace", ring),
    )


def dist_panel_color(d):
    if d is None:
        return COLOR_DIST_OK
    if d <= VERY_CLOSE_DISTANCE_M:
        return COLOR_DIST_DANGER
    if d <= ALERT_DISTANCE_M:
        return COLOR_DIST_WARN
    if d <= CAUTION_DISTANCE_M:
        return (220, 180, 60)
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


def get_connected_wall_components(grid_occ):
    """BFS group neighbouring strong wall cells; filter small noise blobs."""
    strong = {(ix, iy) for (ix, iy), h in grid_occ.items() if h >= WALL_STRONG_HITS}
    visited = set()
    components = []
    for cell in strong:
        if cell in visited:
            continue
        queue = [cell]
        comp = []
        visited.add(cell)
        while queue:
            c = queue.pop(0)
            comp.append(c)
            for dix, diy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (c[0] + dix, c[1] + diy)
                if nb in strong and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if len(comp) >= MIN_WALL_COMPONENT_SIZE:
            components.append(comp)
    return components


def draw_strong_wall_cells(screen, sw, sh, grid_occ):
    """Thick bright markers on confirmed wall cells."""
    cell_px = max(5, int(GRID_RESOLUTION_M * PIXELS_PER_METER) + 3)
    if ACCESSIBILITY_MODE:
        cell_px += 2
    for (ix, iy), hits in grid_occ.items():
        if hits < WALL_STRONG_HITS:
            continue
        x_m, y_m = grid_to_world(ix, iy)
        sx, sy = world_to_screen(x_m, y_m, sw, sh)
        colour = occupied_color(hits)
        rect = pygame.Rect(sx - cell_px // 2, sy - cell_px // 2, cell_px, cell_px)
        pygame.draw.rect(screen, colour, rect)
        pygame.draw.rect(screen, (235, 255, 245), rect, 2)


def draw_connected_walls(screen, sw, sh, grid_occ):
    """Draw continuous wall segments from connected components."""
    line_w = 5 if ACCESSIBILITY_MODE else 4
    wall_col = (200, 255, 220) if ACCESSIBILITY_MODE else (140, 240, 180)
    cell_px = max(4, int(GRID_RESOLUTION_M * PIXELS_PER_METER) + 2)
    layer = pygame.Surface((sw, sh), pygame.SRCALPHA)

    for comp in get_connected_wall_components(grid_occ):
        screen_pts = []
        for ix, iy in comp:
            x_m, y_m = grid_to_world(ix, iy)
            screen_pts.append(world_to_screen(x_m, y_m, sw, sh))

        # Draw thick squares for each cell
        for sx, sy in screen_pts:
            size = cell_px + (2 if ACCESSIBILITY_MODE else 0)
            pygame.draw.rect(layer, (*wall_col, 230),
                             pygame.Rect(sx - size // 2, sy - size // 2, size, size))

        # Connect neighbours within component
        for i, p1 in enumerate(screen_pts):
            for p2 in screen_pts[i + 1:]:
                if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < cell_px * 2.2:
                    pygame.draw.line(layer, (*wall_col, 200), p1, p2, line_w)

    screen.blit(layer, (0, 0))


def draw_weak_occupied(screen, sw, sh, grid_occ):
    """Dim cyan cells below strong wall threshold."""
    cell_px = max(3, int(GRID_RESOLUTION_M * PIXELS_PER_METER))
    for (ix, iy), hits in grid_occ.items():
        if OCCUPIED_MIN_HITS <= hits < WALL_STRONG_HITS:
            x_m, y_m = grid_to_world(ix, iy)
            sx, sy = world_to_screen(x_m, y_m, sw, sh)
            pygame.draw.rect(screen, (40, 160, 200),
                             pygame.Rect(sx - cell_px // 2, sy - cell_px // 2, cell_px, cell_px))


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


def draw_warning_zones(screen, sw, sh, banner_text):
    """Transparent front/left/right alert zones coloured by danger."""
    surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
    margin = 0.55 if ACCESSIBILITY_MODE else 0.40

    def zone_rect(x0, y0, w, h, base_col, active):
        sx0, sy0 = world_to_screen(x0 + w, y0, sw, sh)
        sx1, sy1 = world_to_screen(x0, y0 + h, sw, sh)
        rect = pygame.Rect(min(sx0, sx1), min(sy0, sy1), abs(sx1 - sx0), abs(sy1 - sy0))
        alpha = 90 if ACCESSIBILITY_MODE and active else (70 if active else 25)
        pygame.draw.rect(surf, (*base_col, alpha), rect)
        pygame.draw.rect(surf, (*base_col, 140 if active else 40), rect, 2 if ACCESSIBILITY_MODE else 1)

    front_on = "AHEAD" in banner_text or "FRONT" in banner_text
    back_on = "BEHIND" in banner_text or "BACK" in banner_text
    left_on = "LEFT" in banner_text
    right_on = "RIGHT" in banner_text

    zone_rect(0, -margin, ALERT_DISTANCE_M, margin * 2, (255, 220, 50), front_on)
    zone_rect(-0.75, -margin, 0.75, margin * 2, (255, 200, 80), back_on)
    zone_rect(-0.3, -3.0, 1.5, 2.65, (255, 150, 50), left_on)
    zone_rect(-0.3, 0.35, 1.5, 2.65, (255, 150, 50), right_on)

    cx, cy = sw // 2, sh // 2
    tip_x, tip_y = world_to_screen(ALERT_DISTANCE_M, 0, sw, sh)
    left_x, left_y = world_to_screen(0, -margin, sw, sh)
    right_x, right_y = world_to_screen(0, margin, sw, sh)
    cone_col = (255, 60, 60, 70) if "STOP" in banner_text else (255, 220, 50, 40)
    pygame.draw.polygon(surf, cone_col, [(cx, cy), (left_x, left_y), (tip_x, tip_y), (right_x, right_y)])

    screen.blit(surf, (0, 0))


def draw_distance_panel(screen, font, sw):
    """Side panel: nearest distance back / left / front / right."""
    pw, ph = 200, 135
    px, py = sw - pw - 14, 60
    panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
    panel.fill((*COLOR_PANEL_BG, 210))
    screen.blit(panel, (px, py))
    pygame.draw.rect(screen, (60, 90, 130), (px, py, pw, ph), 1)

    rows = [
        ("BACK", nearest_back_m),
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


def draw_debug_panel(screen, font, sw, sh):
    """Voice alert state machine debug (D key)."""
    now = time.time()
    with data_lock:
        zc = dict(zone_counts)
    lines = [
        "VOICE DEBUG",
        f"raw: {raw_alert_state}",
        f"candidate: {candidate_alert_state}  ({candidate_count}/{VOICE_MIN_DETECTIONS})",
        f"confirmed: {confirmed_alert_state}",
        f"last spoken: {last_spoken_alert_state or '-'}",
        f"since voice: {now - last_voice_time:.1f}s",
        f"zones F/L/R/B: {zc.get('front', 0)}/{zc.get('left', 0)}/"
        f"{zc.get('right', 0)}/{zc.get('back', 0)}",
        f"very close F/L/R: {zc.get('vc_front', 0)}/{zc.get('vc_left', 0)}/{zc.get('vc_right', 0)}",
    ]
    pw = 340 if ACCESSIBILITY_MODE else 300
    ph = len(lines) * 18 + 16
    px, py = 12, sh - ph - 28
    panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
    panel.fill((*COLOR_PANEL_BG, 220))
    screen.blit(panel, (px, py))
    pygame.draw.rect(screen, (80, 120, 180), (px, py, pw, ph), 1)
    y = py + 8
    for i, line in enumerate(lines):
        col = (255, 220, 100) if i == 0 else COLOR_HUD
        screen.blit(font.render(line, True, col), (px + 10, y))
        y += 18


def draw_hud(screen, fonts, sw, sh, fps):
    small, med, large, _ring = fonts
    warn_col = warning_color(display_banner_text)

    banner_h = large.get_height() + (12 if ACCESSIBILITY_MODE else 4)
    banner_bg = pygame.Surface((sw, banner_h), pygame.SRCALPHA)
    banner_bg.fill((0, 0, 0, 120))
    screen.blit(banner_bg, (0, 0))

    banner = large.render(display_banner_text, True, warn_col)
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
        "D6 AA55 Clear Room Mapper v2",
        f"{PORT if not SIMULATED_MODE else 'SIM'} @ {BAUD}  |  {mode_names[view_mode]}  "
        f"|  Zoom:{PIXELS_PER_METER} px/m",
        f"Points:{n_pts}  Free:{n_free}  Walls:{n_occ}  Pkts:{n_pkt}  FPS:{fps:.0f}",
        f"Voice:{'ON' if voice_enabled else 'OFF'}  "
        f"Confirm:{candidate_count}/{VOICE_MIN_DETECTIONS}  "
        f"A11y:{'ON' if ACCESSIBILITY_MODE else 'OFF'}  "
        f"Debug:{'ON' if show_debug else 'OFF'}  "
        f"{'PAUSED' if mapping_paused else 'LIVE'}",
    ]
    y = banner_h + 4
    for i, line in enumerate(lines):
        fnt = med if i == 0 else small
        screen.blit(fnt.render(line, True, COLOR_HUD), (12, y))
        y += fnt.get_height() + 2

    controls = ("Q=Quit  C=Clear  S=Save  M=Mode  V=Voice  T=Test  D=Debug  "
                "A=A11y  +/-=Zoom  Space=Pause  R=Rays  H=Help")
    screen.blit(small.render(controls, True, (110, 130, 155)), (12, sh - 22))

    draw_distance_panel(screen, small, sw)

    if show_debug:
        draw_debug_panel(screen, small, sw, sh)

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
        "Alerts within 1 metre (voice after 8 stable scans):",
        "  Front / Left / Right / Back zones",
        "",
        "M = cycle RAW / MAP / COMBINED view",
        "D = voice debug panel",
        "A = accessibility mode (larger text/walls)",
        "+/- = zoom map scale",
        "Display updates instantly; voice uses confirmed state",
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
    global show_help, fullscreen, view_mode, voice_enabled, show_debug
    global ACCESSIBILITY_MODE, PIXELS_PER_METER

    print("=" * 72)
    print("Clear 2D Room Mapper + Blind Navigation Alerts (v2)")
    print("=" * 72)
    print("NOT full SLAM — no odometry, IMU, or pose tracking.")
    print("Prototype aid only — test voice in a safe environment.")
    print()

    init_voice_system()

    connection = None
    if not SIMULATED_MODE:
        connection = open_serial_port(PORT, BAUD, TIMEOUT)
        ser = connection

    reader = threading.Thread(target=serial_reader_loop, args=(connection,), daemon=True)
    reader.start()

    pygame.init()
    pygame.display.set_caption("D6 AA55 Clear Room Mapper v2")
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    fonts = make_fonts()

    if voice_enabled and check_tts():
        pygame.time.wait(800)
        test_voice()

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
                    elif event.key == pygame.K_t:
                        test_voice()
                    elif event.key == pygame.K_d:
                        show_debug = not show_debug
                        print(f"Debug {'ON' if show_debug else 'OFF'}")
                    elif event.key == pygame.K_a:
                        ACCESSIBILITY_MODE = not ACCESSIBILITY_MODE
                        fonts = make_fonts()
                        print(f"Accessibility mode {'ON' if ACCESSIBILITY_MODE else 'OFF'}")
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        PIXELS_PER_METER = min(PIXELS_PER_METER_MAX, PIXELS_PER_METER + 10)
                        print(f"Zoom: {PIXELS_PER_METER} px/m")
                    elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                        PIXELS_PER_METER = max(PIXELS_PER_METER_MIN, PIXELS_PER_METER - 10)
                        print(f"Zoom: {PIXELS_PER_METER} px/m")
                    elif event.key == pygame.K_m:
                        view_mode = (view_mode + 1) % 3
                        names = ["RAW", "MAP", "COMBINED"]
                        print(f"View mode: {names[view_mode]}")
                    elif event.key == pygame.K_h:
                        show_help = not show_help

            sw, sh = screen.get_size()
            update_alerts()

            with data_lock:
                free_snap = dict(free_grid)
                occ_snap = dict(occupied_grid)
                latest = list(latest_scan_points)
                pts = list(points_xy)

            screen.fill(COLOR_BG)
            draw_range_rings(screen, sw, sh, fonts[3])

            if view_mode == VIEW_RAW:
                draw_raw_points(screen, sw, sh, pts)
                draw_latest_points(screen, sw, sh, latest)

            if show_grid_map and view_mode in (VIEW_MAP, VIEW_COMBINED):
                draw_free_cells(screen, sw, sh, free_snap, occ_snap)
                draw_weak_occupied(screen, sw, sh, occ_snap)
                draw_connected_walls(screen, sw, sh, occ_snap)
                draw_strong_wall_cells(screen, sw, sh, occ_snap)
                if view_mode == VIEW_COMBINED:
                    draw_occupied_cells(screen, sw, sh, occ_snap)

            if view_mode in (VIEW_MAP, VIEW_COMBINED):
                draw_latest_points(screen, sw, sh, latest)

            if show_rays and view_mode != VIEW_MAP:
                draw_scan_rays(screen, sw, sh, latest)

            draw_warning_zones(screen, sw, sh, display_banner_text)
            draw_sensor(screen, sw, sh)

            fx, fy = world_to_screen(0.6, 0, sw, sh)
            bx, by = world_to_screen(-0.6, 0, sw, sh)
            screen.blit(fonts[0].render("FRONT", True, (180, 200, 220)), (fx - 20, fy - 20))
            screen.blit(fonts[0].render("BACK", True, (140, 160, 180)), (bx - 18, by - 10))

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
