"""
Real-Time 2D Room and Surroundings Mapping Using D6 AA55 LiDAR on Raspberry Pi 5
=================================================================================

Pygame top-down LiDAR viewer — robot / radar / driving-simulator style display.

Install:
    sudo apt update
    sudo apt install python3-serial python3-numpy python3-pygame

Run:
    python3 d6_aa55_pygame_room_mapper.py

What the LiDAR does:
--------------------
The D6 AA55 spins and measures distance at many angles around 360°. Each
measurement is a polar coordinate: angle (degrees) + distance (cm). We convert
those into flat X/Y map coordinates (forward = +X, right = +Y) to draw a
top-down room map around the sensor.

Occupancy grid:
---------------
The map is also stored in a grid of small cells. Every hit increments a cell's
counter. Random noise rarely repeats in the same cell, but real walls do — so
repeated hits make wall outlines brighter and clearer over time.

This is NOT full SLAM:
----------------------
Scans are drawn in a fixed frame centred on the sensor. There is no wheel
odometry, IMU, robot pose tracking, or loop closure. True SLAM needs movement
tracking (encoders + IMU + pose estimation) to merge scans while the robot moves.

Test without hardware:
    Set SIMULATED_MODE = True
"""

import csv
import math
import os
import random
import struct
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
PIXELS_PER_METER = 80
FPS_TARGET = 60

# Colours (R, G, B)
COLOR_BG = (8, 12, 28)
COLOR_GRID_RING = (30, 80, 140)
COLOR_GRID_RING_FAINT = (20, 50, 90)
COLOR_SENSOR = (220, 230, 255)
COLOR_SWEEP = (80, 220, 255)
COLOR_SWEEP_GLOW = (40, 120, 180)
COLOR_RAY = (30, 70, 120)
COLOR_POINT_NEW = (100, 255, 220)
COLOR_POINT_OLD = (40, 120, 180)
COLOR_HUD = (180, 200, 220)
COLOR_WARN_CLEAR = (80, 220, 100)
COLOR_WARN_AHEAD = (255, 220, 60)
COLOR_WARN_SIDE = (255, 160, 50)
COLOR_WARN_STOP = (255, 60, 60)
COLOR_ZONE_FRONT = (40, 80, 40)
COLOR_ZONE_LEFT = (40, 40, 80)
COLOR_ZONE_RIGHT = (80, 40, 40)

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
# Point cloud / grid
# ---------------------------------------------------------------------------
MAX_POINTS = 12000
GRID_RESOLUTION_M = 0.05
OCCUPANCY_MIN_HITS = 2
MAX_OCCUPANCY_HITS = 20
RAY_DRAW_STEP = 5
POINT_FADE_SEC = 8.0

# ---------------------------------------------------------------------------
# Obstacle warning zones
# ---------------------------------------------------------------------------
VERY_CLOSE_M = 0.45
FRONT_WARNING_DISTANCE_M = 1.2
SIDE_WARNING_DISTANCE_M = 0.8

# ---------------------------------------------------------------------------
# Save filenames
# ---------------------------------------------------------------------------
POINTS_CSV = "lidar_room_points.csv"
OCCUPANCY_CSV = "lidar_room_occupancy.csv"
MAP_PNG = "lidar_room_map.png"

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
data_lock = threading.Lock()
points_xy = []              # (x_m, y_m, distance_m, timestamp)
occupancy_grid = {}         # (ix, iy) -> hit count
latest_scan_points = []     # most recent scan batch
packet_count = 0
latest_angle_deg = 0.0
mapping_paused = False
shutdown_requested = False
current_warning = "CLEAR"
closest_obstacle_m = None
last_printed_warning = ""

# Display toggles
show_rays = True
show_grid = True
show_raw_points = True
fullscreen = False

ser = None

# Simulated room geometry (metres, X forward, Y right)
SIM_ROOM = {
    "x_min": -1.0, "x_max": 4.5,
    "y_min": -2.8, "y_max": 2.8,
}
SIM_OBSTACLES = [
    # table
    {"x_min": 1.5, "x_max": 2.3, "y_min": -0.5, "y_max": 0.5},
    # chair left
    {"x_min": 0.8, "x_max": 1.2, "y_min": -1.6, "y_max": -1.1},
    # chair right
    {"x_min": 2.0, "x_max": 2.5, "y_min": 1.2, "y_max": 1.7},
    # box
    {"x_min": 3.2, "x_max": 3.7, "y_min": -1.8, "y_max": -1.3},
]


# ---------------------------------------------------------------------------
# Serial helpers
# ---------------------------------------------------------------------------
def list_serial_ports():
    """Return available serial ports using pyserial's list_ports."""
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
    """Connect to the LiDAR or exit with a helpful message."""
    try:
        connection = serial.Serial(port, baud, timeout=timeout)
        print(f"Connected to LiDAR: {port} @ {baud} baud")
        return connection
    except serial.SerialException as exc:
        print(f"ERROR: Could not open '{port}': {exc}")
        available = list_serial_ports()
        if available:
            print("Available serial ports:")
            for p in available:
                print(f"  - {p}")
            print("Edit PORT at the top of this file.")
        else:
            print("No serial ports found. Check USB cable.")
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# AA55 packet reading and parsing
# ---------------------------------------------------------------------------
def read_packet(connection):
    """
    Search for AA55 header and return one complete packet.
    Returns None on timeout — does not crash on incomplete data.
    """
    while True:
        try:
            b = connection.read(1)
        except serial.SerialException as exc:
            print(f"WARNING: Serial read error: {exc}")
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


def parse_packet(packet):
    """Return list of (angle_deg, distance_cm) from one AA55 packet."""
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


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------
def polar_to_xy(angle_deg, distance_cm):
    """
    Convert polar LiDAR reading to map coordinates (metres).
    0° = forward (+X), 90° = right (+Y).
    """
    distance_m = distance_cm / 100.0
    rad = math.radians(angle_deg)
    x = distance_m * math.cos(rad)
    y = distance_m * math.sin(rad)
    return x, y, distance_m


def world_to_screen(x_m, y_m, screen_width, screen_height):
    """
    Map world metres to pixel coordinates.
    Sensor at screen centre; forward (X) points up; right (Y) points right.
    """
    center_x = screen_width // 2
    center_y = screen_height // 2
    screen_x = int(center_x + y_m * PIXELS_PER_METER)
    screen_y = int(center_y - x_m * PIXELS_PER_METER)
    return screen_x, screen_y


def grid_index(x_m, y_m):
    ix = round(x_m / GRID_RESOLUTION_M)
    iy = round(y_m / GRID_RESOLUTION_M)
    return ix, iy


def cell_center_m(ix, iy):
    return ix * GRID_RESOLUTION_M, iy * GRID_RESOLUTION_M


# ---------------------------------------------------------------------------
# Point cloud and occupancy
# ---------------------------------------------------------------------------
def add_scan_points(scan_points):
    """Add scan hits to point cloud, occupancy grid, and latest scan buffer."""
    global points_xy, occupancy_grid, packet_count
    global latest_scan_points, latest_angle_deg

    now = time.time()
    batch = []

    for angle_deg, distance_cm in scan_points:
        x_m, y_m, distance_m = polar_to_xy(angle_deg, distance_cm)
        batch.append((x_m, y_m, distance_m, now))

        key = grid_index(x_m, y_m)
        hits = occupancy_grid.get(key, 0) + 1
        occupancy_grid[key] = min(hits, MAX_OCCUPANCY_HITS)

    if scan_points:
        latest_angle_deg = scan_points[-1][0]

    with data_lock:
        points_xy.extend(batch)
        if len(points_xy) > MAX_POINTS:
            points_xy = points_xy[-MAX_POINTS:]
        latest_scan_points = batch
        packet_count += 1


def clear_map():
    """Clear all stored map data."""
    global points_xy, occupancy_grid, packet_count
    global latest_scan_points, current_warning, closest_obstacle_m
    with data_lock:
        points_xy = []
        occupancy_grid = {}
        latest_scan_points = []
        packet_count = 0
        current_warning = "CLEAR"
        closest_obstacle_m = None
    print("Map cleared.")


# ---------------------------------------------------------------------------
# Simulated LiDAR (test without hardware)
# ---------------------------------------------------------------------------
def ray_cast_distance(angle_deg):
    """
    Cast a ray in simulated room and return distance to nearest wall/obstacle.
    Used when SIMULATED_MODE = True.
    """
    rad = math.radians(angle_deg)
    dx = math.cos(rad)
    dy = math.sin(rad)

    best = MAX_RANGE_M

    def check_horizontal(y_wall):
        nonlocal best
        if abs(dy) < 1e-9:
            return
        t = y_wall / dy
        if t > 0:
            x_hit = t * dx
            if SIM_ROOM["x_min"] <= x_hit <= SIM_ROOM["x_max"]:
                best = min(best, t)

    def check_vertical(x_wall):
        nonlocal best
        if abs(dx) < 1e-9:
            return
        t = x_wall / dx
        if t > 0:
            y_hit = t * dy
            if SIM_ROOM["y_min"] <= y_hit <= SIM_ROOM["y_max"]:
                best = min(best, t)

    # Room walls
    check_vertical(SIM_ROOM["x_min"])
    check_vertical(SIM_ROOM["x_max"])
    check_horizontal(SIM_ROOM["y_min"])
    check_horizontal(SIM_ROOM["y_max"])

    # Obstacle boxes
    for box in SIM_OBSTACLES:
        for x_wall in (box["x_min"], box["x_max"]):
            if abs(dx) > 1e-9:
                t = x_wall / dx
                if t > 0:
                    y_hit = t * dy
                    if box["y_min"] <= y_hit <= box["y_max"]:
                        best = min(best, t)
        for y_wall in (box["y_min"], box["y_max"]):
            if abs(dy) > 1e-9:
                t = y_wall / dy
                if t > 0:
                    x_hit = t * dx
                    if box["x_min"] <= x_hit <= box["x_max"]:
                        best = min(best, t)

    # Small noise
    best += random.uniform(-0.02, 0.02)
    best = max(MIN_RANGE_CM / 100.0, min(best, MAX_RANGE_M))
    return best * 100.0  # return cm


def generate_simulated_scan():
    """Generate one fake AA55-style scan (list of angle, distance_cm)."""
    points = []
    # Simulate ~24 samples per packet, full rotation over many packets
    base = random.uniform(0, 360)
    for i in range(24):
        angle = (base + i * 4.5) % 360.0
        dist_cm = ray_cast_distance(angle)
        points.append((angle, dist_cm))
    return points


# ---------------------------------------------------------------------------
# Obstacle detection
# ---------------------------------------------------------------------------
def detect_obstacle_zones(points):
    """
    Detect navigation warnings from current point cloud.
    Returns (warning_text, closest_distance_m).
    """
    if not points:
        return "CLEAR", None

    front_nearest = None
    left_nearest = None
    right_nearest = None
    overall_nearest = None
    very_close = False

    for x_m, y_m, distance_m, _ts in points:
        if overall_nearest is None or distance_m < overall_nearest:
            overall_nearest = distance_m

        if distance_m < VERY_CLOSE_M and abs(y_m) < 0.5 and x_m > 0:
            very_close = True

        if abs(y_m) < 0.35 and 0 < x_m < FRONT_WARNING_DISTANCE_M:
            if front_nearest is None or distance_m < front_nearest:
                front_nearest = distance_m

        if y_m < -0.35 and abs(x_m) < 1.0 and distance_m < SIDE_WARNING_DISTANCE_M:
            if left_nearest is None or distance_m < left_nearest:
                left_nearest = distance_m

        if y_m > 0.35 and abs(x_m) < 1.0 and distance_m < SIDE_WARNING_DISTANCE_M:
            if right_nearest is None or distance_m < right_nearest:
                right_nearest = distance_m

    if very_close:
        return "STOP: VERY CLOSE OBJECT", overall_nearest
    if front_nearest is not None:
        return "OBSTACLE AHEAD", overall_nearest
    if left_nearest is not None:
        return "OBSTACLE LEFT", overall_nearest
    if right_nearest is not None:
        return "OBSTACLE RIGHT", overall_nearest

    return "CLEAR", overall_nearest


def update_warning_state():
    """Update warning and print to terminal only when it changes."""
    global current_warning, closest_obstacle_m, last_printed_warning

    with data_lock:
        snapshot = list(points_xy)

    warning, nearest = detect_obstacle_zones(snapshot)
    current_warning = warning
    closest_obstacle_m = nearest

    if warning != last_printed_warning:
        dist_txt = f"{nearest:.2f} m" if nearest is not None else "n/a"
        print(f"WARNING: {warning}  (closest: {dist_txt})")
        last_printed_warning = warning


def warning_color(warning):
    if "STOP" in warning:
        return COLOR_WARN_STOP
    if "AHEAD" in warning:
        return COLOR_WARN_AHEAD
    if "LEFT" in warning or "RIGHT" in warning:
        return COLOR_WARN_SIDE
    return COLOR_WARN_CLEAR


# ---------------------------------------------------------------------------
# Background serial / simulation thread
# ---------------------------------------------------------------------------
def serial_reader_loop(connection):
    """Read LiDAR packets (or simulate) and update the map."""
    global shutdown_requested, mapping_paused, latest_angle_deg

    mode = "SIMULATED" if SIMULATED_MODE else "SERIAL"
    print(f"Reader thread started ({mode}).")

    sim_sweep = 0.0

    while not shutdown_requested:
        if mapping_paused:
            time.sleep(0.05)
            continue

        if SIMULATED_MODE:
            scan_points = generate_simulated_scan()
            sim_sweep = (sim_sweep + 6.0) % 360.0
            latest_angle_deg = sim_sweep
            time.sleep(0.04)
        else:
            packet = read_packet(connection)
            scan_points = parse_packet(packet)

        if scan_points:
            add_scan_points(scan_points)

        time.sleep(0.001)

    print("Reader thread stopped.")


# ---------------------------------------------------------------------------
# Save map
# ---------------------------------------------------------------------------
def save_map(surface):
    """Save point CSV, occupancy CSV, and PNG screenshot."""
    with data_lock:
        pts = list(points_xy)
        grid = dict(occupancy_grid)

    if not pts and not grid:
        print("Nothing to save — map is empty.")
        return

    with open(POINTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_m", "y_m", "distance_m", "timestamp"])
        for x_m, y_m, dist_m, ts in pts:
            writer.writerow([f"{x_m:.5f}", f"{y_m:.5f}", f"{dist_m:.5f}", f"{ts:.3f}"])

    with open(OCCUPANCY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ix", "iy", "x_m", "y_m", "hit_count"])
        for (ix, iy), hits in sorted(grid.items()):
            cx, cy = cell_center_m(ix, iy)
            writer.writerow([ix, iy, f"{cx:.3f}", f"{cy:.3f}", hits])

    pygame.image.save(surface, MAP_PNG)
    print(f"Saved: {POINTS_CSV}")
    print(f"Saved: {OCCUPANCY_CSV}")
    print(f"Saved: {MAP_PNG}")


# ---------------------------------------------------------------------------
# Pygame drawing helpers
# ---------------------------------------------------------------------------
def hit_count_color(hits):
    """Map occupancy hit count to a wall-outline colour."""
    ratio = min(hits / MAX_OCCUPANCY_HITS, 1.0)
    if ratio < 0.35:
        r = int(30 + 40 * ratio)
        g = int(60 + 80 * ratio)
        b = int(140 + 60 * ratio)
    elif ratio < 0.7:
        t = (ratio - 0.35) / 0.35
        r = int(50 + 80 * t)
        g = int(180 + 60 * t)
        b = int(180 - 40 * t)
    else:
        t = (ratio - 0.7) / 0.3
        r = int(130 + 125 * t)
        g = int(240 + 15 * t)
        b = int(140 + 115 * t)
    return (r, g, b)


def fade_point_color(age_sec):
    """Fading trail: newer points are brighter cyan/green."""
    fade = max(0.0, 1.0 - age_sec / POINT_FADE_SEC)
    if fade <= 0:
        return None
    r = int(COLOR_POINT_OLD[0] + (COLOR_POINT_NEW[0] - COLOR_POINT_OLD[0]) * fade)
    g = int(COLOR_POINT_OLD[1] + (COLOR_POINT_NEW[1] - COLOR_POINT_OLD[1]) * fade)
    b = int(COLOR_POINT_OLD[2] + (COLOR_POINT_NEW[2] - COLOR_POINT_OLD[2]) * fade)
    return (r, g, b)


def draw_range_rings(screen, sw, sh):
    """Blue circular range rings every 1 metre."""
    cx, cy = sw // 2, sh // 2
    for metres in range(1, int(MAX_RANGE_M) + 1):
        radius = int(metres * PIXELS_PER_METER)
        colour = COLOR_GRID_RING if metres % 2 == 0 else COLOR_GRID_RING_FAINT
        pygame.draw.circle(screen, colour, (cx, cy), radius, 1)
        label = pygame.font.SysFont("monospace", 11).render(f"{metres}m", True, colour)
        screen.blit(label, (cx + 4, cy - radius - 14))


def draw_warning_zones(screen, sw, sh):
    """Semi-transparent warning zone overlays."""
    zone_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)

    def rect_world(x0, y0, w, h, color):
        sx0, sy0 = world_to_screen(x0 + w, y0, sw, sh)
        sx1, sy1 = world_to_screen(x0, y0 + h, sw, sh)
        rect = pygame.Rect(min(sx0, sx1), min(sy0, sy1), abs(sx1 - sx0), abs(sy1 - sy0))
        pygame.draw.rect(zone_surf, color, rect, 1)

    rect_world(0, -0.35, FRONT_WARNING_DISTANCE_M, 0.70,
               (*COLOR_ZONE_FRONT, 60))
    rect_world(-1.0, -SIDE_WARNING_DISTANCE_M, 1.0, SIDE_WARNING_DISTANCE_M - 0.35,
               (*COLOR_ZONE_LEFT, 60))
    rect_world(-1.0, 0.35, 1.0, SIDE_WARNING_DISTANCE_M,
               (*COLOR_ZONE_RIGHT, 60))

    screen.blit(zone_surf, (0, 0))


def draw_sensor(screen, sw, sh):
    """Draw sensor icon, forward arrow, and glow at centre."""
    cx, cy = sw // 2, sh // 2

    # Soft glow
    for r, alpha in ((18, 40), (12, 70), (6, 120)):
        glow = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*COLOR_SENSOR, alpha), (r * 2, r * 2), r)
        screen.blit(glow, (cx - r * 2, cy - r * 2))

    # Forward arrow (points up = forward)
    pygame.draw.polygon(screen, COLOR_SENSOR, [
        (cx, cy - 22), (cx - 8, cy - 6), (cx + 8, cy - 6),
    ])
    pygame.draw.circle(screen, COLOR_SENSOR, (cx, cy), 7)
    pygame.draw.circle(screen, (100, 140, 200), (cx, cy), 7, 1)


def draw_sweep_line(screen, sw, sh, angle_deg):
    """Rotating radar sweep line with glow."""
    cx, cy = sw // 2, sh // 2
    rad = math.radians(angle_deg)
    x_m = MAX_RANGE_M * math.cos(rad)
    y_m = MAX_RANGE_M * math.sin(rad)
    ex, ey = world_to_screen(x_m, y_m, sw, sh)

    for width, colour in ((4, COLOR_SWEEP_GLOW), (2, COLOR_SWEEP)):
        pygame.draw.line(screen, colour, (cx, cy), (ex, ey), width)


def draw_scan_rays(screen, sw, sh, scan_batch):
    """Faint rays from sensor to latest scan hits."""
    cx, cy = sw // 2, sh // 2
    for i, (x_m, y_m, _d, _ts) in enumerate(scan_batch):
        if i % RAY_DRAW_STEP != 0:
            continue
        ex, ey = world_to_screen(x_m, y_m, sw, sh)
        pygame.draw.line(screen, COLOR_RAY, (cx, cy), (ex, ey), 1)
        pygame.draw.circle(screen, COLOR_POINT_NEW, (ex, ey), 2)


def draw_occupancy_grid(screen, sw, sh, grid):
    """Draw occupied cells as coloured squares — walls brighten with hits."""
    half_px = max(1, int(GRID_RESOLUTION_M * PIXELS_PER_METER / 2))
    for (ix, iy), hits in grid.items():
        if hits < OCCUPANCY_MIN_HITS:
            continue
        cx_m, cy_m = cell_center_m(ix, iy)
        sx, sy = world_to_screen(cx_m, cy_m, sw, sh)
        colour = hit_count_color(hits)
        rect = pygame.Rect(sx - half_px, sy - half_px, half_px * 2, half_px * 2)
        pygame.draw.rect(screen, colour, rect)
        if hits >= MAX_OCCUPANCY_HITS * 0.75:
            pygame.draw.rect(screen, (255, 255, 255), rect, 1)


def draw_raw_points(screen, sw, sh, points, now):
    """Draw fading trail of raw scan points."""
    for x_m, y_m, _d, ts in points:
        age = now - ts
        colour = fade_point_color(age)
        if colour is None:
            continue
        sx, sy = world_to_screen(x_m, y_m, sw, sh)
        pygame.draw.circle(screen, colour, (sx, sy), 2)


def draw_hud(screen, font, small_font, stats, fps):
    """Draw status overlay text."""
    sw, sh = screen.get_size()
    warning = stats["warning"]
    w_color = warning_color(warning)
    closest = stats["closest"]
    closest_txt = f"{closest:.2f} m" if closest is not None else "n/a"

    lines_top = [
        "D6 AA55 Real-Time Room Mapper",
        f"Port: {stats['port']} @ {stats['baud']}  |  Mode: {stats['mode']}",
        f"Points: {stats['points']}  |  Cells: {stats['cells']}  |  Packets: {stats['packets']}",
        f"FPS: {fps:.0f}  |  Closest: {closest_txt}",
    ]

    y = 10
    for i, line in enumerate(lines_top):
        fnt = font if i == 0 else small_font
        surf = fnt.render(line, True, COLOR_HUD)
        screen.blit(surf, (12, y))
        y += surf.get_height() + 2

    warn_surf = font.render(warning, True, w_color)
    screen.blit(warn_surf, (12, y + 4))

    controls = [
        "Q/ESC=Quit  C=Clear  S=Save  Space=Pause",
        "R=Rays  G=Grid  P=Points  F=Fullscreen",
    ]
    y = sh - 50
    for line in controls:
        surf = small_font.render(line, True, (120, 140, 160))
        screen.blit(surf, (12, y))
        y += 18

    toggles = []
    toggles.append(f"Rays:{'ON' if stats['rays'] else 'OFF'}")
    toggles.append(f"Grid:{'ON' if stats['grid'] else 'OFF'}")
    toggles.append(f"Pts:{'ON' if stats['pts'] else 'OFF'}")
    toggles.append(f"{'PAUSED' if stats['paused'] else 'LIVE'}")
    toggle_surf = small_font.render("  |  ".join(toggles), True, (100, 130, 160))
    screen.blit(toggle_surf, (sw - toggle_surf.get_width() - 12, 12))


# ---------------------------------------------------------------------------
# Main Pygame loop
# ---------------------------------------------------------------------------
def main():
    global ser, shutdown_requested, mapping_paused
    global show_rays, show_grid, show_raw_points, fullscreen
    global latest_angle_deg

    print("=" * 70)
    print("D6 AA55 Pygame Room Mapper")
    print("=" * 70)
    print(f"Port: {PORT} @ {BAUD}")
    print(f"Simulated mode: {SIMULATED_MODE}")
    print("NOT full SLAM — no odometry, IMU, or pose tracking.")
    print()

    connection = None
    if not SIMULATED_MODE:
        connection = open_serial_port(PORT, BAUD, TIMEOUT)
        ser = connection

    reader = threading.Thread(
        target=serial_reader_loop, args=(connection,), daemon=True,
    )
    reader.start()

    pygame.init()
    pygame.display.set_caption("D6 AA55 Real-Time Room Mapper")
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18, bold=True)
    small_font = pygame.font.SysFont("monospace", 13)

    running = True

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
                        show_grid = not show_grid
                    elif event.key == pygame.K_p:
                        show_raw_points = not show_raw_points
                    elif event.key == pygame.K_f:
                        fullscreen = not fullscreen
                        if fullscreen:
                            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        else:
                            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

            sw, sh = screen.get_size()
            now = time.time()

            update_warning_state()

            with data_lock:
                pts = list(points_xy)
                grid = dict(occupancy_grid)
                scan_batch = list(latest_scan_points)
                n_packets = packet_count
                sweep_angle = latest_angle_deg
                paused = mapping_paused

            screen.fill(COLOR_BG)
            draw_range_rings(screen, sw, sh)
            draw_warning_zones(screen, sw, sh)

            if show_grid:
                draw_occupancy_grid(screen, sw, sh, grid)

            if show_rays and scan_batch:
                draw_scan_rays(screen, sw, sh, scan_batch)

            if show_raw_points:
                draw_raw_points(screen, sw, sh, pts, now)

            draw_sweep_line(screen, sw, sh, sweep_angle)
            draw_sensor(screen, sw, sh)

            n_cells = sum(1 for h in grid.values() if h >= OCCUPANCY_MIN_HITS)
            stats = {
                "port": "SIM" if SIMULATED_MODE else PORT,
                "baud": BAUD,
                "mode": "SIMULATED" if SIMULATED_MODE else "LIVE",
                "points": len(pts),
                "cells": n_cells,
                "packets": n_packets,
                "warning": current_warning,
                "closest": closest_obstacle_m,
                "rays": show_rays,
                "grid": show_grid,
                "pts": show_raw_points,
                "paused": paused,
            }
            fps = clock.get_fps()
            draw_hud(screen, font, small_font, stats, fps)

            pygame.display.flip()
            clock.tick(FPS_TARGET)

    except KeyboardInterrupt:
        print("\nCtrl+C received.")
    finally:
        shutdown_requested = True
        reader.join(timeout=2.0)
        if connection is not None and connection.is_open:
            connection.close()
            print("Serial port closed.")
        pygame.quit()
        print("Stopped safely.")


if __name__ == "__main__":
    main()
