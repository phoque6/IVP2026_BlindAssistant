"""
Real-Time 2D LiDAR Mapping for Obstacle Detection and Blind Navigation Support
================================================================================

Stage 1 — D6 AA55 spinning LiDAR on Raspberry Pi 5.

Install:
    sudo apt update
    sudo apt install python3-serial python3-matplotlib python3-numpy

Run:
    python3 d6_aa55_realtime_2d_map.py

What this program does:
-----------------------
The D6 LiDAR spins and sends distance readings at many angles around a full
360° circle. Each reading is a polar coordinate: an angle (degrees) and a
distance (how far away an obstacle is at that angle).

Polar coordinates are converted into flat X/Y map coordinates so we can draw
a top-down "bird's eye" map. Hits are also stored in an occupancy grid — a
2D table of small cells. When the same cell is hit repeatedly, we trust it more
as a real wall or obstacle (noise usually does not repeat in the same cell).

This is NOT full SLAM (Simultaneous Localisation and Mapping):
- The sensor is assumed stationary or moved very slowly.
- There is no wheel odometry, IMU, pose estimation, or loop closure.
- Scans are simply accumulated in a fixed map frame centred on the sensor.
  For true SLAM you would need extra sensors to track how the robot moves.

Safety note:
    Prototype assistive system only — not a certified mobility aid.
"""

import csv
import math
import os
import struct
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Serial / LiDAR settings
# ---------------------------------------------------------------------------
PORT = "/dev/ttyUSB0"
BAUD = 230400
TIMEOUT = 0.5

MIN_RANGE_CM = 8
MAX_RANGE_M = 6.0
MAX_RANGE_CM = int(MAX_RANGE_M * 100)

# ---------------------------------------------------------------------------
# Point cloud settings
# ---------------------------------------------------------------------------
MAX_POINTS = 8000

# ---------------------------------------------------------------------------
# Occupancy grid settings
# ---------------------------------------------------------------------------
GRID_SIZE_M = 12.0
GRID_RESOLUTION_M = 0.05
OCCUPANCY_MIN_HITS = 2

# ---------------------------------------------------------------------------
# Obstacle warning zones (blind navigation support)
# ---------------------------------------------------------------------------
FRONT_WARNING_DISTANCE_M = 1.2
SIDE_WARNING_DISTANCE_M = 0.8
VERY_CLOSE_DISTANCE_M = 0.45

# ---------------------------------------------------------------------------
# Display / save settings
# ---------------------------------------------------------------------------
UPDATE_MS = 80
POINTS_CSV = "lidar_2d_points.csv"
OCCUPANCY_CSV = "lidar_occupancy_grid.csv"
MAP_PNG = "lidar_2d_map.png"

# ---------------------------------------------------------------------------
# Shared state (thread-safe)
# ---------------------------------------------------------------------------
data_lock = threading.Lock()
points_xy = []              # list of (x, y, distance_m, timestamp)
occupancy_grid = {}         # (ix, iy) -> hit count
packet_count = 0
mapping_paused = False
show_occupancy_view = False
shutdown_requested = False
current_warning = "CLEAR"
last_printed_warning = ""
closest_obstacle_m = None

ser = None


# ---------------------------------------------------------------------------
# Serial helpers
# ---------------------------------------------------------------------------
def list_serial_ports():
    """List likely LiDAR USB serial devices on Linux."""
    try:
        names = os.listdir("/dev")
    except OSError:
        return []

    ports = []
    for name in sorted(names):
        if name.startswith("ttyUSB") or name.startswith("ttyACM"):
            ports.append("/dev/" + name)
    return ports


def open_serial_port(port, baud, timeout):
    """
    Connect to the LiDAR. Print helpful errors and exit if the port is missing.
    """
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
            print("Edit PORT at the top of this file to match your LiDAR.")
        else:
            print("No /dev/ttyUSB* or /dev/ttyACM* devices found.")
            print("Check the USB cable and run:  ls /dev/ttyUSB* /dev/ttyACM*")
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# AA55 packet reading and parsing
# ---------------------------------------------------------------------------
def read_packet(connection):
    """
    Search for an AA55 header and return one complete scan packet.

    Packet layout:
        AA 55 | CT | LSN | FSA | LSA | CS | sample data (LSN x 2 bytes)

    Returns None on timeout or incomplete data — the caller should retry.
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
    """
    Decode one AA55 packet into a list of (angle_deg, distance_cm).

    The LiDAR reports a start angle, end angle, and samples in between.
    Raw distance values are converted to centimetres and filtered by range.
    """
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
    Convert polar LiDAR reading to flat map coordinates (metres).

    Coordinate system (top-down view):
        0°   = forward (+X)
        90°  = right   (+Y)
        X    = forward
        Y    = right
    """
    distance_m = distance_cm / 100.0
    rad = math.radians(angle_deg)
    x = distance_m * math.cos(rad)
    y = distance_m * math.sin(rad)
    return x, y


def grid_index(x, y):
    """Return occupancy grid cell indices for a map point."""
    ix = int(x / GRID_RESOLUTION_M)
    iy = int(y / GRID_RESOLUTION_M)
    return ix, iy


def cell_center_m(ix, iy):
    """Return the centre of a grid cell in metres."""
    return (ix + 0.5) * GRID_RESOLUTION_M, (iy + 0.5) * GRID_RESOLUTION_M


# ---------------------------------------------------------------------------
# Point cloud and occupancy grid management
# ---------------------------------------------------------------------------
def add_scan_points(scan_points):
    """Add a batch of polar scan hits to the point cloud and occupancy grid."""
    global points_xy, occupancy_grid, packet_count

    now = time.time()
    new_entries = []

    for angle_deg, distance_cm in scan_points:
        x, y = polar_to_xy(angle_deg, distance_cm)
        distance_m = distance_cm / 100.0
        new_entries.append((x, y, distance_m, now))

        key = grid_index(x, y)
        occupancy_grid[key] = occupancy_grid.get(key, 0) + 1

    with data_lock:
        points_xy.extend(new_entries)
        if len(points_xy) > MAX_POINTS:
            points_xy = points_xy[-MAX_POINTS:]
        packet_count += 1


def clear_map():
    """Clear the point cloud and occupancy grid."""
    global points_xy, occupancy_grid, packet_count, current_warning, closest_obstacle_m
    with data_lock:
        points_xy = []
        occupancy_grid = {}
        packet_count = 0
        current_warning = "CLEAR"
        closest_obstacle_m = None
    print("Map cleared.")


# ---------------------------------------------------------------------------
# Obstacle detection (blind navigation zones)
# ---------------------------------------------------------------------------
def detect_obstacles():
    """
    Find the closest obstacle in front / left / right warning zones.

    Returns (warning_text, closest_distance_m).
    """
    with data_lock:
        snapshot = list(points_xy)

    if not snapshot:
        return "CLEAR", None

    front_nearest = None
    left_nearest = None
    right_nearest = None
    overall_nearest = None

    for x, y, distance_m, _ts in snapshot:
        if overall_nearest is None or distance_m < overall_nearest:
            overall_nearest = distance_m

        # Front corridor: narrow band ahead of the sensor
        if abs(y) < 0.35 and 0 < x < FRONT_WARNING_DISTANCE_M:
            if front_nearest is None or distance_m < front_nearest:
                front_nearest = distance_m

        # Left side (negative Y = left when Y is right)
        if y < -0.35 and abs(x) < 1.0 and distance_m < SIDE_WARNING_DISTANCE_M:
            if left_nearest is None or distance_m < left_nearest:
                left_nearest = distance_m

        # Right side
        if y > 0.35 and abs(x) < 1.0 and distance_m < SIDE_WARNING_DISTANCE_M:
            if right_nearest is None or distance_m < right_nearest:
                right_nearest = distance_m

    # Priority: stop > front > left > right > clear
    if front_nearest is not None and front_nearest <= VERY_CLOSE_DISTANCE_M:
        return "STOP: VERY CLOSE OBJECT", overall_nearest
    if front_nearest is not None:
        return "OBSTACLE AHEAD", overall_nearest
    if left_nearest is not None:
        return "OBSTACLE LEFT", overall_nearest
    if right_nearest is not None:
        return "OBSTACLE RIGHT", overall_nearest

    return "CLEAR", overall_nearest


def update_warning_state():
    """Update global warning and print to terminal only when it changes."""
    global current_warning, closest_obstacle_m, last_printed_warning

    warning, nearest = detect_obstacles()
    current_warning = warning
    closest_obstacle_m = nearest

    if warning != last_printed_warning:
        dist_text = f"{nearest:.2f} m" if nearest is not None else "n/a"
        print(f"WARNING: {warning}  (closest hit: {dist_text})")
        last_printed_warning = warning


# ---------------------------------------------------------------------------
# Background serial reader thread
# ---------------------------------------------------------------------------
def serial_reader_loop(connection):
    """
    Continuously read LiDAR packets and build the live 2D map.

    Runs in a background thread so the matplotlib display stays responsive.
    """
    global shutdown_requested, mapping_paused

    print("Serial reader thread started.")

    while not shutdown_requested:
        if mapping_paused:
            time.sleep(0.05)
            continue

        packet = read_packet(connection)
        scan_points = parse_packet(packet)

        if scan_points:
            add_scan_points(scan_points)

        time.sleep(0.001)

    print("Serial reader thread stopped.")


# ---------------------------------------------------------------------------
# Save map to disk
# ---------------------------------------------------------------------------
def save_map(fig):
    """Save point cloud CSV, occupancy CSV, and a PNG screenshot."""
    with data_lock:
        points_snapshot = list(points_xy)
        grid_snapshot = dict(occupancy_grid)

    if not points_snapshot and not grid_snapshot:
        print("Nothing to save — map is empty.")
        return

    with open(POINTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "distance_m", "timestamp"])
        for x, y, dist_m, ts in points_snapshot:
            writer.writerow([f"{x:.5f}", f"{y:.5f}", f"{dist_m:.5f}", f"{ts:.3f}"])

    with open(OCCUPANCY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ix", "iy", "x_m", "y_m", "hit_count"])
        for (ix, iy), hits in sorted(grid_snapshot.items()):
            cx, cy = cell_center_m(ix, iy)
            writer.writerow([ix, iy, f"{cx:.3f}", f"{cy:.3f}", hits])

    fig.savefig(MAP_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    print(f"Saved: {POINTS_CSV}")
    print(f"Saved: {OCCUPANCY_CSV}")
    print(f"Saved: {MAP_PNG}")


# ---------------------------------------------------------------------------
# Matplotlib 2D visualisation
# ---------------------------------------------------------------------------
def setup_axes(ax):
    """Draw static map elements: grid lines, range rings, sensor marker."""
    half_grid = GRID_SIZE_M / 2.0
    ax.set_xlim(-MAX_RANGE_M, MAX_RANGE_M)
    ax.set_ylim(-MAX_RANGE_M, MAX_RANGE_M)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Forward X (m)")
    ax.set_ylabel("Right Y (m)")
    ax.set_title("D6 AA55 Real-Time 2D LiDAR Map")
    ax.set_facecolor("#0d0d0d")

    # Faint full grid boundary (12 m x 12 m)
    ax.plot(
        [-half_grid, half_grid, half_grid, -half_grid, -half_grid],
        [-half_grid, -half_grid, half_grid, half_grid, -half_grid],
        color="#333333", linewidth=0.8, linestyle="--",
    )

    # Range rings every 1 metre
    theta = np.linspace(0, 2 * np.pi, 120)
    for radius_m in range(1, int(MAX_RANGE_M) + 1):
        ax.plot(
            radius_m * np.cos(theta),
            radius_m * np.sin(theta),
            color="#444444", linewidth=0.7, alpha=0.8,
        )
        ax.text(radius_m + 0.05, 0, f"{radius_m}m", color="#666666", fontsize=7)

    # Forward direction arrow
    ax.annotate(
        "", xy=(1.0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="white", lw=2),
    )
    ax.text(0.05, 0.12, "FRONT 0°", color="white", fontsize=9)

    # Sensor origin
    ax.plot(0, 0, "o", color="white", markersize=8, zorder=10)

    # Warning zone outlines (for teaching / debugging)
    ax.add_patch(Rectangle(
        (0, -0.35), FRONT_WARNING_DISTANCE_M, 0.70,
        linewidth=0.6, edgecolor="#335533", facecolor="none", linestyle=":",
    ))
    ax.add_patch(Rectangle(
        (-1.0, -SIDE_WARNING_DISTANCE_M), 1.0, SIDE_WARNING_DISTANCE_M + 0.35,
        linewidth=0.6, edgecolor="#333355", facecolor="none", linestyle=":",
    ))
    ax.add_patch(Rectangle(
        (-1.0, 0.35), 1.0, SIDE_WARNING_DISTANCE_M,
        linewidth=0.6, edgecolor="#553333", facecolor="none", linestyle=":",
    ))

    ax._occupancy_patches = []


def clear_occupancy_patches(ax):
    for patch in getattr(ax, "_occupancy_patches", []):
        patch.remove()
    ax._occupancy_patches = []


def draw_occupancy_cells(ax):
    """Draw occupied grid cells as small squares."""
    clear_occupancy_patches(ax)

    with data_lock:
        cells = [
            (ix, iy, hits)
            for (ix, iy), hits in occupancy_grid.items()
            if hits >= OCCUPANCY_MIN_HITS
        ]

    half_cell = GRID_RESOLUTION_M / 2.0
    for ix, iy, hits in cells:
        cx, cy = cell_center_m(ix, iy)
        intensity = min(hits / 10.0, 1.0)
        color = (1.0, 0.2 + 0.5 * (1 - intensity), 0.1)
        rect = Rectangle(
            (cx - half_cell, cy - half_cell),
            GRID_RESOLUTION_M, GRID_RESOLUTION_M,
            facecolor=color, edgecolor="none", alpha=0.85,
        )
        ax.add_patch(rect)
        ax._occupancy_patches.append(rect)


def update_plot(_frame, ax, scatter_artist, warning_text_artist, status_text):
    """Animation callback — refresh points, occupancy, warnings, and status."""
    update_warning_state()

    with data_lock:
        paused = mapping_paused
        view_mode = show_occupancy_view
        n_points = len(points_xy)
        n_packets = packet_count
        n_cells = sum(1 for h in occupancy_grid.values() if h >= OCCUPANCY_MIN_HITS)
        if points_xy:
            arr = np.array([(p[0], p[1], p[2]) for p in points_xy])
            xs, ys, dists = arr[:, 0], arr[:, 1], arr[:, 2]
        else:
            xs, ys, dists = [], [], []

    if show_occupancy_view:
        scatter_artist.set_offsets(np.empty((0, 2)))
        draw_occupancy_cells(ax)
    else:
        clear_occupancy_patches(ax)
        if len(xs) > 0:
            scatter_artist.set_offsets(np.column_stack([xs, ys]))
            scatter_artist.set_array(dists)
            scatter_artist.set_clim(0, MAX_RANGE_M)
        else:
            scatter_artist.set_offsets(np.empty((0, 2)))

    # Warning banner colour
    warn = current_warning
    if "STOP" in warn:
        warn_color = "#ff2222"
    elif warn != "CLEAR":
        warn_color = "#ffaa00"
    else:
        warn_color = "#44cc44"

    warning_text_artist.set_text(warn)
    warning_text_artist.set_color(warn_color)

    nearest_text = (
        f"{closest_obstacle_m:.2f} m" if closest_obstacle_m is not None else "n/a"
    )
    view_label = "OCCUPANCY" if view_mode else "POINTS"
    pause_label = "PAUSED" if paused else "LIVE"

    status_text.set_text(
        f"Points: {n_points}  |  Cells: {n_cells}  |  Packets: {n_packets}\n"
        f"Closest: {nearest_text}  |  View: {view_label}  |  {pause_label}\n"
        f"Keys: S=save  C=clear  V=view  P/Space=pause  Q=quit"
    )

    return scatter_artist, warning_text_artist, status_text


def on_key_press(event):
    """Keyboard controls — click the plot window first."""
    global mapping_paused, show_occupancy_view, shutdown_requested

    key = event.key
    if key is None:
        return

    key_lower = key.lower()

    if key_lower in ("q", "escape"):
        print("Quit requested.")
        shutdown_requested = True
        plt.close(event.canvas.figure)
    elif key_lower == "c":
        clear_map()
    elif key_lower == "s":
        save_map(event.canvas.figure)
    elif key_lower == "p" or key == " ":
        mapping_paused = not mapping_paused
        state = "PAUSED" if mapping_paused else "RESUMED"
        print(f"Mapping {state}.")
    elif key_lower == "v":
        show_occupancy_view = not show_occupancy_view
        mode = "occupancy grid" if show_occupancy_view else "raw point cloud"
        print(f"Display mode: {mode}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global ser, shutdown_requested

    print("=" * 70)
    print("D6 AA55 Real-Time 2D LiDAR Map — Stage 1")
    print("=" * 70)
    print(f"Port   : {PORT} @ {BAUD}")
    print(f"Range  : {MIN_RANGE_CM} cm – {MAX_RANGE_CM} cm ({MAX_RANGE_M} m)")
    print(f"Grid   : {GRID_SIZE_M} m x {GRID_SIZE_M} m @ {GRID_RESOLUTION_M} m/cell")
    print()
    print("NOTE: This is real-time 2D mapping, NOT full SLAM.")
    print("      SLAM would need wheel odometry, IMU, pose tracking, loop closure.")
    print()
    print("Controls (click the plot window first):")
    print("  Q / Esc     quit")
    print("  S           save CSV + PNG")
    print("  C           clear map")
    print("  P / Space   pause / resume")
    print("  V           toggle points vs occupancy view")
    print()

    ser = open_serial_port(PORT, BAUD, TIMEOUT)

    reader_thread = threading.Thread(
        target=serial_reader_loop, args=(ser,), daemon=True,
    )
    reader_thread.start()

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 9))
    setup_axes(ax)

    scatter = ax.scatter(
        [], [], c=[], cmap="plasma", s=12, alpha=0.75, vmin=0, vmax=MAX_RANGE_M,
    )

    warning_text = fig.text(
        0.5, 0.96, "CLEAR", ha="center", va="top",
        fontsize=14, fontweight="bold", color="#44cc44",
    )
    status_text = fig.text(
        0.02, 0.02, "", color="lightgray", fontsize=9, family="monospace",
    )

    fig.canvas.mpl_connect("key_press_event", on_key_press)

    def on_close(_event):
        global shutdown_requested
        shutdown_requested = True

    fig.canvas.mpl_connect("close_event", on_close)

    anim = FuncAnimation(
        fig,
        update_plot,
        fargs=(ax, scatter, warning_text, status_text),
        interval=UPDATE_MS,
        blit=False,
        cache_frame_data=False,
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nCtrl+C received.")
    finally:
        shutdown_requested = True
        reader_thread.join(timeout=2.0)
        if ser is not None and ser.is_open:
            ser.close()
        print("Stopped safely.")


if __name__ == "__main__":
    main()
