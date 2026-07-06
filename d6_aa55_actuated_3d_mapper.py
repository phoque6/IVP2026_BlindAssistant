"""
Real-Time 3D Mapping in Complex Environments Using a Spinning Actuated LiDAR System
====================================================================================

D6 AA55 TOF LiDAR + tilt actuator (servo/stepper) on Raspberry Pi 5.

Install:
    sudo apt update
    sudo apt install python3-serial python3-matplotlib python3-numpy python3-gpiozero

Run:
    python3 d6_aa55_actuated_3d_mapper.py

For testing without a motor:
    SIMULATED_TILT_MODE = True

For a real servo on GPIO:
    SIMULATED_TILT_MODE = False
    SERVO_PIN = 18

Scientific background (read this first):
----------------------------------------
A normal 2D LiDAR only scans one flat horizontal plane. Every hit shares the
same vertical angle, so you get a flat "slice" of the world — not a volume.

A true 3D point cloud is built when that 2D LiDAR is tilted up and down by a
servo or stepper. At each tilt angle the sensor captures a different 2D scan.
Software combines three measurements for every hit:

    horizontal angle  (spin angle from the LiDAR)
    tilt angle        (vertical angle from the actuator)
    distance          (range to the obstacle)

Together these become a real (X, Y, Z) point in 3D space.

This script is a basic real-time 3D mapping system — not full SLAM.
If the robot or sensor moves while scanning, an IMU or wheel odometry would
be needed to correctly merge scans taken at different positions over time.
"""

import glob
import math
import struct
import threading
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

# ---------------------------------------------------------------------------
# Serial / LiDAR settings
# ---------------------------------------------------------------------------
PORT = "/dev/ttyUSB0"
BAUD = 230400

MAX_RANGE_M = 6.0
MIN_RANGE_CM = 5
MAX_RANGE_CM = int(MAX_RANGE_M * 100)

# ---------------------------------------------------------------------------
# Tilt actuator settings
# ---------------------------------------------------------------------------
SIMULATED_TILT_MODE = True   # True = software sweep, False = real GPIO servo
SERVO_PIN = 18
TILT_MIN_DEG = -30.0
TILT_MAX_DEG = 30.0
TILT_STEP_DEG = 2.0
SCANS_PER_TILT = 4           # 2D packets to collect at each tilt before moving
SERVO_SETTLE_SEC = 0.35        # wait after moving real servo before scanning

# ---------------------------------------------------------------------------
# Point cloud / voxel settings
# ---------------------------------------------------------------------------
MAX_POINTS = 30000
VOXEL_SIZE_M = 0.10
VOXEL_MIN_HITS = 2

# ---------------------------------------------------------------------------
# Display / save settings
# ---------------------------------------------------------------------------
UPDATE_MS = 100
Z_MIN_M = -2.0
Z_MAX_M = 2.0
SAVE_EVERY_SECONDS = 0         # 0 = disabled; e.g. 60 for auto-save every minute
CSV_FILENAME = "point_cloud.csv"
PLY_FILENAME = "point_cloud.ply"

# ---------------------------------------------------------------------------
# Shared state (protected by locks)
# ---------------------------------------------------------------------------
points_lock = threading.Lock()
points_xyz = []          # list of (x, y, z, distance_m, tilt_deg)
voxel_grid = {}          # (ix, iy, iz) -> hit count
packet_count = 0
current_tilt_deg = 0.0
mapping_paused = False
show_voxel_view = False
shutdown_requested = False

# Tilt sweep direction for simulated mode (+1 = increasing, -1 = decreasing)
_tilt_sweep_direction = 1

# Hardware handles (initialised in main)
ser = None
servo = None
GPIOZERO_AVAILABLE = False

try:
    from gpiozero import Servo
    GPIOZERO_AVAILABLE = True
except ImportError:
    Servo = None


# ---------------------------------------------------------------------------
# Serial helpers
# ---------------------------------------------------------------------------
def list_serial_ports():
    """Return likely LiDAR serial device paths on Linux."""
    ports = sorted(set(
        glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    ))
    return ports


def open_serial_port(port, baud):
    """
    Open the serial port with clear error messages for students.
    Returns an open serial.Serial object or raises SystemExit.
    """
    try:
        connection = serial.Serial(port, baud, timeout=0.5)
        print(f"Serial open: {port} @ {baud} baud")
        return connection
    except serial.SerialException as exc:
        print(f"ERROR: Could not open serial port '{port}': {exc}")
        available = list_serial_ports()
        if available:
            print("Available serial ports:")
            for p in available:
                print(f"  - {p}")
            print("Edit PORT at the top of this file to match your LiDAR.")
        else:
            print("No /dev/ttyUSB* or /dev/ttyACM* ports found.")
            print("Check USB cable and run:  ls /dev/ttyUSB* /dev/ttyACM*")
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# AA55 packet reading and parsing
# ---------------------------------------------------------------------------
def read_packet(connection):
    """
    Search for an AA55 header and read one complete scan packet.

    Packet layout (YDLIDAR / AA55 style):
        AA 55 | CT | LSN | FSA | LSA | CS | sample data (LSN x 2 bytes)

    Returns raw bytes or None on timeout / incomplete packet.
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
    Decode one AA55 packet into a list of (horizontal_angle_deg, distance_cm).

    Invalid or out-of-range readings are filtered out.
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

        # Raw sample -> millimetres -> centimetres
        distance_cm = (raw_sample / 4.0) / 10.0

        if lsn > 1:
            horizontal_angle = start_angle + angle_diff * i / (lsn - 1)
        else:
            horizontal_angle = start_angle

        horizontal_angle = horizontal_angle % 360.0

        if MIN_RANGE_CM <= distance_cm <= MAX_RANGE_CM:
            points.append((horizontal_angle, distance_cm))

    return points


# ---------------------------------------------------------------------------
# 3D geometry
# ---------------------------------------------------------------------------
def spherical_to_xyz(horizontal_angle_deg, tilt_angle_deg, distance_cm):
    """
    Convert horizontal angle + tilt angle + distance into 3D Cartesian metres.

    Coordinate system:
        X = forward
        Y = right
        Z = up
        0° horizontal  = forward
        90° horizontal = right
        +tilt          = LiDAR tilted upward
        -tilt          = LiDAR tilted downward
    """
    distance_m = distance_cm / 100.0
    horizontal_rad = math.radians(horizontal_angle_deg)
    tilt_rad = math.radians(tilt_angle_deg)

    xy_distance = distance_m * math.cos(tilt_rad)
    x = xy_distance * math.cos(horizontal_rad)
    y = xy_distance * math.sin(horizontal_rad)
    z = distance_m * math.sin(tilt_rad)

    return x, y, z


def voxel_index(x, y, z):
    """Return integer voxel cell indices for a 3D point."""
    return (
        int(x / VOXEL_SIZE_M),
        int(y / VOXEL_SIZE_M),
        int(z / VOXEL_SIZE_M),
    )


def add_points_to_cloud(scan_points, tilt_deg):
    """Thread-safe: convert a 2D scan at the given tilt into 3D points and voxels."""
    global points_xyz, voxel_grid, packet_count

    new_entries = []
    for horizontal_angle, distance_cm in scan_points:
        x, y, z = spherical_to_xyz(horizontal_angle, tilt_deg, distance_cm)
        distance_m = distance_cm / 100.0
        new_entries.append((x, y, z, distance_m, tilt_deg))

        key = voxel_index(x, y, z)
        voxel_grid[key] = voxel_grid.get(key, 0) + 1

    with points_lock:
        points_xyz.extend(new_entries)
        if len(points_xyz) > MAX_POINTS:
            points_xyz = points_xyz[-MAX_POINTS:]
        packet_count += 1


def clear_map():
    """Clear the point cloud and voxel grid."""
    global points_xyz, voxel_grid, packet_count
    with points_lock:
        points_xyz = []
        voxel_grid = {}
        packet_count = 0
    print("Map cleared.")


def get_display_points():
    """
    Return (xs, ys, zs, distances) for the current view mode.

    Raw mode: every stored point.
    Voxel mode: centre of each occupied voxel with enough hits.
    """
    with points_lock:
        if not show_voxel_view:
            if not points_xyz:
                return [], [], [], []
            arr = np.array([(p[0], p[1], p[2], p[3]) for p in points_xyz])
            return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        centres = []
        for (ix, iy, iz), hits in voxel_grid.items():
            if hits < VOXEL_MIN_HITS:
                continue
            cx = (ix + 0.5) * VOXEL_SIZE_M
            cy = (iy + 0.5) * VOXEL_SIZE_M
            cz = (iz + 0.5) * VOXEL_SIZE_M
            dist = math.sqrt(cx * cx + cy * cy + cz * cz)
            centres.append((cx, cy, cz, dist))

        if not centres:
            return [], [], [], []
        arr = np.array(centres)
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


# ---------------------------------------------------------------------------
# Save / export
# ---------------------------------------------------------------------------
def save_point_cloud():
    """Export the current map to CSV and ASCII PLY."""
    with points_lock:
        snapshot = list(points_xyz)

    if not snapshot:
        print("Nothing to save — point cloud is empty.")
        return

    # CSV
    with open(CSV_FILENAME, "w", encoding="utf-8") as csv_file:
        csv_file.write("x,y,z,distance_m,tilt_deg\n")
        for x, y, z, dist_m, tilt in snapshot:
            csv_file.write(f"{x:.5f},{y:.5f},{z:.5f},{dist_m:.5f},{tilt:.2f}\n")

    # ASCII PLY (opens in MeshLab, CloudCompare, Blender)
    with open(PLY_FILENAME, "w", encoding="utf-8") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(snapshot)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")
        for x, y, z, _dist_m, _tilt in snapshot:
            ply_file.write(f"{x:.5f} {y:.5f} {z:.5f}\n")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] Saved {len(snapshot)} points -> {CSV_FILENAME}, {PLY_FILENAME}")


# ---------------------------------------------------------------------------
# Tilt actuator (simulated or real servo)
# ---------------------------------------------------------------------------
def tilt_to_servo_value(tilt_deg):
    """Map tilt degrees to gpiozero Servo value in range -1 .. +1."""
    span = TILT_MAX_DEG - TILT_MIN_DEG
    if span <= 0:
        return 0.0
    normalised = (tilt_deg - TILT_MIN_DEG) / span
    return normalised * 2.0 - 1.0


def init_servo():
    """Create a gpiozero Servo on SERVO_PIN, or return None in simulated mode."""
    global servo

    if SIMULATED_TILT_MODE:
        print("Tilt mode: SIMULATED (no GPIO servo)")
        return None

    if not GPIOZERO_AVAILABLE:
        print("ERROR: gpiozero is not installed.")
        print("Install with:  sudo apt install python3-gpiozero")
        raise SystemExit(1)

    try:
        servo = Servo(SERVO_PIN, min_pulse_width=0.5 / 1000, max_pulse_width=2.5 / 1000)
        print(f"Tilt mode: REAL SERVO on GPIO {SERVO_PIN}")
        return servo
    except Exception as exc:
        print(f"ERROR: Could not initialise servo on GPIO {SERVO_PIN}: {exc}")
        raise SystemExit(1) from exc


def set_tilt(tilt_deg):
    """Move the servo (or update simulated tilt variable)."""
    global current_tilt_deg

    tilt_deg = max(TILT_MIN_DEG, min(TILT_MAX_DEG, tilt_deg))
    current_tilt_deg = tilt_deg

    if SIMULATED_TILT_MODE:
        return

    if servo is not None:
        servo.value = tilt_to_servo_value(tilt_deg)
        time.sleep(SERVO_SETTLE_SEC)


def stop_servo():
    """Release the servo safely on exit."""
    if servo is not None:
        try:
            servo.detach()
            servo.close()
            print("Servo detached and closed.")
        except Exception as exc:
            print(f"WARNING: Servo shutdown issue: {exc}")


def next_simulated_tilt():
    """
    Advance simulated tilt by TILT_STEP_DEG, bouncing between min and max.
    Returns the new tilt angle.
    """
    global current_tilt_deg, _tilt_sweep_direction

    next_tilt = current_tilt_deg + _tilt_sweep_direction * TILT_STEP_DEG

    if next_tilt >= TILT_MAX_DEG:
        next_tilt = TILT_MAX_DEG
        _tilt_sweep_direction = -1
    elif next_tilt <= TILT_MIN_DEG:
        next_tilt = TILT_MIN_DEG
        _tilt_sweep_direction = 1

    current_tilt_deg = next_tilt
    return current_tilt_deg


def init_tilt_sweep():
    """Start the tilt sweep at the minimum angle."""
    global _tilt_sweep_direction
    _tilt_sweep_direction = 1
    set_tilt(TILT_MIN_DEG)


# ---------------------------------------------------------------------------
# Background mapping thread
# ---------------------------------------------------------------------------
def mapping_loop(connection):
    """
    Main acquisition loop: set tilt angle, read 2D scans, build 3D cloud.

    In simulated mode the tilt sweeps automatically.
    In servo mode the GPIO servo moves to each tilt before scanning.
    """
    global shutdown_requested, mapping_paused

    init_tilt_sweep()
    print("Mapping thread started.")

    while not shutdown_requested:
        if mapping_paused:
            time.sleep(0.05)
            continue

        # Collect several 2D packets at the current tilt angle
        scans_collected = 0
        while scans_collected < SCANS_PER_TILT and not shutdown_requested:
            if mapping_paused:
                break

            packet = read_packet(connection)
            scan_points = parse_packet(packet)

            if scan_points:
                with points_lock:
                    tilt_now = current_tilt_deg
                add_points_to_cloud(scan_points, tilt_now)
                scans_collected += 1

        # Move to the next tilt angle
        if SIMULATED_TILT_MODE:
            next_simulated_tilt()
        else:
            # Step through tilt range in SERVO mode
            with points_lock:
                tilt_now = current_tilt_deg
            next_tilt = tilt_now + TILT_STEP_DEG
            if next_tilt > TILT_MAX_DEG:
                next_tilt = TILT_MIN_DEG
            set_tilt(next_tilt)

        time.sleep(0.01)

    print("Mapping thread stopped.")


# ---------------------------------------------------------------------------
# 3D visualisation
# ---------------------------------------------------------------------------
def make_range_ring(radius_m, z=0.0, segments=72):
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    x = radius_m * np.cos(angles)
    y = radius_m * np.sin(angles)
    z_arr = np.full_like(x, z)
    return x, y, z_arr


def setup_axes(ax):
    ax.set_xlim(-MAX_RANGE_M, MAX_RANGE_M)
    ax.set_ylim(-MAX_RANGE_M, MAX_RANGE_M)
    ax.set_zlim(Z_MIN_M, Z_MAX_M)
    ax.set_xlabel("Forward X (m)")
    ax.set_ylabel("Right Y (m)")
    ax.set_zlabel("Height Z (m)")
    ax.set_title("Actuated LiDAR — Real-Time 3D Map")
    ax.view_init(elev=25, azim=-55)

    grid_step = 1.0
    grid_vals = np.arange(-MAX_RANGE_M, MAX_RANGE_M + grid_step, grid_step)
    for gv in grid_vals:
        ax.plot([-MAX_RANGE_M, MAX_RANGE_M], [gv, gv], [0, 0],
                color="#333333", linewidth=0.4)
        ax.plot([gv, gv], [-MAX_RANGE_M, MAX_RANGE_M], [0, 0],
                color="#333333", linewidth=0.4)

    for radius_m in (1, 2, 3, 4, 5, MAX_RANGE_M):
        rx, ry, rz = make_range_ring(radius_m)
        ax.plot(rx, ry, rz, color="#555555", linewidth=0.8, alpha=0.7)

    ax.plot([0, MAX_RANGE_M * 0.35], [0, 0], [0, 0], color="white", linewidth=2)
    ax.text(MAX_RANGE_M * 0.36, 0, 0, "FRONT 0°", color="white", fontsize=8)
    ax.scatter([0], [0], [0], color="white", s=45, depthshade=False)


def update_plot(_frame, scatter_artist, status_text, last_save_holder):
    xs, ys, zs, dists = get_display_points()

    if len(xs) > 0:
        scatter_artist._offsets3d = (xs, ys, zs)
        scatter_artist.set_array(dists)
        scatter_artist.set_clim(0, MAX_RANGE_M)
        scatter_artist.set_sizes(np.full(len(xs), 14 if show_voxel_view else 6))
    else:
        scatter_artist._offsets3d = ([], [], [])

    with points_lock:
        n_points = len(points_xyz)
        n_packets = packet_count
        tilt = current_tilt_deg
        n_voxels = sum(1 for h in voxel_grid.values() if h >= VOXEL_MIN_HITS)

    mode = "SIMULATED" if SIMULATED_TILT_MODE else f"GPIO {SERVO_PIN}"
    view = "VOXELS" if show_voxel_view else "RAW"
    paused = "PAUSED" if mapping_paused else "MAPPING"

    status_text.set_text(
        f"Points: {n_points}  |  Packets: {n_packets}  |  Voxels: {n_voxels}\n"
        f"Tilt: {tilt:+.1f}°  |  Mode: {mode}  |  View: {view}  |  {paused}\n"
        f"Port: {PORT}  |  Range: {MAX_RANGE_M} m  |  "
        f"Keys: S=save  C=clear  V=view  Space=pause  Q=quit"
    )

    # Timed auto-save
    if SAVE_EVERY_SECONDS > 0:
        now = time.time()
        if now - last_save_holder[0] >= SAVE_EVERY_SECONDS:
            save_point_cloud()
            last_save_holder[0] = now

    return scatter_artist, status_text


def on_key_press(event):
    """Keyboard controls for the matplotlib window."""
    global mapping_paused, show_voxel_view, shutdown_requested

    key = event.key
    if key is None:
        return

    key_lower = key.lower()

    if key_lower in ("q", "escape"):
        print("Quit requested.")
        shutdown_requested = True
        plt.close(event.canvas.figure)
    elif key_lower == "s":
        save_point_cloud()
    elif key_lower == "c":
        clear_map()
    elif key == " ":
        mapping_paused = not mapping_paused
        state = "PAUSED" if mapping_paused else "RESUMED"
        print(f"Mapping {state}.")
    elif key_lower == "v":
        show_voxel_view = not show_voxel_view
        mode = "voxel-filtered" if show_voxel_view else "raw point cloud"
        print(f"Display mode: {mode}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global ser, shutdown_requested

    print("=" * 72)
    print("Real-Time 3D Mapping — Spinning Actuated LiDAR System")
    print("=" * 72)
    print(f"LiDAR port : {PORT} @ {BAUD}")
    print(f"Range      : {MIN_RANGE_CM} cm – {MAX_RANGE_CM} cm ({MAX_RANGE_M} m)")
    print(f"Tilt range : {TILT_MIN_DEG}° to {TILT_MAX_DEG}°  (step {TILT_STEP_DEG}°)")
    print(f"Motor mode : {'SIMULATED' if SIMULATED_TILT_MODE else f'GPIO servo pin {SERVO_PIN}'}")
    print()
    print("Controls (click the plot window first):")
    print("  Q / Esc  = quit")
    print("  S        = save point_cloud.csv + point_cloud.ply")
    print("  C        = clear map")
    print("  Space    = pause / resume mapping")
    print("  V        = toggle raw points vs voxel-filtered view")
    print()

    init_servo()
    ser = open_serial_port(PORT, BAUD)

    mapper_thread = threading.Thread(
        target=mapping_loop, args=(ser,), daemon=True,
    )
    mapper_thread.start()

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    setup_axes(ax)

    point_size = 14 if show_voxel_view else 6
    scatter = ax.scatter(
        [], [], [],
        c=[], cmap="plasma", s=point_size,
        alpha=0.85, depthshade=True,
    )

    status_text = fig.text(
        0.02, 0.02, "",
        color="lightgray", fontsize=9, family="monospace",
    )

    last_save_holder = [time.time()]
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    anim = FuncAnimation(
        fig,
        update_plot,
        fargs=(scatter, status_text, last_save_holder),
        interval=UPDATE_MS,
        blit=False,
        cache_frame_data=False,
    )

    def on_close(_event):
        global shutdown_requested
        shutdown_requested = True

    fig.canvas.mpl_connect("close_event", on_close)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nCtrl+C received.")
    finally:
        shutdown_requested = True
        mapper_thread.join(timeout=2.0)
        stop_servo()
        if ser is not None and ser.is_open:
            ser.close()
            print("Serial port closed.")
        print("Program stopped.")


if __name__ == "__main__":
    main()
