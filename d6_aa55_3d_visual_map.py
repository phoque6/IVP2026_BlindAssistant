"""
D6 AA55 TOF LiDAR — 3D surroundings map for Raspberry Pi 5.

Reads AA55 scan packets over serial, converts hits to 3D Cartesian points,
and displays an interactive 3D diagram of obstacles within MAX_RANGE_M metres.

Note: the D6 is a single-plane (2D) spinner. This script builds a 3D view by
placing scan hits on the ground plane and drawing vertical pillars at each hit
so walls and furniture read clearly in 3D. True volumetric sensing would need a
3D sensor or a tilted/moving 2D unit.

Install on Pi 5:
    sudo apt install python3-serial python3-matplotlib python3-numpy

Run:
    python3 d6_aa55_3d_visual_map.py
"""

import math
import struct
import threading

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

PORT = "/dev/ttyUSB0"
BAUD = 230400

MAX_RANGE_M = 6.0
MIN_RANGE_CM = 5
MAX_RANGE_CM = int(MAX_RANGE_M * 100)

# Display tuning (Pi 5 friendly defaults)
UPDATE_MS = 80
MAX_POINTS = 2500
WALL_HEIGHT_M = 1.8
PILLAR_SAMPLE_STEP = 3  # draw every Nth point as a vertical pillar

ser = serial.Serial(PORT, BAUD, timeout=0.5)

points_lock = threading.Lock()
points_xyz = []
scan_count = 0


def read_packet():
    while True:
        b = ser.read(1)
        if not b:
            return None

        if b[0] == 0xAA:
            second = ser.read(1)
            if second and second[0] == 0x55:
                header_rest = ser.read(8)
                if len(header_rest) != 8:
                    return None

                lsn = header_rest[1]
                if lsn <= 0 or lsn > 100:
                    return None

                sample_data = ser.read(lsn * 2)
                if len(sample_data) != lsn * 2:
                    return None

                return bytes([0xAA, 0x55]) + header_rest + sample_data


def parse_packet(packet):
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
            angle = start_angle + angle_diff * i / (lsn - 1)
        else:
            angle = start_angle

        angle = angle % 360

        if MIN_RANGE_CM <= distance_cm <= MAX_RANGE_CM:
            points.append((angle, distance_cm))

    return points


def polar_to_xyz(angle_deg, distance_cm):
    """
    Convert polar scan to metres.
    0° = forward (+X), 90° = right (+Y), Z = up.
    """
    distance_m = distance_cm / 100.0
    rad = math.radians(angle_deg)
    x = distance_m * math.cos(rad)
    y = distance_m * math.sin(rad)
    z = 0.0
    return x, y, z


def serial_reader_loop():
    global points_xyz, scan_count

    while True:
        packet = read_packet()
        new_points = parse_packet(packet)
        if not new_points:
            continue

        new_xyz = [polar_to_xyz(a, d) for a, d in new_points]

        with points_lock:
            points_xyz.extend(new_xyz)
            if len(points_xyz) > MAX_POINTS:
                points_xyz = points_xyz[-MAX_POINTS:]
            scan_count += 1


def make_range_ring(axis_max, z=0.0, segments=72):
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    x = axis_max * np.cos(angles)
    y = axis_max * np.sin(angles)
    z_arr = np.full_like(x, z)
    return x, y, z_arr


def setup_axes(ax):
    ax.set_xlim(-MAX_RANGE_M, MAX_RANGE_M)
    ax.set_ylim(-MAX_RANGE_M, MAX_RANGE_M)
    ax.set_zlim(0, WALL_HEIGHT_M + 0.3)
    ax.set_xlabel("Forward X (m)")
    ax.set_ylabel("Right Y (m)")
    ax.set_zlabel("Height Z (m)")
    ax.set_title(f"D6 LiDAR 3D map — {MAX_RANGE_M:.0f} m range")
    ax.view_init(elev=28, azim=-60)

    # Ground grid
    grid_step = 1.0
    grid_vals = np.arange(-MAX_RANGE_M, MAX_RANGE_M + grid_step, grid_step)
    for gv in grid_vals:
        ax.plot([-MAX_RANGE_M, MAX_RANGE_M], [gv, gv], [0, 0], color="#333333", linewidth=0.4)
        ax.plot([gv, gv], [-MAX_RANGE_M, MAX_RANGE_M], [0, 0], color="#333333", linewidth=0.4)

    # Range rings on the ground plane
    for radius_m in (1, 2, 3, 4, 5, MAX_RANGE_M):
        rx, ry, rz = make_range_ring(radius_m)
        ax.plot(rx, ry, rz, color="#555555", linewidth=0.8, alpha=0.7)

    # Forward marker
    ax.plot([0, MAX_RANGE_M * 0.35], [0, 0], [0, 0], color="white", linewidth=2)
    ax.text(MAX_RANGE_M * 0.36, 0, 0, "FRONT 0°", color="white", fontsize=8)

    # Sensor origin
    ax.scatter([0], [0], [0], color="white", s=40, depthshade=False)
    ax._pillar_lines = []


def clear_pillars(ax):
    for line in getattr(ax, "_pillar_lines", []):
        line.remove()
    ax._pillar_lines = []


def draw_pillars(ax, pts):
    clear_pillars(ax)
    if not pts:
        return

    arr = np.array(pts, dtype=float)
    for i in range(0, len(arr), PILLAR_SAMPLE_STEP):
        x, y, _ = arr[i]
        dist = math.sqrt(x * x + y * y)
        height = WALL_HEIGHT_M * (1.0 - min(dist / MAX_RANGE_M, 1.0) * 0.35)
        (line,) = ax.plot(
            [x, x], [y, y], [0, height],
            color="#00ff88", linewidth=0.6, alpha=0.35,
        )
        ax._pillar_lines.append(line)


def update_plot(_frame, ax, scatter_artist, status_text):
    with points_lock:
        pts = list(points_xyz)
        scans = scan_count

    if pts:
        arr = np.array(pts, dtype=float)
        xs, ys, zs = arr[:, 0], arr[:, 1], arr[:, 2]
        scatter_artist._offsets3d = (xs, ys, zs)

        dist = np.sqrt(xs * xs + ys * ys)
        scatter_artist.set_array(dist)
        scatter_artist.set_clim(0, MAX_RANGE_M)
    else:
        scatter_artist._offsets3d = ([], [], [])

    draw_pillars(ax, pts)

    status_text.set_text(f"Points: {len(pts)}  |  Packets: {scans}")
    return scatter_artist, status_text


def main():
    print("D6 AA55 3D visual map")
    print(f"Port: {PORT} @ {BAUD} baud")
    print(f"Range: {MIN_RANGE_CM} cm – {MAX_RANGE_CM} cm ({MAX_RANGE_M} m)")
    print("Close the plot window or press Ctrl+C to stop.")

    reader = threading.Thread(target=serial_reader_loop, daemon=True)
    reader.start()

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    setup_axes(ax)

    scatter = ax.scatter(
        [], [], [],
        c=[],
        cmap="plasma",
        s=8,
        alpha=0.85,
        depthshade=True,
    )

    status_text = fig.text(
        0.02, 0.02, "Points: 0  |  Packets: 0",
        color="lightgray", fontsize=9,
    )

    anim = FuncAnimation(
        fig,
        update_plot,
        fargs=(ax, scatter, status_text),
        interval=UPDATE_MS,
        blit=False,
        cache_frame_data=False,
    )

    def on_close(_event):
        ser.close()
        print("Serial closed.")

    fig.canvas.mpl_connect("close_event", on_close)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if ser.is_open:
            ser.close()
        print("Stopped.")


if __name__ == "__main__":
    main()
