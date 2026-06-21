import serial
import struct
import math
import tkinter as tk

PORT = "/dev/ttyUSB0"
BAUD = 230400

ser = serial.Serial(PORT, BAUD, timeout=0.5)

WIDTH = 700
HEIGHT = 700
CENTER_X = WIDTH // 2
CENTER_Y = HEIGHT // 2
SCALE = 0.8  # pixels per cm

root = tk.Tk()
root.title("D6 AA55 LiDAR Visual Map")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black")
canvas.pack()

points_buffer = []

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

                packet_header = bytes([0xAA, 0x55]) + header_rest
                lsn = header_rest[1]

                if lsn <= 0 or lsn > 100:
                    return None

                sample_data = ser.read(lsn * 2)

                if len(sample_data) != lsn * 2:
                    return None

                return packet_header + sample_data

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

        if 5 <= distance_cm <= 400:
            points.append((angle, distance_cm))

    return points

def draw_map():
    global points_buffer

    packet = read_packet()
    new_points = parse_packet(packet)

    if new_points:
        points_buffer.extend(new_points)

    # Keep recent points only
    if len(points_buffer) > 1000:
        points_buffer = points_buffer[-1000:]

    canvas.delete("all")

    # Draw range circles
    for r_cm in [50, 100, 200, 300, 400]:
        r = r_cm * SCALE
        canvas.create_oval(
            CENTER_X - r, CENTER_Y - r,
            CENTER_X + r, CENTER_Y + r,
            outline="gray"
        )
        canvas.create_text(CENTER_X + 5, CENTER_Y - r, text=f"{r_cm}cm", fill="gray", anchor="nw")

    # Draw forward direction line
    canvas.create_line(CENTER_X, CENTER_Y, CENTER_X, CENTER_Y - 300, fill="white")
    canvas.create_text(CENTER_X, CENTER_Y - 320, text="FRONT 0°", fill="white")

    # Draw LiDAR center
    canvas.create_oval(CENTER_X - 6, CENTER_Y - 6, CENTER_X + 6, CENTER_Y + 6, fill="white")

    for angle_deg, distance_cm in points_buffer:
        # 0° is drawn upward/front
        rad = math.radians(angle_deg - 90)
        x = CENTER_X + math.cos(rad) * distance_cm * SCALE
        y = CENTER_Y + math.sin(rad) * distance_cm * SCALE

        canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="lime", outline="lime")

    root.after(20, draw_map)

def on_close():
    ser.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
draw_map()
root.mainloop()
