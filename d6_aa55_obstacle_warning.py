import serial
import struct
import time
import os

PORT = "/dev/ttyUSB0"
BAUD = 230400

# Warning thresholds in centimetres
DANGER_CM = 60
WARNING_CM = 120

# Front detection zone:
# 330° to 360° and 0° to 30°
FRONT_LEFT = 330
FRONT_RIGHT = 30

ALERT_GAP = 1.5

ser = serial.Serial(PORT, BAUD, timeout=0.5)
last_alert_time = 0

def speak(text):
    print("ALERT:", text)
    os.system(f'espeak "{text}" 2>/dev/null &')

def is_front_angle(angle):
    return angle >= FRONT_LEFT or angle <= FRONT_RIGHT

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

        if 5 <= distance_cm <= 800:
            points.append({
                "angle_deg": angle,
                "distance_cm": distance_cm,
                "raw": raw_sample
            })

    return points

print("D6 AA55 front obstacle warning started.")
print(f"Using {PORT} at {BAUD} baud.")
speak("Obstacle warning system started")

try:
    while True:
        packet = read_packet()
        points = parse_packet(packet)

        front_distances = []

        for p in points:
            angle = p["angle_deg"]
            distance = p["distance_cm"]

            if is_front_angle(angle):
                if 20 <= distance <= 300:
                    front_distances.append(distance)

        if front_distances:
            nearest = min(front_distances)
            print(f"Nearest front obstacle: {nearest:.1f} cm")

            now = time.time()

            if now - last_alert_time > ALERT_GAP:
                if nearest <= DANGER_CM:
                    speak("Stop. Obstacle very near.")
                    last_alert_time = now

                elif nearest <= WARNING_CM:
                    speak("Obstacle ahead.")
                    last_alert_time = now

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    ser.close()
