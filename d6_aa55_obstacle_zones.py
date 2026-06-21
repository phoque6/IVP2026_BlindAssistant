import serial
import struct
import time
import os

PORT = "/dev/ttyUSB0"
BAUD = 230400

DANGER_CM = 60
WARNING_CM = 120
ALERT_GAP = 1.5

ser = serial.Serial(PORT, BAUD, timeout=0.5)
last_alert_time = 0

def speak(text):
    print("ALERT:", text)
    os.system(f'espeak "{text}" 2>/dev/null &')

def zone_for_angle(angle):
    """
    Approximate zones. You may need to adjust depending on how the LiDAR is mounted.
    This assumes 0°/360° is forward.
    """
    if angle >= 330 or angle <= 30:
        return "front"
    elif 30 < angle <= 100:
        return "right"
    elif 260 <= angle < 330:
        return "left"
    else:
        return None

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

print("D6 AA55 obstacle zone warning started.")
print(f"Using {PORT} at {BAUD} baud.")
speak("Obstacle zone warning started")

try:
    while True:
        packet = read_packet()
        points = parse_packet(packet)

        nearest = {
            "front": None,
            "left": None,
            "right": None
        }

        for p in points:
            angle = p["angle_deg"]
            distance = p["distance_cm"]

            if not (20 <= distance <= 300):
                continue

            zone = zone_for_angle(angle)

            if zone:
                if nearest[zone] is None or distance < nearest[zone]:
                    nearest[zone] = distance

        print(
            f"Front: {nearest['front']} cm | "
            f"Left: {nearest['left']} cm | "
            f"Right: {nearest['right']} cm"
        )

        now = time.time()

        if now - last_alert_time > ALERT_GAP:
            danger_zones = [
                zone for zone, dist in nearest.items()
                if dist is not None and dist <= DANGER_CM
            ]

            warning_zones = [
                zone for zone, dist in nearest.items()
                if dist is not None and DANGER_CM < dist <= WARNING_CM
            ]

            if "front" in danger_zones:
                speak("Stop. Obstacle very near in front.")
                last_alert_time = now
            elif "left" in danger_zones:
                speak("Obstacle very near on the left.")
                last_alert_time = now
            elif "right" in danger_zones:
                speak("Obstacle very near on the right.")
                last_alert_time = now
            elif "front" in warning_zones:
                speak("Obstacle ahead.")
                last_alert_time = now
            elif "left" in warning_zones:
                speak("Obstacle left.")
                last_alert_time = now
            elif "right" in warning_zones:
                speak("Obstacle right.")
                last_alert_time = now

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    ser.close()
