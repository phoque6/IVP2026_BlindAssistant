import serial
import struct

PORT = "/dev/ttyUSB0"
BAUD = 230400

ser = serial.Serial(PORT, BAUD, timeout=0.5)

def read_packet():
    """
    Reads a possible AA55 / YDLIDAR-style scan packet.

    Common format:
    AA 55 | CT | LSN | FSA | LSA | CS | sample data

    Header:
    - AA 55 = packet start
    - CT = packet type
    - LSN = number of samples
    - FSA = first/start angle
    - LSA = last/end angle
    - CS = checksum
    - sample data = LSN x 2 bytes
    """
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

    # AA55/YDLIDAR-style angle conversion
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

        # Common distance conversion for AA55/YDLIDAR style
        distance_mm = raw_sample / 4.0
        distance_cm = distance_mm / 10.0

        if lsn > 1:
            angle = start_angle + angle_diff * i / (lsn - 1)
        else:
            angle = start_angle

        angle = angle % 360

        # Filter out invalid readings
        if 5 <= distance_cm <= 800:
            points.append({
                "angle_deg": angle,
                "distance_cm": distance_cm,
                "raw": raw_sample
            })

    return points

print("D6 AA55 LiDAR test started.")
print(f"Using {PORT} at {BAUD} baud.")
print("Press Ctrl+C to stop.")

try:
    while True:
        packet = read_packet()
        points = parse_packet(packet)

        for p in points:
            print(
                f"Angle: {p['angle_deg']:6.1f}° | "
                f"Distance: {p['distance_cm']:6.1f} cm | "
                f"Raw: {p['raw']}"
            )

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    ser.close()
