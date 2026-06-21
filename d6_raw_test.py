import serial
import time

# Change this if your LiDAR appears as /dev/ttyACM0 or /dev/ttyUSB1
PORT = "/dev/ttyUSB0"

# Common baud rates to test
BAUD_RATES = [115200, 230400, 460800, 512000, 921600]

for baud in BAUD_RATES:
    print(f"\nTrying baud rate: {baud}")

    try:
        ser = serial.Serial(PORT, baud, timeout=1)
        time.sleep(2)

        data = ser.read(160)

        if data:
            print(f"Received {len(data)} bytes at {baud}:")
            print(data.hex(" "))
        else:
            print("No data received.")

        ser.close()

    except Exception as e:
        print(f"Error at {baud}: {e}")

print("\nDone.")
print("Your LiDAR appears to use AA 55 packets.")
print("Use the baud rate that shows repeated AA 55 patterns, likely 230400.")
