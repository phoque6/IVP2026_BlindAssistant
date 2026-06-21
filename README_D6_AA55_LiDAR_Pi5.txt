D6 AA55 TOF LiDAR + Raspberry Pi 5 Files
=======================================

Your raw output showed that the LiDAR is detected as:

/dev/ttyUSB0

and sends repeated AA 55 packet patterns at:

230400 baud

This ZIP uses an AA55/YDLIDAR-style parser instead of the earlier 54 2C parser.

Install dependencies:
---------------------
sudo apt update
sudo apt install python3-serial espeak python3-tk

Or run:
bash setup_pi5_lidar.sh

Recommended testing order:
--------------------------

1. Raw serial test:
python3 d6_raw_test.py

You should see repeated AA 55 patterns at 230400.

2. Distance/angle decoder test:
python3 d6_aa55_lidar_test.py

You should see:
Angle: xxx.x° | Distance: xxx.x cm | Raw: xxxx

3. Front-only obstacle warning:
python3 d6_aa55_obstacle_warning.py

This checks only the front zone:
330° to 360° and 0° to 30°

4. Front/left/right zone warning:
python3 d6_aa55_obstacle_zones.py

Approximate zones:
Front: 330° to 360° and 0° to 30°
Right: 30° to 100°
Left: 260° to 330°

5. Visual map:
python3 d6_aa55_visual_map.py

This opens a simple 2D map window.
0° is drawn as forward/up.

If the port changes:
--------------------
Run:
ls /dev/ttyUSB* /dev/ttyACM*

Then edit the PORT line in the Python files, for example:
PORT = "/dev/ttyUSB1"
or
PORT = "/dev/ttyACM0"

If the decoder prints strange distances:
----------------------------------------
The packet parser is based on your raw AA55 output and a common YDLIDAR-style format.
Some D6 models may use slight protocol differences.
If readings look wrong, send a longer raw dump from:
python3 d6_raw_test.py

Safety note:
------------
This is a student/prototype assistive obstacle detection system.
Do not rely on it as a real mobility aid without extensive testing.
LiDAR may miss glass, reflective surfaces, steps, slopes, very thin obstacles, and objects outside the scan height.
