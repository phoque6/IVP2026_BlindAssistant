Run on Pi 5
sudo apt install python3-serial python3-matplotlib python3-numpy
# or: bash setup_pi5_lidar.sh
python3 d6_aa55_3d_visual_map.py
If the port differs, edit the top of the file:

PORT = "/dev/ttyUSB0"   # or /dev/ttyACM