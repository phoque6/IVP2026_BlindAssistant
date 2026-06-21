#!/bin/bash

echo "Updating Raspberry Pi packages..."
sudo apt update

echo "Installing Python serial support and espeak..."
sudo apt install -y python3-serial espeak python3-tk

echo "Checking USB serial devices..."
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null

echo ""
echo "Setup complete."
echo ""
echo "Testing order:"
echo "1. python3 d6_raw_test.py"
echo "2. python3 d6_aa55_lidar_test.py"
echo "3. python3 d6_aa55_obstacle_warning.py"
echo "4. python3 d6_aa55_obstacle_zones.py"
echo "5. python3 d6_aa55_visual_map.py"
