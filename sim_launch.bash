#!/bin/bash

# Define the user's password (replace YOUR_PASSWORD with the actual password)
SUDO_PASSWORD="YOUR_PASSWORD"

# Kill any existing CARLA processes
CARLA_PROCESS=$(ps aux | grep '[C]arlaUE4-Linux-Shipping' | awk '{print $2}')
if [ ! -z "$CARLA_PROCESS" ]; then
    echo "Terminating existing CARLA processes..."
    echo $SUDO_PASSWORD | sudo -S kill -9 $CARLA_PROCESS
fi

cd

# Start the first terminal and run CARLA simulator
gnome-terminal -- bash -c "
cd carla_simulator/
source carla-env/bin/activate
./CarlaUE4.sh
exec bash
" &  # Run in background
wait $!  # Wait for the terminal process to start
sleep 10  # Wait for 10 seconds

# Start the second terminal and run the main Python script
gnome-terminal -- bash -c "
cd carla_simulator/
source carla-env/bin/activate
cd ~/Desktop/Self-Driving-Stack/Project/Collaborative-Vehicle-2-Vehicle-System/
python3 main.py
exec bash
" &  # Run in background
wait $!  # Wait for the terminal process to start
sleep 10  # Wait for 10 seconds

# Start the third terminal and run the visualization scripts
gnome-terminal -- bash -c "
cd carla_simulator/
source carla-env/bin/activate
cd ~/Desktop/Self-Driving-Stack/Project/Collaborative-Vehicle-2-Vehicle-System/agents/
python3 visualize_ego.py
python3 visualize_ego.py
exec bash
" &  # Run in background
wait $!  # Wait for the terminal process to start
sleep 10  # Wait for 10 seconds
