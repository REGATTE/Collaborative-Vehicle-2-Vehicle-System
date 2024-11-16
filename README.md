# Collaborative-Vehicle-2-Vehicle-System
Vehicle to vehicle communication systems, for intelligent collaborative driving. 

## Setup

```bash
git clone https://github.com/REGATTE/Collaborative-Vehicle-2-Vehicle-System.git
```

## Simulation

In the first terminal

```bash
cd carla_simulator
source {carla_env}/bin/activate

./CarlaUE4.sh
```

In another terminal, with the venv activated

```bash
cd /path/to/Collaborative-Vehicle-2-Vehicle-System.git
python3 main.py
```

## To-Do

- [x] Task: Launch carla with multiple vehicles [1 ego-vehicle, 4 smart vehicles with the same sensor suite, press tab to shift between vehicles]
- [] Task: Designate the vehicles with labels to determine which is which vehicles [`main.py -> designate_ego_and_smart_vehicles() edit this]
- [] Task: Increase traffic vehicles, right now there is minimal NPC vehicles.
- [] Task: Add functions to visualize sensor data in another pygame window.
- [] Task: Process sensor data to view bounding boxes.
- [] Task: Write function to find which vehicle is in close proximity.