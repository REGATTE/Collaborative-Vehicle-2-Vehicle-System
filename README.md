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

---

### To visualize the ego_vehicle sensor data

Run the above steps, and in another window with the venv

```bash
python3 main.py --vis_ego
```

## Visual Results

| Ego Vehicle | Smart Vehicle 1 | Smart Vehicle 2 | Smart Vehicle 3 | Smart Vehicle 4 |
|----------------------|---------------------|---------------------|---------------------|---------------------|
| ![Image 1](docs/Images/ego_veh.png) | ![Image 2](docs/Images/smart_veh_1.png) | ![Image 3](docs/Images/smart_veh_2.png) | ![Image 4](docs/Images/smart_veh_3.png) | ![Image 4](docs/Images/smart_veh_4.png) |

All the vehicles are autonomous and do not follow a custom path waypoint. 