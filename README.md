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
