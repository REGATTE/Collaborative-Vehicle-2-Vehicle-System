# Collaborative-Vehicle-2-Vehicle-System
Vehicle to vehicle communication systems, for intelligent collaborative driving. 

## Setup - Pyenv

Creating environment and installing dependencies

```bash
git clone https://github.com/REGATTE/Collaborative-Vehicle-2-Vehicle-System.git
cd Collaborative-Vehicle-2-Vehicle-System
python3 -m venv .cvs
source .cvs/bin/activate
python3 setup.py
pip3 install -r requirements.txt
```

## Setup - conda

Creating conda environment.

```bash
conda env create -f environment.yml
```

The above command creates a conda environment named cvs and installs the packages that are necessary. 

## Simulation

### Launch with a bash file

update the file [sim_launch](sim_launch.bash) with your password and path to the project. 

run 

```bash
./sim_launch.bash
```

### Launch manually

If you need to install carla, follow these steps - [Install Carla](docs/installation.md)

In the first terminal

```bash
cd carla_simulator
source {carla_env}/bin/activate

./CarlaUE4.sh
```

In another terminal, with the venv activated

```bash
cd /path/to/Collaborative-Vehicle-2-Vehicle-System
python3 main.py
```

To visualize **ego_vehicle** data, in another terminal [with the venv activated]

```bash
cd /path/to/Collaborative-Vehicle-2-Vehicle-System/agents
python3 visualize_ego.py
```

## Visual Results

| Ego Vehicle | Smart Vehicle 1 | Smart Vehicle 2 | Smart Vehicle 3 | Smart Vehicle 4 |
|----------------------|---------------------|---------------------|---------------------|---------------------|
| ![Image 1](docs/Images/ego_veh.png) | ![Image 2](docs/Images/smart_veh_1.png) | ![Image 3](docs/Images/smart_veh_2.png) | ![Image 4](docs/Images/smart_veh_3.png) | ![Image 4](docs/Images/smart_veh_4.png) |

All the vehicles are autonomous and do not follow a custom path waypoint. 

Ego Car BEV Visualisation

![EGO Car BEV Viz](docs/Images/EGO_Car_BEV.png)

Combined data LiDAR View

![Combined lidar view](docs/Images/combined_lidar_frame.png)

## Errors 

we have compiled all the errors that we have faced, and how to fix them. The instructions can be found [here](docs/CarlaErrors.md)

## License

This project is licensed under the [Apache License](LICENSE).

## Collaborators

- [REGATTE](https://github.com/REGATTE)
- [bundle-adjuster](https://github.com/bundle-adjuster)
- [minnakan](https://github.com/minnakan)