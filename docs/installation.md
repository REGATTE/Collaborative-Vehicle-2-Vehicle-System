# Installation for Carla

Referenced from [Linux Build](https://carla.readthedocs.io/en/latest/build_linux/)
---

## System Requirements

OS: Ubuntu 20.04
RAM: 16gb (recommended)
VRAM: 8gb (minimum)
Disk Space: 30gb (minimum)

Python 3.7

---

## Package Install - Steps

1. Install the carla tar.gz package from the git repository, for Ubuntu.

[Carla v0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15/)

2. Extract the files

```bash
tar -xvzf CARLA_0.9.15.tar.gz
```

3. move the package to the Import folder and run the following script to extract the contents:

first set up a virtual-env

```bash
python3 -m venv carla_venv
```

```bash
cd path/to/carla/root
./ImportAssets.sh
```

4. Build Carla Python Wheel


```bash
cd PythonAPI/carla/dist/
pip3 install <wheel-file-name>.whl
```

5.  Run Carla

```bash
source carla_venv/bin/activate
./CarlaUE4.sh
```