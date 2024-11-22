# Carla Error Management

## If carla doesnt launch

If carla doesnt launch, when you run `./CarlaUE4.sh`, and you get the following error:

```bash
4.26.2-0+++UE4+Release-4.26 522 0
Disabling core dumps.
LowLevelFatalError [File:Unknown] [Line: 136] 
Exception thrown: bind: Address already in use
Signal 11 caught.
Malloc Size=65538 LargeMemoryPoolOffset=65554 
CommonUnixCrashHandler: Signal=11
Malloc Size=131160 LargeMemoryPoolOffset=196744 
Malloc Size=131160 LargeMemoryPoolOffset=327928 
Engine crash handling finished; re-raising signal 11 for the default handler. Good bye.
Segmentation fault (core dumped)
```

Fix:

run 

```bash
ps aux | grep carla
```

this will give an output like this

```bash
reggast+    9288  142 11.8 8804880 3851872 ?     Rl   20:33   5:04 /home/reggastation/carla_simulator/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4
reggast+   10043  0.0  0.0   9040   660 pts/0    S+   20:36   0:00 grep --color=auto carla
```

then run this with your PID generated

```bash
sudo kill -9 9288
```

This will kill the carla background process. You can launch carla again by

```bash
./CarlaUE4.sh
```