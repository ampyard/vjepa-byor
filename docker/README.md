# Running with ROS 2 (Docker)

The container is `ros:jazzy-ros-base` (Ubuntu 24.04) plus this repository's uv workspace. Jazzy and the workspace both use Python 3.12, so one interpreter imports `rclpy`, torch and the `vjepa_byor` packages. It works on any machine that runs Docker, with the GPU wherever Docker can reach an NVIDIA GPU.

## Which setup do I need?

- **Ubuntu 24.04:** you can skip Docker. Install ROS 2 Jazzy natively ([install guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)), run `source /opt/ros/jazzy/setup.bash`, and create the workspace venv with the system packages visible so it can import `rclpy`: `uv venv --system-site-packages --python /usr/bin/python3`, then `uv sync`. The numpy 2 note under [Known issues](#known-issues) still applies.
- **Any other Linux, including Ubuntu 22.04 and 26.04:** use this container. Jazzy does not support 26.04, and 22.04's ROS 2 (Humble) is built for Python 3.10 while this workspace needs 3.11 or later.
- **Windows:** use this container under WSL 2 with Docker Desktop, which passes NVIDIA GPUs through to `--gpus all`. Run the commands below from the WSL shell.
- **macOS:** the container runs, but without a GPU: start it with `GPUS= docker/run.sh`. The planner falls back to the CPU and takes several seconds per decision, which is fine for ROS 2 development but too slow for the demo.

Tested so far: the container on Ubuntu 26.04 with an RTX 5060 Ti, with and without the GPU. The native Ubuntu 24.04, Windows and macOS routes are untested.

## Prerequisites

- Docker, with your user in the `docker` group
- An NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Check with `docker run --rm --gpus all ubuntu nvidia-smi`
- The V-JEPA 2-AC weights in the torch.hub cache, `~/.cache/torch/hub` (the same place the host uses). For a different location set `TORCH_HUB=/path/to/hub`
- Fine-tuned checkpoints in `checkpoints/` (for example `checkpoints/sim-10k`), as on the host

## Build

From the repository root:

```bash
docker build -f docker/Dockerfile -t vjepa-ros2:jazzy .
```

The first build downloads torch and the CUDA libraries (several GB) and takes 10–15 minutes. Rebuild only after changing dependencies (`pyproject.toml`, `uv.lock`, a package's `pyproject.toml`, or the Dockerfile). Source code changes need no rebuild: the repository is mounted into the container and the packages are installed in editable mode.

## Run

```bash
docker/run.sh                                          # interactive shell, ROS already sourced
docker/run.sh ros2 topic list                          # one command
docker/run.sh python apps/planner/sim_planner.py run   # the planner on the sim, on the GPU
```

A second shell in a running container:

```bash
docker ps                                  # find the container id
docker exec -it <container-id> bash
```

Each `docker/run.sh` starts its own container, and they find each other through DDS, so a second `docker/run.sh` works as a second terminal too.

### What `run.sh` sets up

| Setting | Why |
|---|---|
| `--gpus all` (`GPUS=` to turn off, `GPUS=device=0` to pick one) | the planner runs on the GPU |
| `--network host`, `--ipc host` | ROS 2 discovery works between containers and the host; the dashboard port is reachable without `-p` |
| `ROS_DOMAIN_ID=42` (override with `ROS_DOMAIN_ID=...`) | keeps the demo's topics apart from other ROS 2 traffic on the network |
| `--user $(id -u):$(id -g)`, `HOME=/tmp` | files written into the repository stay yours |
| repository → `/workspace` | live source, data and checkpoints |
| `~/.cache/torch/hub` → `/cache/torch/hub` | V-JEPA weights and the pinned hub code (`TORCH_HOME=/cache/torch`) |
| `DISPLAY` and `/tmp/.X11-unix`, if `DISPLAY` is set | GUI tools such as `rqt_image_view` |

Use another image tag with `IMAGE=...`.

### GUI tools

```bash
xhost +local:                     # once per login, on the host
docker/run.sh ros2 run rqt_image_view rqt_image_view
```

## The demo: ROS 2 end to end

One command starts the robot (sim bridge), the planner (`/reach_goal`, on the GPU) and the dashboard, each in a detached container, and waits until all three are ready:

```bash
docker/demo.sh up --host ryzen.tail123187.ts.net   # the dashboard listens only on this name's address
docker/demo.sh status                              # what's running
docker/demo.sh logs planner                        # follow one: bridge, planner or dashboard
docker/demo.sh down                                # stop everything
```

`up` also takes `--port` (8080), `--start X Y` (the carriage's start, mm), `--checkpoint DIR` and `--no-gpu`, and prints the two commands for the demo terminals. If a piece fails to start, it shows that piece's log and stops the rest.

By hand, each `docker/run.sh` is a new container; they find each other through DDS:

```bash
docker/run.sh python -m vjepa_byor_sim.ros2_bridge          # 1: the simulated robot
docker/run.sh python apps/ros2/reach_goal_server.py         # 2: the planner as /reach_goal (GPU)
docker/run.sh python apps/ros2/dashboard.py --host <name>  # 3: the dashboard, http://<name>:8080
docker/run.sh ros2 action send_goal --feedback /reach_goal \
    vjepa_byor_interfaces/action/ReachGoal "{goal_state: [120, 60]}"   # 4: send a goal (or click the mat)
```

The dashboard saves every run to `data/dashboard-runs/` and can replay it (the fallback if live inference fails).

Step by step: switch on "Step by step" in the dashboard, or add `step_by_step: true` to the goal. Before every move the planner shows all its predictions and waits; "Next step" (or →, PageDown, Space, so a presentation clicker works) makes the move. From a terminal: `docker/run.sh ros2 service call /reach_goal/next std_srvs/srv/Trigger`.

The robot is described by `apps/ros2/sim_robot.yaml` (the built-in `ros2` rig). The same file collects data over ROS 2: `docker/run.sh vjepa-byor collect --rig ros2 --rig-config apps/ros2/sim_robot.yaml --out data/ros2-sim`. The `ReachGoal` action comes from `ros2/vjepa_byor_interfaces`, built into the image; rebuild after changing it.

## Check that it works

```bash
docker/run.sh nvidia-smi                                  # the GPU is visible
docker/run.sh ros2 topic list                             # /parameter_events, /rosout
docker/run.sh python apps/planner/sim_planner.py run      # "reached: 4 moves, 0.0 mm from the goal"
```

The planner should take about 0.25 s per decision on an RTX 5060 Ti, the same as on the host.

## Known issues

- **The prompt shows `I have no name!`.** The container runs as your host uid, which has no entry in the container's `/etc/passwd`. Harmless.
- **numpy 2 vs `cv_bridge`.** The workspace venv has numpy 2.x, which takes precedence over Jazzy's numpy 1.x. `cv_bridge` is built against numpy 1.x and may fail to import, so build `sensor_msgs/Image` messages directly from the image bytes instead.
- **`uv sync` inside the container fails.** The venv in `/opt/venv` is part of the image and read-only for your user. Change dependencies on the host and rebuild the image.
