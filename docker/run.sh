#!/usr/bin/env bash
# Runs the ROS 2 container with the GPU, the repo, and the V-JEPA weights.
#
#   docker/run.sh                                   # interactive shell
#   docker/run.sh ros2 topic list
#   docker/run.sh python apps/planner/sim_planner.py run
#
# Host networking: DDS discovery works between containers and the host, and the
# dashboard port is reachable without -p. Runs as your user, so files written
# into the repo stay yours. GPUS= (empty) runs without a GPU, e.g. on macOS.
# DOCKER_ARGS adds docker run options, e.g. DOCKER_ARGS="-d --name x" (docker/demo.sh);
# KEEP=1 keeps the container after it exits (no --rm), so its logs can still be read.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-vjepa-ros2:jazzy}
HUB=${TORCH_HUB:-$HOME/.cache/torch/hub}

gpus=()
[ -n "${GPUS-all}" ] && gpus=(--gpus "${GPUS-all}")
tty=()
[ -t 0 ] && [ -t 1 ] && tty=(-it)
x11=()
[ -n "${DISPLAY:-}" ] && x11=(-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro)

rm=(--rm)
[ -n "${KEEP:-}" ] && rm=()
extra=()
[ -n "${DOCKER_ARGS:-}" ] && read -r -a extra <<< "$DOCKER_ARGS"

[ $# -eq 0 ] && set -- bash

exec docker run "${rm[@]}" "${tty[@]}" "${extra[@]}" \
    "${gpus[@]}" --network host --ipc host \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" \
    -v "$ROOT:/workspace" \
    -v "$HUB:/cache/torch/hub" \
    "${x11[@]}" \
    "$IMAGE" "$@"
