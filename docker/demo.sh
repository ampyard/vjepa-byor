#!/usr/bin/env bash
# Brings the whole ROS 2 demo up or down with one command (ROSCon demo backlog item 8).
#
#   docker/demo.sh up --host ryzen.tail123187.ts.net   # bridge + /reach_goal server + dashboard
#   docker/demo.sh status                              # what's running
#   docker/demo.sh logs [bridge|planner|dashboard]     # follow one (default: all, last lines)
#   docker/demo.sh down                                # stop everything
#
# up options: --host NAME (required: the dashboard listens only on its address), --port N (8080),
# --start X Y (where the carriage starts, mm), --checkpoint DIR (checkpoints/sim-10k),
# --no-gpu. Each piece runs in its own detached container (vjepa-bridge, vjepa-planner,
# vjepa-dashboard) on ROS_DOMAIN_ID (42 by default), and `up` waits until each is ready.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN="$ROOT/docker/run.sh"
NAMES=(vjepa-bridge vjepa-planner vjepa-dashboard)
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

running() { docker ps -q --filter "name=^$1$"; }
exists() { docker ps -aq --filter "name=^$1$"; }

# demo containers are kept after they exit (logs stay readable); remove stopped ones
remove_stopped() {
    for n in "${NAMES[@]}"; do
        if [ -n "$(exists "$n")" ] && [ -z "$(running "$n")" ]; then docker rm "$n" >/dev/null; fi
    done
}

down() {
    local any=0
    for n in "${NAMES[@]}"; do
        if [ -n "$(running "$n")" ]; then docker stop -t 5 "$n" >/dev/null && echo "stopped $n"; any=1; fi
    done
    remove_stopped
    [ $any -eq 1 ] || echo "nothing running"
}

# start NAME READY_PATTERN TIMEOUT_S [GPUS] -- command...
start() {
    local name=$1 ready=$2 timeout=$3 gpus=$4; shift 5
    KEEP=1 GPUS=$gpus DOCKER_ARGS="-d --name $name" "$RUN" "$@" >/dev/null
    printf '%-16s starting' "$name"
    local t=0
    until docker logs "$name" 2>&1 | grep -q "$ready"; do
        if [ -z "$(running "$name")" ] || [ $t -ge "$timeout" ]; then
            echo " FAILED"
            docker logs --tail 20 "$name" 2>&1 | sed 's/^/    /' || true
            echo "stopping what was started"; down >/dev/null; remove_stopped
            exit 1
        fi
        sleep 1; t=$((t + 1)); printf '.'
    done
    echo " ready (${t} s)"
}

up() {
    local host="" port=8080 start_x=100 start_y=75 checkpoint="checkpoints/sim-10k" gpus=all
    while [ $# -gt 0 ]; do
        case $1 in
            --host) host=$2; shift 2 ;;
            --port) port=$2; shift 2 ;;
            --start) start_x=$2; start_y=$3; shift 3 ;;
            --checkpoint) checkpoint=$2; shift 2 ;;
            --no-gpu) gpus=""; shift ;;
            -h|--help) usage ;;
            *) echo "unknown option $1" >&2; usage 1 ;;
        esac
    done
    [ -n "$host" ] || { echo "up needs --host NAME (the dashboard listens only there)" >&2; exit 1; }
    for n in "${NAMES[@]}"; do
        [ -z "$(running "$n")" ] || { echo "$n is already running: docker/demo.sh down first" >&2; exit 1; }
    done
    [ -d "$ROOT/$checkpoint" ] || { echo "no checkpoint at $checkpoint" >&2; exit 1; }
    local addresses
    addresses=$( (getent ahostsv4 "$host" || true) | awk '{print $1}' | sort -u)
    [ -n "$addresses" ] || { echo "can't resolve $host" >&2; exit 1; }
    case $addresses in
        127.*) echo "warning: $host is $addresses (loopback): only this machine can open the dashboard;" \
                    "for Tailscale use the full MagicDNS name" >&2 ;;
    esac
    remove_stopped

    echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
    start vjepa-bridge "sim stage" 30 "" -- \
        python -m vjepa_byor_sim.ros2_bridge --ros-args -p "start_mm:=[$start_x.0, $start_y.0]"
    start vjepa-planner "/reach_goal ready" 180 "$gpus" -- \
        python apps/ros2/reach_goal_server.py --finetuned "$checkpoint"
    start vjepa-dashboard "dashboard on" 30 "" -- \
        python apps/ros2/dashboard.py --host "$host" --port "$port"

    cat <<MSG

Dashboard:  http://$host:$port

Demo terminals (each a new container on ROS_DOMAIN_ID=$ROS_DOMAIN_ID):
  docker/run.sh ros2 action send_goal --feedback /reach_goal vjepa_byor_interfaces/action/ReachGoal "{goal_state: [120, 60]}"
  docker/run.sh ros2 topic echo /rig/cmd

Logs: docker/demo.sh logs [bridge|planner|dashboard]    Stop: docker/demo.sh down
MSG
}

status() {
    for n in "${NAMES[@]}"; do
        if [ -n "$(running "$n")" ]; then
            printf '%-16s up %s\n' "$n" "$(docker ps --filter "name=^$n$" --format '{{.RunningFor}}')"
        else
            printf '%-16s down\n' "$n"
        fi
    done
}

logs() {
    if [ $# -gt 0 ]; then
        docker logs -f "vjepa-${1#vjepa-}"
    else
        for n in "${NAMES[@]}"; do
            echo "== $n"; docker logs --tail 5 "$n" 2>&1 | sed 's/^/    /' || echo "    (not running)"
        done
    fi
}

cmd=${1:-}; [ $# -gt 0 ] && shift
case $cmd in
    up) up "$@" ;;
    down) down ;;
    status) status ;;
    logs) logs "$@" ;;
    -h|--help|help|"") usage ;;
    *) echo "unknown command $cmd" >&2; usage 1 ;;
esac
