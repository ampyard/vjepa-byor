# ROS 2 Jazzy + the vjepa-byor workspace, with the GPU (demo backlog item 3).
#
#   docker build -f docker/Dockerfile -t vjepa-ros2:jazzy .
#   docker/run.sh                      # interactive shell, ROS sourced
#
# Ubuntu 26.04 isn't a Jazzy target, so ROS lives here. Jazzy's base (24.04) has
# Python 3.12, the same as the workspace, so the venv sees rclpy and our packages.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-venv \
        ros-jazzy-rqt-image-view \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /usr/local/bin/

# The venv includes system site-packages (rclpy's deps); ROS's own Python
# packages come from PYTHONPATH once /opt/ros/jazzy/setup.bash is sourced.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    TORCH_HOME=/cache/torch
WORKDIR /workspace
RUN uv venv --system-site-packages --python /usr/bin/python3 /opt/venv

# Third-party dependencies first (the big, slow layer: torch + CUDA wheels)
COPY pyproject.toml uv.lock .python-version ./
COPY packages/vjepa-byor/pyproject.toml packages/vjepa-byor/README.md packages/vjepa-byor/
COPY packages/vjepa-byor-sim/pyproject.toml packages/vjepa-byor-sim/README.md packages/vjepa-byor-sim/
COPY packages/vjepa-byor-esp32-stepper/pyproject.toml packages/vjepa-byor-esp32-stepper/README.md packages/vjepa-byor-esp32-stepper/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace

# Workspace packages, installed editable: at run time the repo is mounted over
# /workspace, so the installed packages are the live source. The dev group
# (pytest) comes in here, so the ROS tests can run in the container.
COPY packages packages
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen \
    && chmod -R a+rX /opt/venv

# docker exec shells get ROS too (docker run goes through ros_entrypoint.sh)
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /etc/bash.bashrc
