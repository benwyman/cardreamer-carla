#!/bin/bash
# Activate venv
source /workspace/base/carla_project/venv/bin/activate

# Set paths
export CARLA_ROOT=/workspace/base/carla_project/carla
export DREAMER_ROOT=$CARLA_ROOT/../CarDreamer
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/agents:$PYTHONPATH
export PYTHONPATH=/workspace/base/carla_project/CarDreamer:$PYTHONPATH   # <-- add DreamerV3 repo

# Fix XDG runtime
export XDG_RUNTIME_DIR=/tmp/$USER-runtime
mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR

# Headless: make sure nothing points to X
unset DISPLAY
export QT_QPA_PLATFORM=offscreen
export SDL_VIDEODRIVER=offscreen

cd $DREAMER_ROOT

echo "[env.sh] Ready inside CarDreamer"

export TF_XLA_FLAGS=--tf_xla_auto_jit=2
