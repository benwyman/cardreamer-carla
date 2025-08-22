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

cd $DREAMER_ROOT

echo "[env.sh] Ready inside CarDreamer"

#
# ---- Stability knobs (JAX/XLA + quiet TF) ----
# 1) Turn OFF TF auto-JIT (JAX doesn’t use this; avoids extra TF compile churn)
unset TF_XLA_FLAGS
# 2) Make JAX/XLA GPU convs deterministic and dial back cuDNN autotuning
#    (prevents those bf16 conv “results mismatch” crashes on Ada + CUDA 12.9 + cuDNN 9.12)
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
# 3) Optional: reduce glibc heap fragmentation (helps host RAM spikes a bit)
export MALLOC_ARENA_MAX=2
# 4) Optional: quiet TF warnings (since TF runs on CPU for you)
export TF_CPP_MIN_LOG_LEVEL=2
# 5) Optional safety: log (don’t crash) if any implicit host→device copy slips through.
#    Leave commented out if you applied the explicit device_put fix in train.py.
export JAX_TRANSFER_GUARD=log
