#!/usr/bin/env bash
set -euo pipefail

CARLA_URL="https://github.com/carla-simulator/carla/releases/download/0.9.15/CARLA_0.9.15.tar.gz"
TAR="CARLA_0.9.15.tar.gz"
CARLA_DIR="carla"

echo "[get_carla] Downloading CARLA 0.9.15..."
wget -q --show-progress "$CARLA_URL" -O "$TAR"

echo "[get_carla] Extracting to $CARLA_DIR (verbose, no --strip-components)..."
mkdir -p "$CARLA_DIR"
tar -xzvf "$TAR" -C "$CARLA_DIR"

echo "[get_carla] Extraction complete."

