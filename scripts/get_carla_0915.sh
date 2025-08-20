#!/usr/bin/env bash
set -euo pipefail

CARLA_DIR="carla"
TAR="CARLA_0.9.15.tar.gz"
URL_PRIMARY="https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz"
URL_FALLBACK="https://tiny.carla.org/carla-0-9-15-linux"

# Skip if already installed
if [[ -x "$CARLA_DIR/CarlaUE4.sh" ]]; then
  echo "[get_carla] $CARLA_DIR/CarlaUE4.sh exists — skipping."
  exit 0
fi

echo "[get_carla] Downloading CARLA 0.9.15..."
wget -c -L "$URL_PRIMARY" -O "$TAR" || wget -c -L "$URL_FALLBACK" -O "$TAR"

echo "[get_carla] Extracting to $CARLA_DIR..."
rm -rf "$CARLA_DIR"
mkdir -p "$CARLA_DIR"
tar -xzvf "$TAR" -C "$CARLA_DIR"

chmod +x "$CARLA_DIR/CarlaUE4.sh" || true
echo "[get_carla] ✅ Ready: $CARLA_DIR/CarlaUE4.sh"
