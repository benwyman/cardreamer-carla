#!/usr/bin/env bash
# Usage: replay_guard.sh <REPLAY_DIR> [SOFT_GB=12] [CHECK_SECS=30]
set -euo pipefail
DIR="${1:?Replay directory required}"
SOFT_GB="${2:-12}"
CHECK_EVERY="${3:-30}"
while true; do
  [ -d "$DIR" ] || { sleep "$CHECK_EVERY"; continue; }
  bytes=$(du -sb "$DIR" 2>/dev/null | awk '{print $1}')
  gb=$(( bytes / 1024 / 1024 / 1024 ))
  if (( gb > SOFT_GB )); then
    mapfile -t old < <(find "$DIR" -maxdepth 1 -type f -name '*.npz' -mmin +2 -printf '%T@ %p\n' \
                       | sort -n | head -n 10 | awk '{ $1=""; sub(/^ /,""); print }')
    for f in "${old[@]:-}"; do
      [ -n "${f:-}" ] || continue
      echo "[replay_guard] removing $f"
      rm -f -- "$f" || true
    done
  fi
  sleep "$CHECK_EVERY"
done
