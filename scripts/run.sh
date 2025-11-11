#!/usr/bin/env bash
set -euo pipefail

# Create venv
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt

# Helpful system deps (best-effort)
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update || true
  sudo apt-get install -y ffmpeg fonts-dejavu libcairo2-dev libpango1.0-dev libgdk-pixbuf-2.0-0 pkg-config || true
fi

python src/doodle_bodmas_autorender.py

echo "Done. See build/final_doodle_1080x1080.mp4"
