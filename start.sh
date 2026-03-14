#!/usr/bin/env bash
set -euo pipefail
cd "${COMFYUI_DIR:-/opt/ComfyUI}"
exec python3 main.py --listen "${COMFY_HOST:-127.0.0.1}" --port "${COMFY_PORT:-8188}" --disable-auto-launch
