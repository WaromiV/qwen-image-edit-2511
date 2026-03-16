#!/usr/bin/env bash
set -euo pipefail

ts() {
  date -u +"%Y-%m-%dT%H:%M:%S.%3NZ"
}

log() {
  printf '{"ts":"%s","stage":"%s","message":"%s"}\n' "$(ts)" "$1" "$2"
}

log "start.sh.begin" "Initializing ComfyUI startup wrapper"

if [ ! -e /usr/lib/x86_64-linux-gnu/libcuda.so ] && [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so || true
  log "start.sh.libcuda" "Created libcuda.so symlink"
fi

cd "${COMFYUI_DIR:-/comfyui}"
log "start.sh.exec" "Launching ComfyUI main.py"
exec python -u main.py --disable-auto-launch --disable-metadata --listen "${COMFY_HOST:-127.0.0.1}" --port "${COMFY_PORT:-8188}" --log-stdout
