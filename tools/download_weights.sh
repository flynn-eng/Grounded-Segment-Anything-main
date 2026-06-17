#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

download() {
  local url="$1"
  local output="$2"

  echo "==> ${output}"
  wget \
    --continue \
    --tries=0 \
    --timeout=30 \
    --read-timeout=30 \
    --waitretry=5 \
    --retry-connrefused \
    --progress=bar:force:noscroll \
    --output-document="$output" \
    "$url"
}

download \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
  "sam_vit_h_4b8939.pth"

download \
  "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
  "groundingdino_swint_ogc.pth"

echo "==> done"
