#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$PROJECT_DIR/bin"
PIXI_EXE="$BIN_DIR/pixi"
INSTALL_SCRIPT="$(dirname "$0")/install.sh"

# Check if pixi exists, if not run installer
if [ ! -f "$PIXI_EXE" ]; then
  echo "pixi not found. Running installer first..."
  bash "$INSTALL_SCRIPT" --yes
fi

# Start server
echo "Starting XTTS FastAPI server..."
cd "$PROJECT_DIR"
"$PIXI_EXE" run serve
