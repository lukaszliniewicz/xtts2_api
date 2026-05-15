#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="$PROJECT_DIR/bin"
PIXI_EXE="$BIN_DIR/pixi"

# Detect OS/arch for pixi download
ARCH="$(uname -m)"
OS="$(uname -s)"
PIXI_ARCH=""
case "$OS" in
  Linux)
    case "$ARCH" in
      x86_64)  PIXI_ARCH="x86_64-unknown-linux-musl" ;;
      aarch64) PIXI_ARCH="aarch64-unknown-linux-musl" ;;
      *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
    esac
    ;;
  Darwin)
    case "$ARCH" in
      x86_64)  PIXI_ARCH="x86_64-apple-darwin" ;;
      arm64)   PIXI_ARCH="aarch64-apple-darwin" ;;
      *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
    esac
    ;;
  *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

# Download pixi if missing
if [ ! -f "$PIXI_EXE" ]; then
  echo "Downloading pixi..."
  mkdir -p "$BIN_DIR"
  PIXI_VERSION="v0.68.1"
  URL="https://github.com/prefix-dev/pixi/releases/download/$PIXI_VERSION/pixi-$PIXI_ARCH.tar.gz"
  if command -v curl &>/dev/null; then
    curl -fsL "$URL" -o "$BIN_DIR/pixi.tar.gz"
  elif command -v wget &>/dev/null; then
    wget -qO "$BIN_DIR/pixi.tar.gz" "$URL"
  else
    echo "Need curl or wget to download pixi."
    exit 1
  fi
  tar -xzf "$BIN_DIR/pixi.tar.gz" -C "$BIN_DIR/"
  chmod +x "$PIXI_EXE"
  rm -f "$BIN_DIR/pixi.tar.gz"
fi

# Ensure pixi env is installed
if [ ! -f "$PROJECT_DIR/.pixi/envs/default/bin/python" ]; then
  echo "Installing pixi environment..."
  cd "$PROJECT_DIR"
  "$PIXI_EXE" install
elif ! "$PIXI_EXE" install --frozen &>/dev/null; then
  cd "$PROJECT_DIR"
  "$PIXI_EXE" install
fi

# Set local cache dirs
export PIXI_CACHE_DIR="$PROJECT_DIR/.pixi-cache"
export PIP_CACHE_DIR="$PROJECT_DIR/.pip-cache"
mkdir -p "$PIXI_CACHE_DIR" "$PIP_CACHE_DIR"

# Start bootstrapper
cd "$PROJECT_DIR"
exec "$PIXI_EXE" run python run.py "$@"
