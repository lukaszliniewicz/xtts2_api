#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$PROJECT_DIR/bin"
PIXI_EXE="$BIN_DIR/pixi"
FORCE=""
BACKEND=""
YES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) BACKEND="cpu" ;;
    --gpu|--cuda) BACKEND="cuda" ;;
    --force) FORCE="1" ;;
    --yes) YES="1" ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

echo ""
echo "============================================"
echo " XTTS FastAPI Server - Installer"
echo "============================================"
echo ""

# Detect OS/arch
ARCH="$(uname -m)"
OS="$(uname -s)"

case "$OS" in
  Linux)
    case "$ARCH" in
      x86_64)  PIXI_ARCH="x86_64-unknown-linux-musl" ;;
      aarch64) PIXI_ARCH="aarch64-unknown-linux-musl" ;;
      *) echo "Unsupported arch: $ARCH"; exit 1 ;;
    esac
    ;;
  Darwin)
    case "$ARCH" in
      x86_64)  PIXI_ARCH="x86_64-apple-darwin" ;;
      arm64)   PIXI_ARCH="aarch64-apple-darwin" ;;
      *) echo "Unsupported arch: $ARCH"; exit 1 ;;
    esac
    ;;
  *)
    echo "Unsupported OS: $OS"
    echo "On Windows, use scripts/install.bat"
    exit 1
    ;;
esac

# Check if already installed
if [ -f "$PIXI_EXE" ] && [ -z "$FORCE" ]; then
  echo "pixi already found at $PIXI_EXE"
  echo "Use --force to reinstall."
  goto_install_env
fi

# Create bin dir
mkdir -p "$BIN_DIR"

# Download pixi
PIXI_VERSION="v0.68.1"
echo "[1/4] Downloading pixi ($PIXI_ARCH)..."

if command -v curl &>/dev/null; then
  curl -fsSL "https://github.com/prefix-dev/pixi/releases/download/$PIXI_VERSION/pixi-$PIXI_ARCH.tar.gz" -o "$BIN_DIR/pixi.tar.gz"
elif command -v wget &>/dev/null; then
  wget -qO "$BIN_DIR/pixi.tar.gz" "https://github.com/prefix-dev/pixi/releases/download/$PIXI_VERSION/pixi-$PIXI_ARCH.tar.gz"
else
  echo "ERROR: need curl or wget"
  exit 1
fi

tar -xzf "$BIN_DIR/pixi.tar.gz" -C "$BIN_DIR/"
chmod +x "$PIXI_EXE"
rm -f "$BIN_DIR/pixi.tar.gz"
echo "pixi downloaded to $PIXI_EXE"

goto_install_env() {
  :
}

echo ""
echo "[2/4] Installing base environment..."
cd "$PROJECT_DIR"
"$PIXI_EXE" install || { echo "ERROR: pixi install failed"; exit 1; }

# Choose backend
echo ""
echo "[3/4] Configuring PyTorch backend..."

if [ -z "$BACKEND" ]; then
  if [ "$OS" = "Darwin" ]; then
    echo "macOS detected: installing PyTorch (MPS-enabled)"
    BACKEND="mac"
  else
    read -r -p "Select backend (cpu/cuda, default=cuda): " BACKEND
    BACKEND="${BACKEND:-cuda}"
  fi
fi

case "$BACKEND" in
  cuda)
    echo "Installing PyTorch with CUDA support..."
    "$PIXI_EXE" run install-torch-cuda || {
      echo "WARNING: CUDA install failed, falling back to CPU..."
      "$PIXI_EXE" run install-torch-cpu
    }
    ;;
  rocm)
    echo "Installing PyTorch with ROCm support..."
    "$PIXI_EXE" run install-torch-rocm || "$PIXI_EXE" run install-torch-cpu
    ;;
  mac)
    "$PIXI_EXE" run pip install torch torchvision torchaudio
    ;;
  *)
    echo "Installing PyTorch with CPU support..."
    "$PIXI_EXE" run install-torch-cpu
    ;;
esac

# Install coqui-tts
echo ""
echo "[4/4] Installing coqui-tts..."
"$PIXI_EXE" run pip install coqui-tts || echo "WARNING: coqui-tts install had issues"

# Verify
echo ""
echo "============================================"
echo " Installation complete!"
echo "============================================"
echo ""
"$PIXI_EXE" run check-runtime
echo ""
echo "To start the server, run:"
echo "  scripts/run.sh"
echo ""
