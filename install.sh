#!/bin/bash

set -e  # Exit on error

# ------------------------
# SYSTEM-LEVEL DEPENDENCIES
# ------------------------
echo "???  Installing system packages..."
sudo apt update
sudo apt install -y \
    python3-picamera2 \
    python3-simplejpeg \
    python3-opencv \
    python3-numpy \
    python3-pip \
    python3-venv \
    libjpeg-dev \
    libatlas-base-dev \
    libopenblas-dev \
    libblas-dev

# ------------------------
# CREATE VIRTUAL ENV
# ------------------------
echo "?? Creating virtual environment..."
python3 -m venv repleye-venv --system-site-packages
source repleye-venv/bin/activate

# ------------------------
# PYTHON DEPENDENCIES
# ------------------------
echo "??  Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "??  Installing PyTorch and TorchVision..."
pip install torch torchvision

echo "?? Installing ReplEye package from GitHub..."
pip install --upgrade --force-reinstall git+https://github.com/sevakuksin/ReplEye.git

echo "? All done! To start using:"
echo "source repleye-venv/bin/activate"

echo "?? Reinstalling simplejpeg to match numpy..."
pip install simplejpeg --force-reinstall
