#!/bin/bash

# Stop script bij fouten
set -e

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing CPU-only PyTorch and dependencies..."
# Let op: dit installeert de standaard CPU-versie van PyTorch
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1

echo "Installing project dependencies..."
pip install -e .  # Als je in de AIRS repo zit
pip install pytorch-lightning==1.7.7
pip install phiflow
pip install -U 'jsonargparse[signatures]'
pip install torchmetrics==0.11.4
pip install -U rich

echo "Setup complete!"
echo "Run: source .venv/bin/activate to activate your environment."
