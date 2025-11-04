#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# setup.sh
# -----------------------------------------------------------------------------
# Quick bootstrap script for the Algorithmic Asset Management project.
# It creates a Python virtual environment (named .venv), activates it, and
# installs all dependencies from requirements.txt. Run this script from the
# project root:
#   bash setup.sh
# -----------------------------------------------------------------------------
set -e  # Exit immediately if a command exits with a non-zero status

PYTHON_VERSION_REQUIRED="3.10"

# Detect python3
if ! command -v python3 &> /dev/null; then
    echo "python3 could not be found. Please install Python $PYTHON_VERSION_REQUIRED or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

if [[ $(printf '%s\n' "$PYTHON_VERSION_REQUIRED" "$PYTHON_VERSION" | sort -V | head -n1) != "$PYTHON_VERSION_REQUIRED" ]]; then
    echo "Python $PYTHON_VERSION_REQUIRED or higher is required. Your version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv) using python3 ..."
    python3 -m venv .venv
fi

# Activate virtual environment
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip ..."
pip install --upgrade pip setuptools wheel

echo "Installing project dependencies ..."
pip install -r requirements.txt

echo "Setup complete. Activate the virtual environment with:"
echo "  source .venv/bin/activate"
