#!/usr/bin/env bash
set -euo pipefail

# Install as editable so you can modify the borrowed repo codes locally
python -m pip install -e extern/ChartDete
python -m pip install -e extern/LineFormer