#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

# Get absolute path to project root
PROJECT_ROOT=$(cd "$(dirname $0)/.." && pwd)

# Ensure local project code takes precedence over installed packages
PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
