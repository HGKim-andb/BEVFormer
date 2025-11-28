#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

# Get absolute path to project root
PROJECT_ROOT=$(cd "$(dirname $0)/.." && pwd)

# Ensure local project code takes precedence over installed packages
PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
