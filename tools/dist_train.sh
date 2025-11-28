#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

# Get absolute path to project root and projects directory
PROJECT_ROOT=$(cd "$(dirname $0)/.." && pwd)
PROJECTS_DIR="${PROJECT_ROOT}/projects"

# Only add projects directory to PYTHONPATH (not the whole repo)
# This allows custom plugins to be imported while using installed mmdet3d
PYTHONPATH="${PROJECTS_DIR}:$PYTHONPATH" \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
