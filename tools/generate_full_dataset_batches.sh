#!/bin/bash
# Generate V5 risk labels for nuScenes full dataset in batches

DATAROOT="/home/hg-main/data2/datasets/nuscenes/data/nuscenes"
VERSION="v1.0-trainval"
OUTPUT_DIR="data/emergence_risk_v5_full"

# Total 850 scenes, split into 10 batches of ~85 scenes each
# Each batch takes ~2-3 hours

echo "========================================"
echo "Full Dataset V5 Label Generation"
echo "========================================"
echo "Total scenes: 850"
echo "Total samples: ~34,149"
echo "Estimated time: 15-20 hours (with parallel processing)"
echo ""
echo "Strategy: Split into 10 batches of ~85 scenes"
echo "========================================"
echo ""

# Get all scene names
python -c "
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='$VERSION', dataroot='$DATAROOT', verbose=False)
scene_names = [s['name'] for s in nusc.scene]
print(' '.join(scene_names))
" > /tmp/all_scenes.txt

ALL_SCENES=$(cat /tmp/all_scenes.txt)
SCENE_ARRAY=($ALL_SCENES)
TOTAL_SCENES=${#SCENE_ARRAY[@]}

echo "Found $TOTAL_SCENES scenes"
echo ""

# Batch configuration
BATCH_SIZE=85
NUM_BATCHES=$(( (TOTAL_SCENES + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "Creating $NUM_BATCHES batches of ~$BATCH_SIZE scenes each"
echo ""

# Function to run a batch
run_batch() {
    BATCH_NUM=$1
    START_IDX=$(( BATCH_NUM * BATCH_SIZE ))
    END_IDX=$(( START_IDX + BATCH_SIZE ))

    if [ $END_IDX -gt $TOTAL_SCENES ]; then
        END_IDX=$TOTAL_SCENES
    fi

    # Extract scene names for this batch
    BATCH_SCENES=""
    for ((i=START_IDX; i<END_IDX; i++)); do
        BATCH_SCENES="$BATCH_SCENES ${SCENE_ARRAY[$i]}"
    done

    NUM_SCENES_IN_BATCH=$(( END_IDX - START_IDX ))

    echo "========================================"
    echo "Batch $((BATCH_NUM + 1))/$NUM_BATCHES"
    echo "========================================"
    echo "Scenes: $START_IDX - $((END_IDX - 1)) ($NUM_SCENES_IN_BATCH scenes)"
    echo "Output: ${OUTPUT_DIR}_batch_$((BATCH_NUM + 1))"
    echo ""

    # Run label generation
    python tools/create_risk_labels.py \
        --dataroot $DATAROOT \
        --version $VERSION \
        --output_dir ${OUTPUT_DIR}_batch_$((BATCH_NUM + 1)) \
        --scenes $BATCH_SCENES \
        --parallel

    echo ""
    echo "✅ Batch $((BATCH_NUM + 1)) complete!"
    echo ""
}

# Ask user which batches to run
echo "Which batches do you want to run?"
echo "  1) All batches (1-$NUM_BATCHES)"
echo "  2) Single batch (specify number)"
echo "  3) Range of batches (specify start-end)"
read -p "Choice [1/2/3]: " CHOICE

case $CHOICE in
    1)
        echo "Running ALL batches..."
        for ((batch=0; batch<NUM_BATCHES; batch++)); do
            run_batch $batch
        done
        ;;
    2)
        read -p "Enter batch number (1-$NUM_BATCHES): " BATCH_NUM
        BATCH_NUM=$((BATCH_NUM - 1))
        if [ $BATCH_NUM -ge 0 ] && [ $BATCH_NUM -lt $NUM_BATCHES ]; then
            run_batch $BATCH_NUM
        else
            echo "❌ Invalid batch number"
            exit 1
        fi
        ;;
    3)
        read -p "Enter start batch (1-$NUM_BATCHES): " START_BATCH
        read -p "Enter end batch (1-$NUM_BATCHES): " END_BATCH
        START_BATCH=$((START_BATCH - 1))
        END_BATCH=$((END_BATCH - 1))

        if [ $START_BATCH -ge 0 ] && [ $END_BATCH -lt $NUM_BATCHES ] && [ $START_BATCH -le $END_BATCH ]; then
            for ((batch=START_BATCH; batch<=END_BATCH; batch++)); do
                run_batch $batch
            done
        else
            echo "❌ Invalid batch range"
            exit 1
        fi
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "All requested batches complete!"
echo "========================================"
echo ""
echo "Next step: Merge batches using:"
echo "  python tools/merge_risk_batches.py \\"
echo "    --input_dirs ${OUTPUT_DIR}_batch_* \\"
echo "    --output_dir $OUTPUT_DIR"
echo ""
