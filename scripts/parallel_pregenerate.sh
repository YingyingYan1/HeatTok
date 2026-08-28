#!/bin/bash
# Parallel semantic patch cache pregeneration script
# Usage: bash scripts/parallel_pregenerate.sh [num_gpus]
# Example: bash scripts/parallel_pregenerate.sh 8

# Default GPU count
NUM_GPUS=${1:-5}
# Optional: explicit GPU list (comma/space separated), e.g. "4,5,6,7" or "4 5 6 7"
GPU_IDS_ARG=${2:-${GPU_IDS:-""}}
GPU_IDS_LIST=()
if [ -n "$GPU_IDS_ARG" ]; then
    GPU_IDS_ARG=${GPU_IDS_ARG//,/ }
    read -r -a GPU_IDS_LIST <<< "$GPU_IDS_ARG"
    NUM_GPUS=${#GPU_IDS_LIST[@]}
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_IDS_LIST[*]}")
fi

# Config paths
CONFIG="examples/train_lora/qwen2_5vl_lora_sft.yaml"
DATASET_DIR="data"
DATASET_INFO="data/dataset_info.json"

echo "========================================"
echo "Parallel semantic patch cache pregeneration"
echo "========================================"
echo "GPU count: $NUM_GPUS"
if [ ${#GPU_IDS_LIST[@]} -gt 0 ]; then
    echo "GPU list: ${GPU_IDS_LIST[*]}"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
echo "Config file: $CONFIG"
echo "Dataset directory: $DATASET_DIR"
echo "========================================"
echo ""

# Create log directory
LOG_DIR="logs/pregenerate_cache"
mkdir -p "$LOG_DIR"

echo "Starting parallel cache pregeneration..."

# Launch one worker process per GPU shard
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    LOG_FILE="$LOG_DIR/gpu_${gpu_id}.log"
    echo "[GPU $gpu_id] Starting pregeneration task, log: $LOG_FILE"
    
    # Start each GPU worker in the background
    python scripts/pregenerate_semantic_cache.py \
        --config "$CONFIG" \
        --dataset_dir "$DATASET_DIR" \
        --dataset_info "$DATASET_INFO" \
        --gpu_id $gpu_id \
        --num_gpus $NUM_GPUS \
        > "$LOG_FILE" 2>&1 &
    
    # Record process ID
    echo "[GPU $gpu_id] PID: $!"
done

echo ""
echo "All pregeneration tasks have started and are running in the background..."
echo "Monitor progress: tail -f $LOG_DIR/gpu_*.log"
echo "Waiting for all tasks to finish..."

# Wait for all background jobs
wait

echo ""
echo "========================================"
echo "All pregeneration tasks completed!"
echo "========================================"
echo ""
echo "Per-GPU summary:"
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    LOG_FILE="$LOG_DIR/gpu_${gpu_id}.log"
    echo ""
    echo "=== GPU $gpu_id ==="
    if [ -f "$LOG_FILE" ]; then
        # Extract success/failure summary
        grep -E "(Success|Failed|Completed)" "$LOG_FILE" | tail -5
    else
        echo "Log file not found: $LOG_FILE"
    fi
done

echo ""
echo "Cache files saved to: src/semantic_patch_cache/"
echo "You can now start training: llamafactory-cli train $CONFIG"
