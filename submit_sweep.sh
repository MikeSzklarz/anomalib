#!/bin/bash

# 1. HARDCODED MODEL LIST
MODELS=(
    "efficientad"
    "dinomaly"
    "fastflow"
    "csflow"
    "reversedistillation"
    "patchcore"
    "stfpm"                           
    "uflow"
    "cflow"
    "padim"
)

# 2. SLURM / SWEEP DEFAULTS
PARTITION="waccamaw"
NODELIST=""
DATASET="unknown" # Default for job naming only

# 3. ARGUMENT PASS-THROUGH LOGIC
# We collect arguments for train.py here
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    # --- SLURM SPECIFIC ARGS (Consumed by this script) ---
    --nodelist)
      NODELIST="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    
    # --- HYBRID ARGS (Used by this script AND train.py) ---
    # We need --dataset for the Job Name, but we also must pass it to python
    --dataset)
      DATASET="$2"
      TRAIN_ARGS+=("$1" "$2")
      shift 2
      ;;

    # --- EVERYTHING ELSE (Passed blindly to train.py) ---
    # This catches --root_dir, --category, --max_epochs, --grayscale, 
    # and any NEW arguments you add to train.py in the future.
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

# 4. PREPARATION
mkdir -p logs/slurm

# Parse Nodes for Round Robin
IFS=',' read -r -a NODES_ARRAY <<< "$NODELIST"
NUM_NODES=${#NODES_ARRAY[@]}

echo "========================================"
echo "Starting Sweep"
echo "Models: ${#MODELS[@]}"
echo "Partition: $PARTITION"
echo "Job Name Suffix: $DATASET"
echo "Passing through args: ${TRAIN_ARGS[*]}"
if [ "$NUM_NODES" -gt 0 ]; then
    echo "Distributing across nodes: ${NODES_ARRAY[*]}"
fi
echo "========================================"

# 5. SUBMISSION LOOP
count=0
for model in "${MODELS[@]}"; do
    
    # --- Round Robin Node Selection ---
    SLURM_NODE_ARG=""
    if [ "$NUM_NODES" -gt 0 ]; then
        node_index=$((count % NUM_NODES))
        selected_node="${NODES_ARRAY[$node_index]}"
        SLURM_NODE_ARG="--nodelist=$selected_node"
        echo "Assigning $model -> $selected_node"
    fi

    # --- Submit Job ---
    # We pass --model explicitly (from loop)
    # We pass "${TRAIN_ARGS[@]}" (everything else found in CLI)
    sbatch \
        --job-name="${model}_${DATASET}" \
        --partition="$PARTITION" \
        $SLURM_NODE_ARG \
        run_slurm.sh \
        --model "$model" \
        "${TRAIN_ARGS[@]}"

    ((count++))
done

echo "Successfully submitted $count jobs."