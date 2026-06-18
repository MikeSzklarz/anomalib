#!/bin/bash

SMALL_MODELS=(
    "uflow"                 # Node 0: 705m
    "csflow"                # Node 1: 340m
    "fastflow"              # Node 0: 121m
    "cflow"                 # Node 1: 239m
    "reversedistillation"   # Node 0: 105m
    "dinomaly"              # Node 1: 199m
    "padim"                 # Node 0: 0m
    "efficientad"           # Node 1: 104m
    "patchcore"             # Node 0: 0m
    "stfpm"                 # Node 1: 20m
    "l2bt"                  # Node 0: ~0m
    "patchflow"             # Node 1: ~121m
)

FULL_MODELS=(
    "uflow"                 # Node 0: 705m
    "csflow"                # Node 1: 340m
    "dinomaly"              # Node 0: 199m
    "cflow"                 # Node 1: 239m
    "reversedistillation"   # Node 0: 105m
    "draem"                 # Node 1: 228m
    "supersimplenet"        # Node 0: 79m
    "fastflow"              # Node 1: 121m
    "fre"                   # Node 0: 24m
    "efficientad"           # Node 1: 104m
    "dsr"                   # Node 0: 13m
    "ganomaly"              # Node 1: 40m
    "dfkde"                 # Node 0: 0m
    "uninet"                # Node 1: 30m
    "padim"                 # Node 0: 0m
    "stfpm"                 # Node 1: 20m
    "dfm"                   # Node 0: 0m
    "cfa"                   # Node 1: 9m
    "patchcore"             # Node 0: 0m
    "anomalydino"           # Node 1: 0m
    "generalad"             # Node 0: ~160m
    "l2bt"                  # Node 1: ~0m
    "patchflow"             # Node 0: ~200m
    "glass"                 # Node 1: ~100m
    "inpformer"             # Node 0: ~200m
    "cfm"                   # Node 1: ~100m
    "anomalyvfm"            # Node 0: 0m (zero-shot)
)

MODELS=()
ZOO_SELECTION="small" 

PARTITION="waccamaw"
NODELIST=""
DATASET="unknown"
TRAIN_ARGS=()
GLOBAL_BATCH_SIZE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    # --- ZOO SELECTION ---
    --zoo)
      if [[ "$2" == "full" ]]; then
        ZOO_SELECTION="full"
      else
        ZOO_SELECTION="small"
      fi
      shift 2
      ;;
    --nodelist)
      NODELIST="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --model)
      MODELS+=("$2")
      shift 2
      ;;
    --models)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    --dataset)
      DATASET="$2"
      TRAIN_ARGS+=("$1" "$2")
      shift 2
      ;;
    --batch_size)
      # Extract batch size from the arguments so we can manipulate it
      GLOBAL_BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ ${#MODELS[@]} -eq 0 ]; then
    if [ "$ZOO_SELECTION" == "full" ]; then
        echo "Using FULL model zoo."
        MODELS=("${FULL_MODELS[@]}")
    else
        echo "Using SMALL model zoo."
        MODELS=("${SMALL_MODELS[@]}")
    fi
fi

mkdir -p logs/slurm

IFS=',' read -r -a NODES_ARRAY <<< "$NODELIST"
NUM_NODES=${#NODES_ARRAY[@]}

echo "========================================"
echo "Starting Sweep"
echo "Zoo Mode: $ZOO_SELECTION"
echo "Models to run: ${#MODELS[@]}"
echo "Partition: $PARTITION"
echo "Job Name Suffix: $DATASET"
if [ -n "$GLOBAL_BATCH_SIZE" ]; then
    echo "Requested Batch Size: $GLOBAL_BATCH_SIZE"
fi
echo "Passing through args: ${TRAIN_ARGS[*]}"
if [ "$NUM_NODES" -gt 0 ]; then
    echo "Distributing across nodes: ${NODES_ARRAY[*]}"
fi
echo "========================================"

count=0
for model in "${MODELS[@]}"; do
    
    SLURM_NODE_ARG=""
    if [ "$NUM_NODES" -gt 0 ]; then
        node_index=$((count % NUM_NODES))
        selected_node="${NODES_ARRAY[$node_index]}"
        SLURM_NODE_ARG="--nodelist=$selected_node"
        echo "Assigning $model -> $selected_node"
    fi

    # Create a model-specific copy of the training arguments
    MODEL_TRAIN_ARGS=("${TRAIN_ARGS[@]}")
    MODEL_BATCH_SIZE="$GLOBAL_BATCH_SIZE"

    # Apply capping logic for memory-intensive models
    if [[ "$model" == "draem" || "$model" == "uninet" || "$model" == "generalad" || "$model" == "glass" ]]; then
        if [[ -z "$MODEL_BATCH_SIZE" ]]; then
            # If no batch size was passed, your python script defaults to 32. Cap it at 16.
            MODEL_BATCH_SIZE="16"
            echo "Applying batch size 16 to $model (capping default 32)."
        elif [[ "$MODEL_BATCH_SIZE" -gt 16 ]]; then
            # If the user passed a batch size > 16, force it down to 16.
            MODEL_BATCH_SIZE="16"
            echo "Capping user batch size to 16 for $model to prevent OOM."
        fi
        # If MODEL_BATCH_SIZE <= 16, it leaves it alone.
    fi

    # Re-inject the batch size into the arguments (if it has a value)
    if [[ -n "$MODEL_BATCH_SIZE" ]]; then
        MODEL_TRAIN_ARGS+=("--batch_size" "$MODEL_BATCH_SIZE")
    fi

    sbatch \
        --job-name="${model}_${DATASET}" \
        --partition="$PARTITION" \
        $SLURM_NODE_ARG \
        run_slurm.sh \
        --model "$model" \
        "${MODEL_TRAIN_ARGS[@]}"

    ((count++))
done

echo "Successfully submitted $count jobs."