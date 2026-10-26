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
    # --- Priority: strong + fast, run these first ---
    "efficientad"
    "dinomaly"
    "inpformer"
    "l2bt"
    "reversedistillation"
    "fastflow"
    "glass"
    # --- Fast: zero-shot / patch-based / memory-bank methods, minimal training time ---
    "anomalyvfm"
    "superadd"
    "patchcore"
    "padim"
    "patchflow"
    "anomalydino"
    "cfa"
    "mhpatchcore"
    "rad"
    # --- Known slow ---
    "uflow"
    # --- Moderate performers ---
    "cflow"
    # --- Historically weak performers (deprioritized, not excluded) ---
    "stfpm"
    "supersimplenet"
    "dfkde"
    "uninet"
    "dsr"
    # --- Unreliable/broken last run - needs debugging before results are meaningful ---
    "draem"                 # NaN issue suspected - inflated AUROC, don't trust until fixed
    "ganomaly"
    "csflow"
    "fre"
    "dfm"
    "generalad"             # AUROC below random last run - likely a score-direction bug
    # --- No historical data on this dataset ---
    "foundad"
)

TOP_MODELS=(
    "anomalyvfm"
    "inpformer"
    "dinomaly"
    "reversedistillation"
    "cfa"
    "efficientad"
    "patchflow"
    "glass"
    "l2bt"
    "generalad"
    "uflow"
)

# Measured wall-clock minutes per model, from the results-rnd-4 full-zoo run.
# Used only to order jobs for node assignment below (longest-first bin-packing) -
# a rough prior, not a guarantee, since real duration depends on dataset size,
# image_size, batch_size, etc. Models with no entry fall back to $DEFAULT_DURATION_MIN.
declare -A MODEL_DURATION_MIN=(
    [cflow]=175 [generalad]=100 [uflow]=90 [draem]=71 [dinomaly]=55
    [ganomaly]=39 [efficientad]=38 [glass]=37 [uninet]=35 [inpformer]=31
    [csflow]=27 [patchflow]=16 [cfa]=16 [supersimplenet]=16 [dsr]=15
    [l2bt]=13 [fastflow]=13 [reversedistillation]=12 [stfpm]=7 [fre]=7
    [anomalydino]=1 [dfkde]=1 [dfm]=1 [padim]=1 [patchcore]=1 [superadd]=1
)
DEFAULT_DURATION_MIN=5  # unknown/unprofiled models (e.g. anomalyvfm's last run errored before logging)

MODELS=()
ZOO_SELECTION="small"

PARTITION="gpu"
NODELIST=""
DATASET="unknown"
TRAIN_ARGS=()
GLOBAL_BATCH_SIZE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    # --- ZOO SELECTION ---
    --zoo)
      case "$2" in
        full) ZOO_SELECTION="full" ;;
        top) ZOO_SELECTION="top" ;;
        *) ZOO_SELECTION="small" ;;
      esac
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
    elif [ "$ZOO_SELECTION" == "top" ]; then
        echo "Using TOP model zoo."
        MODELS=("${TOP_MODELS[@]}")
    else
        echo "Using SMALL model zoo."
        MODELS=("${SMALL_MODELS[@]}")
    fi
fi

# Reorder MODELS longest-known-duration-first, so the node-assignment loop below
# can greedily bin-pack them (LPT heuristic) instead of just alternating by
# array position - array order is arbitrary and can stack several slow models
# onto the same node while the other sits idle.
SORTED_MODELS=()
while IFS=$'\t' read -r _ model; do
    SORTED_MODELS+=("$model")
done < <(
    for m in "${MODELS[@]}"; do
        printf '%s\t%s\n' "${MODEL_DURATION_MIN[$m]:-$DEFAULT_DURATION_MIN}" "$m"
    done | sort -rn -k1,1
)
MODELS=("${SORTED_MODELS[@]}")

mkdir -p logs/slurm

IFS=',' read -r -a NODES_ARRAY <<< "$NODELIST"
NUM_NODES=${#NODES_ARRAY[@]}
declare -a NODE_LOAD_MIN=()
for ((i = 0; i < NUM_NODES; i++)); do NODE_LOAD_MIN[$i]=0; done

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
        # Greedily assign to whichever node currently has the smallest projected load.
        node_index=0
        for ((i = 1; i < NUM_NODES; i++)); do
            if [ "${NODE_LOAD_MIN[$i]}" -lt "${NODE_LOAD_MIN[$node_index]}" ]; then
                node_index=$i
            fi
        done
        selected_node="${NODES_ARRAY[$node_index]}"
        SLURM_NODE_ARG="--nodelist=$selected_node"
        model_duration="${MODEL_DURATION_MIN[$model]:-$DEFAULT_DURATION_MIN}"
        NODE_LOAD_MIN[$node_index]=$((NODE_LOAD_MIN[$node_index] + model_duration))
        echo "Assigning $model (~${model_duration}m) -> $selected_node (node running total: ${NODE_LOAD_MIN[$node_index]}m)"
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
if [ "$NUM_NODES" -gt 0 ]; then
    echo "Projected per-node load (minutes, based on historical/default durations):"
    for ((i = 0; i < NUM_NODES; i++)); do
        echo "  ${NODES_ARRAY[$i]}: ${NODE_LOAD_MIN[$i]}m"
    done
fi