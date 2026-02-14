#!/bin/bash

SMALL_MODELS=(
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

FULL_MODELS=(
    "anomalydino"
    "cfa"
    "cflow"
    "csflow"
    "dfkde"
    "dfm"
    "dinomaly"
    "draem"
    "dsr"
    "efficientad"
    "fastflow"
    "fre"
    "ganomaly"
    "padim"
    "patchcore"
    "reversedistillation"
    "stfpm"
    "supersimplenet"
    "uflow"
    "uninet"
)


MODELS=()
ZOO_SELECTION="small" 

PARTITION="waccamaw"
NODELIST=""
DATASET="unknown"
TRAIN_ARGS=()

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
    
    --dataset)
      DATASET="$2"
      TRAIN_ARGS+=("$1" "$2")
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