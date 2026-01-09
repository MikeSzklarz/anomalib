#!/bin/bash

# =============================================================================
# SCRIPT: submit_sweep.sh
# PURPOSE: Orchestrate a sweep of Anomalib models across specific Slurm nodes.
# USAGE: 
#   ./submit_sweep.sh --root_dir /path/to/data --dataset folder --nodelist waccamaw01,waccamaw02
# =============================================================================

# 1. HARDCODED MODEL LIST
MODELS=(
    # --- TIER 1: Zero-Shot / Statistical (Minutes) ---
    # These effectively have 0 "training" epochs; they just calculate stats.
    "padim"                # One-pass statistical modeling
    "dfm"                  # Deep Feature Modeling (PCA-based)
    "patchcore"            # Memory bank coreset subsampling
    "dfkde"                # Deep Feature Kernel Density Estimation

    # --- TIER 2: Efficient Distillation (Short Training) ---
    # Student-Teacher networks that converge relatively fast.
    "efficientad"          # Designed specifically for efficiency
    "stfpm"                # Student-Teacher Feature Pyramid
    "fre"                  # Feature Reconstruction Error
    "reversedistillation"  # Knowledge distillation
    "supersimplenet"       # Feature adaption (often faster than flows)

    # --- TIER 3: Normalizing Flows & Hybrids (Medium Training) ---
    # These require optimizing flow steps or coupled hyperspheres.
    "fastflow"             # 2D Flow (Faster than CFLOW)
    "cfa"                  # Coupled-hypersphere Autoencoder
    "csflow"               # Cross-Scale Flow
    "uflow"                # Unsupervised Flow
    "ganomaly"             # Encoder-Decoder-Encoder GAN

    # --- TIER 4: Reconstruction & Synthetic Generation (Slow Training) ---
    # High epoch counts needed for convergence or synthetic generation.
    "draem"                # Needs to generate synthetic anomalies on-the-fly
    "cflow"                # Positional encoding flow (often slow convergence)
    "dsr"                  # Dual-Subspace Reprojection

    # --- TIER 5: Complex / Transformer Backbones (Slowest) ---
    # Large backbones or complex attention mechanisms.
    "uninet"               # Unified Network
    "anomalydino"          # DINO-based (Large backbone)
    "dinomaly"             # DINO-based variants
)

# 2. DEFAULT ARGUMENTS
ROOT_DIR="../data"
DATASET="folder"
MAX_EPOCHS=999
NODELIST=""
CATEGORY=""
PARTITION="waccamaw"

# 3. PARSE CLI ARGUMENTS
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root_dir)       ROOT_DIR="$2"; shift 2 ;;
    --dataset)        DATASET="$2"; shift 2 ;;
    --max_epochs)     MAX_EPOCHS="$2"; shift 2 ;;
    --category)       CATEGORY="$2"; shift 2 ;;
    --nodelist)       NODELIST="$2"; shift 2 ;; # Expects comma separated: node1,node2
    --partition)      PARTITION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# 4. PREPARATION
# Fix the log dir issue: Create it explicitly before sbatch is ever called.
mkdir -p logs/slurm
echo "Created logs/slurm directory."

# Parse Nodes for Round Robin
IFS=',' read -r -a NODES_ARRAY <<< "$NODELIST"
NUM_NODES=${#NODES_ARRAY[@]}

echo "========================================"
echo "Starting Sweep"
echo "Models: ${#MODELS[@]}"
echo "Dataset: $DATASET"
echo "Root: $ROOT_DIR"
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
        # Use modulo operator to cycle through nodes
        node_index=$((count % NUM_NODES))
        selected_node="${NODES_ARRAY[$node_index]}"
        SLURM_NODE_ARG="--nodelist=$selected_node"
        echo "Assigning $model -> $selected_node"
    fi

    # --- Construct Job Name ---
    JOB_NAME="${model}_${DATASET}"

    # --- Submit Job ---
    # We pass the python args as command line arguments to the sbatch script
    sbatch \
        --job-name="$JOB_NAME" \
        --partition="$PARTITION" \
        $SLURM_NODE_ARG \
        run_slurm.sh \
        --model "$model" \
        --dataset "$DATASET" \
        --root_dir "$ROOT_DIR" \
        --max_epochs "$MAX_EPOCHS" \
        --category "$CATEGORY"

    ((count++))
done

echo "Successfully submitted $count jobs."