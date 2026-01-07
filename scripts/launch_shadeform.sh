#!/bin/bash
#
# Launch a Shadeform training instance with auto-delete
#
# Prerequisites:
#   1. Create .env file with SHADEFORM_API_KEY and HF_TOKEN (or export them)
#   2. Upload base checkpoint + tokenizer to HuggingFace first
#
# Usage:
#   ./scripts/launch_shadeform.sh                    # Default: SFT training
#   ./scripts/launch_shadeform.sh --phase sft        # SFT training
#   ./scripts/launch_shadeform.sh --phase rl         # RL training
#   ./scripts/launch_shadeform.sh --phase sft+rl     # SFT then RL
#   ./scripts/launch_shadeform.sh --phase all        # Full pipeline: mid→sft→rl
#   ./scripts/launch_shadeform.sh --gpu B200         # Specific GPU type
#   ./scripts/launch_shadeform.sh --all-configs      # Show all GPU counts (1x,2x,4x,8x)
#   ./scripts/launch_shadeform.sh --spend-limit 200
#   ./scripts/launch_shadeform.sh --dry-run
#
set -e

# ============================================================================
# LOAD .env FILE
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Load .env from repo root if it exists
if [ -f "$REPO_ROOT/.env" ]; then
    echo "Loading environment from $REPO_ROOT/.env"
    set -a  # auto-export all variables
    source "$REPO_ROOT/.env"
    set +a
fi

# ============================================================================
# CONFIGURATION
# ============================================================================

# Shadeform settings
TRAINING_PHASE="${TRAINING_PHASE:-sft}"  # mid, sft, rl, mid+sft, sft+rl, all
RUN_NAME="${RUN_NAME:-lanbot-v3}"        # Training run name
SPEND_LIMIT="${SPEND_LIMIT:-150.00}"     # Auto-delete at this spend
GPU_TYPE="B200"                          # Default GPU type
NUM_GPUS=8                               # Default to 8x GPU for distributed training
SHADEFORM_API="https://api.shadeform.ai/v1"

# These will be set by instance selection
CLOUD=""
REGION=""
INSTANCE_TYPE=""

# Parse args
DRY_RUN=false
ALL_CONFIGS=false
SKIP_SELECTION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            TRAINING_PHASE="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --spend-limit)
            SPEND_LIMIT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --gpu)
            GPU_TYPE="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --all-configs)
            ALL_CONFIGS=true
            shift
            ;;
        --cloud)
            # Allow direct specification for non-interactive use
            CLOUD="$2"
            SKIP_SELECTION=true
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo "  --phase PHASE     Training phase: mid, sft, rl, mid+sft, sft+rl, all (default: sft)"
            echo "  --run-name NAME   Training run name (default: lanbot-v3)"
            echo "  --gpu TYPE        GPU type to search for (default: B200)"
            echo "  --num-gpus N      Number of GPUs (default: 8)"
            echo "  --all-configs     Show all GPU counts, not just 8x"
            echo "  --spend-limit N   Auto-delete spend limit (default: 150)"
            echo "  --dry-run         Show what would be launched without launching"
            echo "  --cloud NAME      Directly specify cloud (skip selection)"
            echo "  --region NAME     Directly specify region"
            echo "  --instance-type T Directly specify instance type"
            exit 1
            ;;
    esac
done

# Validate training phase
case "$TRAINING_PHASE" in
    mid|sft|rl|mid+sft|sft+rl|all) ;;
    *)
        echo "ERROR: Invalid training phase: $TRAINING_PHASE"
        echo "Valid phases: mid, sft, rl, mid+sft, sft+rl, all"
        exit 1
        ;;
esac

# Generate instance name from phase
INSTANCE_NAME="lanbot-${TRAINING_PHASE}-$(date +%Y%m%d-%H%M)"

# ============================================================================
# VALIDATION
# ============================================================================

if [ -z "$SHADEFORM_API_KEY" ]; then
    echo "ERROR: SHADEFORM_API_KEY not set"
    echo "Get your API key from: https://platform.shadeform.ai/settings/api-keys"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set - checkpoint upload will fail"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ============================================================================
# INSTANCE DISCOVERY
# ============================================================================

find_instances() {
    local gpu_type="$1"
    local num_gpus="$2"
    local all_configs="$3"

    echo "Searching for $gpu_type instances..." >&2

    # Fetch all available instance types
    local response
    response=$(curl -s -X GET "$SHADEFORM_API/instances/types" \
        -H "X-API-KEY: $SHADEFORM_API_KEY")

    # Convert bash booleans to Python booleans
    local py_all_configs="False"
    if [ "$all_configs" = "true" ]; then
        py_all_configs="True"
    fi

    # Filter and format instances using Python
    RESPONSE="$response" python3 << PYEOF
import os
import json

data = json.loads(os.environ['RESPONSE'])
instances = data.get('instance_types', [])

gpu_type_filter = '${gpu_type}'.upper()
num_gpus_filter = ${num_gpus}
all_configs = ${py_all_configs}

# Filter for GPU type and availability
filtered = []
for inst in instances:
    gpu = inst.get('gpu_type', '').upper()
    available = inst.get('availability', [])

    # Check if this GPU type matches (case-insensitive, partial match)
    if gpu_type_filter not in gpu:
        continue

    # Check availability
    for avail in available:
        if not avail.get('available', False):
            continue

        num_gpus_inst = inst.get('num_gpus', 0)

        # Filter by GPU count unless --all-configs
        if not all_configs and num_gpus_inst != num_gpus_filter:
            continue

        price = inst.get('hourly_price', 0) / 100  # Convert cents to dollars
        config = inst.get('configuration', {})

        filtered.append({
            'cloud': inst.get('cloud', ''),
            'region': avail.get('region', ''),
            'instance_type': inst.get('shade_instance_type', ''),
            'num_gpus': num_gpus_inst,
            'gpu_type': inst.get('gpu_type', ''),
            'vram_per_gpu': config.get('vram_per_gpu_in_gb', 0),
            'memory_gb': inst.get('memory_in_gb', 0),
            'vcpus': inst.get('vcpus', 0),
            'hourly_price': price,
            'nvlink': inst.get('nvlink', False),
        })

# Sort by price (cheapest first)
filtered.sort(key=lambda x: (x['hourly_price'], -x['num_gpus']))

# Output as JSON array
print(json.dumps(filtered[:10]))  # Top 10 options
PYEOF
}

select_instance() {
    local instances_json="$1"

    # Parse and display options
    local count
    count=$(echo "$instances_json" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")

    if [ "$count" -eq 0 ]; then
        echo ""
        echo "ERROR: No available $GPU_TYPE instances found"
        if [ "$ALL_CONFIGS" = false ]; then
            echo "Try --all-configs to see instances with different GPU counts"
        fi
        exit 1
    fi

    echo "" >&2
    echo "=== Available $GPU_TYPE Instances (sorted by price) ===" >&2
    echo "" >&2

    # Display formatted table
    INSTANCES_JSON="$instances_json" python3 << 'PYEOF'
import os
import json
import sys

instances = json.loads(os.environ['INSTANCES_JSON'])

# Show top 3 (or fewer if not available)
show_count = min(3, len(instances))

print(f"  #  | {'Provider':<12} | {'Region':<15} | {'GPUs':<6} | {'VRAM':<8} | {'Price/hr':<10} | NVLink", file=sys.stderr)
print(f"  ---+{'-'*14}+{'-'*17}+{'-'*8}+{'-'*10}+{'-'*12}+--------", file=sys.stderr)

for i, inst in enumerate(instances[:show_count], 1):
    cloud = inst['cloud'][:12]
    region = inst['region'][:15]
    gpus = f"{inst['num_gpus']}x"
    vram = f"{inst['vram_per_gpu']}GB"
    price = f"${inst['hourly_price']:.2f}"
    nvlink = "Yes" if inst['nvlink'] else "No"

    print(f"  {i}  | {cloud:<12} | {region:<15} | {gpus:<6} | {vram:<8} | {price:<10} | {nvlink}", file=sys.stderr)

if len(instances) > show_count:
    print(f"\n  ... and {len(instances) - show_count} more options available", file=sys.stderr)

# Output count for bash
PYEOF

    echo "" >&2

    # Get user selection
    local max_choice
    max_choice=$(echo "$instances_json" | python3 -c "import sys,json; print(min(3, len(json.load(sys.stdin))))")

    while true; do
        read -p "Select instance [1-$max_choice] or 'q' to quit: " choice

        if [ "$choice" = "q" ] || [ "$choice" = "Q" ]; then
            echo "Cancelled." >&2
            exit 0
        fi

        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$max_choice" ]; then
            break
        fi

        echo "Invalid choice. Enter 1-$max_choice or 'q' to quit." >&2
    done

    # Extract selected instance details
    local selected
    selected=$(echo "$instances_json" | python3 -c "
import sys, json
instances = json.load(sys.stdin)
idx = int('$choice') - 1
inst = instances[idx]
print(f\"{inst['cloud']}|{inst['region']}|{inst['instance_type']}|{inst['num_gpus']}|{inst['hourly_price']:.2f}\")
")

    echo "$selected"
}

# ============================================================================
# INSTANCE SELECTION
# ============================================================================

if [ "$SKIP_SELECTION" = false ]; then
    echo ""
    echo "=== Shadeform Instance Finder ==="
    echo "Looking for: $GPU_TYPE"
    if [ "$ALL_CONFIGS" = true ]; then
        echo "GPU count: all configurations"
    else
        echo "GPU count: ${NUM_GPUS}x"
    fi

    # Find available instances
    INSTANCES_JSON=$(find_instances "$GPU_TYPE" "$NUM_GPUS" "$ALL_CONFIGS")

    # Interactive selection
    SELECTED=$(select_instance "$INSTANCES_JSON")

    # Parse selection
    IFS='|' read -r CLOUD REGION INSTANCE_TYPE SELECTED_GPUS HOURLY_PRICE <<< "$SELECTED"

    echo ""
    echo "Selected: $INSTANCE_TYPE on $CLOUD ($REGION)"
    echo "Price: \$$HOURLY_PRICE/hr (${SELECTED_GPUS}x GPUs)"
    echo ""
fi

# Verify we have required values
if [ -z "$CLOUD" ] || [ -z "$REGION" ] || [ -z "$INSTANCE_TYPE" ]; then
    echo "ERROR: Missing cloud, region, or instance type"
    exit 1
fi

# ============================================================================
# BUILD SCRIPT
# ============================================================================

TRAIN_SCRIPT="$SCRIPT_DIR/shadeform_train.sh"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

echo "=== Launch Configuration ==="
echo "Instance: $INSTANCE_NAME"
echo "Training phase: $TRAINING_PHASE"
echo "Run name: $RUN_NAME"
echo "Cloud: $CLOUD / $REGION"
echo "Type: $INSTANCE_TYPE"
echo "Spend limit: \$$SPEND_LIMIT"
echo ""

# Base64 encode the training script
echo "Encoding training script..."
SCRIPT_B64=$(base64 -i "$TRAIN_SCRIPT")

# ============================================================================
# BUILD API REQUEST
# ============================================================================

# Build JSON payload
read -r -d '' PAYLOAD << EOF || true
{
    "cloud": "$CLOUD",
    "region": "$REGION",
    "shade_instance_type": "$INSTANCE_TYPE",
    "shade_cloud": true,
    "name": "$INSTANCE_NAME",
    "launch_configuration": {
        "type": "script",
        "script_configuration": {
            "base64_script": "$SCRIPT_B64"
        }
    },
    "envs": [
        {"name": "HF_TOKEN", "value": "$HF_TOKEN"},
        {"name": "SHADEFORM_API_KEY", "value": "$SHADEFORM_API_KEY"},
        {"name": "INSTANCE_NAME", "value": "$INSTANCE_NAME"},
        {"name": "TRAINING_PHASE", "value": "$TRAINING_PHASE"},
        {"name": "RUN_NAME", "value": "$RUN_NAME"},
        {"name": "WANDB_API_KEY", "value": "${WANDB_API_KEY:-}"}
    ],
    "auto_delete": {
        "spend_threshold": "$SPEND_LIMIT"
    },
    "alert": {
        "spend_threshold": "$(echo "$SPEND_LIMIT * 0.6" | bc)"
    }
}
EOF

# ============================================================================
# LAUNCH
# ============================================================================

if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN - Would send this request ==="
    echo "$PAYLOAD" | python3 -m json.tool 2>/dev/null || echo "$PAYLOAD"
    exit 0
fi

echo "Launching instance..."
RESPONSE=$(curl -s -X POST "https://api.shadeform.ai/v1/instances/create" \
    -H "X-API-KEY: $SHADEFORM_API_KEY" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")

# Parse response
INSTANCE_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))" 2>/dev/null)

if [ -n "$INSTANCE_ID" ]; then
    echo ""
    echo "=== Instance Created ==="
    echo "ID: $INSTANCE_ID"
    echo "Name: $INSTANCE_NAME"
    echo "Phase: $TRAINING_PHASE"
    echo ""
    echo "Monitor training progress:"
    echo "  ./scripts/check_training_status.sh          # One-time check"
    echo "  ./scripts/check_training_status.sh --watch  # Poll every 60s"
    echo ""
    echo "Web dashboard: https://platform.shadeform.ai/instances"
    echo ""
    echo "Auto-delete at: \$$SPEND_LIMIT spend"
    echo "Alert email at: \$$(echo "$SPEND_LIMIT * 0.6" | bc) spend"
else
    echo "ERROR: Failed to create instance"
    echo "$RESPONSE"
    exit 1
fi
