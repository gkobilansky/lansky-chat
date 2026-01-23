#!/bin/bash
#
# Check status of Shadeform training instance
#
# Usage:
#   ./scripts/check_training_status.sh              # One-time check
#   ./scripts/check_training_status.sh --watch      # Poll every 60s
#   ./scripts/check_training_status.sh --watch 30   # Poll every 30s
#
set -e

# Load .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

# Config (override via .env)
SHADEFORM_API="https://api.shadeform.ai/v1"
HF_REPO="${HF_REPO:-gkobilansky/lanbot-checkpoints}"
SSH_KEYS_DIR="$REPO_ROOT/.ssh_keys"

# Create SSH keys directory if needed
mkdir -p "$SSH_KEYS_DIR"
chmod 700 "$SSH_KEYS_DIR"

# Parse args
WATCH_MODE=false
POLL_INTERVAL=60

if [ "$1" = "--watch" ]; then
    WATCH_MODE=true
    if [ -n "$2" ]; then
        POLL_INTERVAL="$2"
    fi
fi

# Validation
if [ -z "$SHADEFORM_API_KEY" ]; then
    echo "ERROR: SHADEFORM_API_KEY not set"
    exit 1
fi

check_status() {
    echo "========================================"
    echo "Training Status Check - $(date)"
    echo "========================================"
    echo ""

    # Get all instances
    INSTANCES=$(curl -s -X GET "$SHADEFORM_API/instances" \
        -H "X-API-KEY: $SHADEFORM_API_KEY")

    # Parse and display
    INSTANCE_OUTPUT="$(INSTANCES_DATA="$INSTANCES" SHADEFORM_API_KEY="$SHADEFORM_API_KEY" SHADEFORM_API="$SHADEFORM_API" SSH_KEYS_DIR="$SSH_KEYS_DIR" python3 << 'PYEOF'
import os
import json
import urllib.request
import urllib.error
import stat

data = json.loads(os.environ['INSTANCES_DATA'])
instances = data.get('instances', [])
api_key = os.environ.get('SHADEFORM_API_KEY', '')
api_base = os.environ.get('SHADEFORM_API', '')
ssh_keys_dir = os.environ.get('SSH_KEYS_DIR', '.ssh_keys')

def get_ssh_key_info(ssh_key_id):
    """Fetch SSH key details from Shadeform API"""
    if not ssh_key_id or not api_key:
        return None, None
    try:
        url = f"{api_base}/sshkeys/{ssh_key_id}/info"
        req = urllib.request.Request(url, headers={'X-API-KEY': api_key})
        with urllib.request.urlopen(req, timeout=5) as resp:
            key_data = json.loads(resp.read().decode())
            return key_data.get('name'), key_data.get('private_key')
    except:
        return None, None

def save_ssh_key(instance_name, private_key):
    """Save private key to file with proper permissions"""
    if not private_key:
        return None
    key_path = os.path.join(ssh_keys_dir, f"{instance_name}.pem")
    with open(key_path, 'w') as f:
        f.write(private_key)
    os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)  # 600
    return key_path

# Filter for lanbot instances
lanbot_instances = [i for i in instances if 'lanbot' in i.get('name', '').lower()]

# Determine checkpoint directory from instance name
checkpoint_dir = 'mid_checkpoints'  # default
if lanbot_instances:
    name_lower = lanbot_instances[0].get('name', '').lower()
    if 'agent_rl' in name_lower or 'agent-rl' in name_lower:
        checkpoint_dir = 'agentrl_checkpoints'
    elif 'sft+rl' in name_lower:
        checkpoint_dir = 'chatrl_checkpoints'
    elif 'mid+sft' in name_lower:
        checkpoint_dir = 'chatsft_checkpoints'
    elif 'rl' in name_lower:
        checkpoint_dir = 'chatrl_checkpoints'
    elif 'sft' in name_lower:
        checkpoint_dir = 'chatsft_checkpoints'
    elif 'mid' in name_lower:
        checkpoint_dir = 'mid_checkpoints'

print(f"CHECKPOINT_DIR:{checkpoint_dir}")

if not lanbot_instances:
    print("No active lanbot instances found.")
    print("")
    print("This means either:")
    print("  - Training hasn't started yet")
    print(f"  - Training completed successfully (check {checkpoint_dir}/)")
    print("  - Training failed (check logs/error-*.log)")
    print("  - Instance was manually deleted")
    print("")
    print("CHECKING_HF_LOGS")  # Signal to bash to check HF logs
else:
    print(f"Found {len(lanbot_instances)} lanbot instance(s):\n")

    for inst in lanbot_instances:
        name = inst.get('name', 'unknown')
        status = inst.get('status', 'unknown')
        cloud = inst.get('cloud', 'unknown')
        region = inst.get('region', 'unknown')
        ip = inst.get('ip', 'pending')
        ssh_user = inst.get('ssh_user', 'root')
        ssh_port = inst.get('ssh_port', 22)
        cost_raw = inst.get('cost_estimate', 0) or 0
        cost = float(cost_raw)  # API returns dollars directly
        created = inst.get('created_at', 'unknown')
        ssh_key_id = inst.get('ssh_key_id', '')

        print(f"  Name:    {name}")
        print(f"  Status:  {status}")
        print(f"  Cloud:   {cloud} / {region}")
        print(f"  IP:      {ip}")
        print(f"  Spend:   ${cost:.2f}")
        print(f"  Created: {created}")

        # Get and save SSH key
        key_path = None
        if ssh_key_id:
            ssh_key_name, private_key = get_ssh_key_info(ssh_key_id)
            if ssh_key_name:
                print(f"  SSH Key: {ssh_key_name}")
            if private_key:
                key_path = save_ssh_key(name, private_key)
                print(f"  Key:     {key_path}")

        print("")

        if status == 'active' and ip and ip != 'pending':
            if key_path:
                print(f"  Connect:")
                print(f"    ssh -i {key_path} {ssh_user}@{ip}")
                print(f"  Watch logs:")
                print(f"    ssh -i {key_path} {ssh_user}@{ip} 'tail -f /var/log/lanbot-training.log'")
            else:
                print(f"  Connect:")
                print(f"    ssh {ssh_user}@{ip}")
                print(f"  Watch logs:")
                print(f"    ssh {ssh_user}@{ip} 'tail -f /var/log/lanbot-training.log'")
        print("")
PYEOF
    )"

    # Extract checkpoint directory from Python output
    CHECKPOINT_DIR=$(echo "$INSTANCE_OUTPUT" | grep "^CHECKPOINT_DIR:" | cut -d: -f2)
    if [ -z "$CHECKPOINT_DIR" ]; then
        CHECKPOINT_DIR="mid_checkpoints"  # Fallback
    fi

    # Display Python output (filter out internal signals)
    echo "$INSTANCE_OUTPUT" | grep -v "CHECKING_HF_LOGS" | grep -v "^CHECKPOINT_DIR:"

    # Check if we need to look for error logs (no active instances)
    if echo "$INSTANCE_OUTPUT" | grep -q "CHECKING_HF_LOGS"; then
        echo "--- Checking HuggingFace for Logs ---"
        echo ""

        # List repo files using HuggingFace API (check logs subdirectory)
        HF_ROOT=$(curl -s "https://huggingface.co/api/models/$HF_REPO/tree/main" | \
            python3 -c "import sys,json; [print(f['path']) for f in json.load(sys.stdin)]" 2>/dev/null || echo "")
        HF_LOGS=$(curl -s "https://huggingface.co/api/models/$HF_REPO/tree/main/logs" | \
            python3 -c "import sys,json; [print(f['path']) for f in json.load(sys.stdin)]" 2>/dev/null || echo "")

        if [ -z "$HF_ROOT" ]; then
            echo "  (Could not list HF repo - check manually at https://huggingface.co/$HF_REPO)"
        else
            # Check for error logs first
            ERROR_LOG=$(echo "$HF_LOGS" | grep "logs/error-" | sort -r | head -1)
            SUCCESS_LOG=$(echo "$HF_LOGS" | grep "logs/training-" | sort -r | head -1)

            if [ -n "$SUCCESS_LOG" ]; then
                # Success log exists - training completed
                echo "✓ Found training log: $SUCCESS_LOG"
                if [ -n "$ERROR_LOG" ]; then
                    echo "  (Also found error log from earlier attempt: $ERROR_LOG)"
                fi
                echo ""
                if echo "$HF_ROOT" | grep -q "$CHECKPOINT_DIR"; then
                    echo "✓ $CHECKPOINT_DIR/: FOUND - Training completed successfully!"
                else
                    echo "  $CHECKPOINT_DIR/: not yet uploaded"
                fi
            elif [ -n "$ERROR_LOG" ]; then
                # Only error log, no success - training failed
                echo "⚠️  Found error log: $ERROR_LOG"
                echo ""
                echo "=== Last Error Log Contents ==="
                echo ""
                # Download and display the error log
                TEMP_DIR=$(mktemp -d)
                hf download "$HF_REPO" "$ERROR_LOG" --local-dir "$TEMP_DIR" --quiet 2>/dev/null
                if [ -f "$TEMP_DIR/$ERROR_LOG" ]; then
                    tail -100 "$TEMP_DIR/$ERROR_LOG"
                    rm -rf "$TEMP_DIR"
                else
                    echo "(Could not download error log)"
                fi
                echo ""
            else
                echo "  No logs found yet - training may not have started"
            fi
        fi
        echo ""
    else
        # Active instance exists - show HF status
        echo "--- HuggingFace Repo Status ---"
        echo ""

        echo "Checking for training outputs..."
        HF_FILES=$(curl -s "https://huggingface.co/api/models/$HF_REPO/tree/main" | \
            python3 -c "import sys,json; [print(f['path']) for f in json.load(sys.stdin)]" 2>/dev/null || echo "")

        if [ -z "$HF_FILES" ]; then
            echo "  (Could not list HF repo - check manually at https://huggingface.co/$HF_REPO)"
        else
            if echo "$HF_FILES" | grep -q "$CHECKPOINT_DIR"; then
                echo "  $CHECKPOINT_DIR/: FOUND"
            else
                echo "  $CHECKPOINT_DIR/: not yet"
            fi

            if echo "$HF_FILES" | grep -q "logs/"; then
                echo "  logs/: FOUND"
            else
                echo "  logs/: not yet"
            fi
        fi
        echo ""
    fi
}

# Main
if [ "$WATCH_MODE" = true ]; then
    echo "Watching training status (poll every ${POLL_INTERVAL}s, Ctrl+C to stop)"
    echo ""

    while true; do
        clear
        check_status
        echo "--- Next check in ${POLL_INTERVAL}s (Ctrl+C to stop) ---"
        sleep "$POLL_INTERVAL"
    done
else
    check_status
fi
