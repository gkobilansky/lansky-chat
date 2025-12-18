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

# Config
SHADEFORM_API="https://api.shadeform.ai/v1"
HF_REPO="gkobilansky/lanbot-checkpoints"
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
    INSTANCES_DATA="$INSTANCES" SHADEFORM_API_KEY="$SHADEFORM_API_KEY" SHADEFORM_API="$SHADEFORM_API" SSH_KEYS_DIR="$SSH_KEYS_DIR" python3 << 'PYEOF'
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

if not lanbot_instances:
    print("No active lanbot instances found.")
    print("")
    print("This means either:")
    print("  - Training hasn't started yet")
    print("  - Training completed and instance self-destructed")
    print("  - Instance was manually deleted")
    print("")
    print("Check HuggingFace for results:")
    print("  hf repo files gkobilansky/lanbot-checkpoints")
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

    # Check HuggingFace for recent uploads
    echo "--- HuggingFace Repo Status ---"
    echo ""

    # Check if hf CLI is available
    if command -v hf &> /dev/null; then
        # List recent files (mid_checkpoints and logs)
        echo "Checking for training outputs..."

        # Try to list mid_checkpoints
        HF_FILES=$(hf repo files "$HF_REPO" 2>/dev/null || echo "")

        if echo "$HF_FILES" | grep -q "mid_checkpoints"; then
            echo "  mid_checkpoints/: FOUND"
        else
            echo "  mid_checkpoints/: not yet"
        fi

        if echo "$HF_FILES" | grep -q "logs/"; then
            echo "  logs/: FOUND"
        else
            echo "  logs/: not yet"
        fi
    else
        echo "  (hf CLI not found - check manually at https://huggingface.co/$HF_REPO)"
    fi

    echo ""
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
