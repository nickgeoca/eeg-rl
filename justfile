default:
    just --list

# Install JS dependencies (run once after cloning)
npm-install:
    npm install

# Kill any Vite process on :8080
_kill-vite:
    #!/usr/bin/env bash
    pid=$(ss -tlnp | grep ':8080' | grep -oP 'pid=\K[0-9]+' | head -1)
    if [ -n "$pid" ]; then
        kill "$pid" && echo "Killed existing Vite (pid $pid)"
    fi

# Start Vite dev server on :8080 (restarts if already running)
_vite: _kill-vite
    #!/usr/bin/env bash
    npx vite >/dev/null 2>&1 &
    disown
    until ss -tlnp | grep -q ':8080'; do sleep 0.2; done
    echo "Vite started → http://localhost:8080/bridge.html"

# Start synthetic Muse BLE bridge (run in a separate terminal, then open bridge.html)
sim:
    cargo run --manifest-path ../elata-bio-sdk/Cargo.toml -p elata-dev-synthetic-ble-bridge

# Online RL training — interactive (asks embedding + fresh/keep)
# Shortcuts: just train mock | just train reve | just train mock-reve
train mode='': _vite _fresh-check
    #!/usr/bin/env bash
    set -euo pipefail
    mode="{{mode}}"
    mock_flag=""
    reve_flag=""

    # --- pre-set flags from mode shortcut ---
    case "$mode" in
        mock)      mock_flag="--mock-eeg" ;;
        reve)      reve_flag="--reve" ;;
        mock-reve) mock_flag="--mock-eeg"; reve_flag="--reve" ;;
        "")        ;;   # ask everything below
        *)         echo "Unknown mode: $mode  (use mock, reve, or mock-reve)"; exit 1 ;;
    esac

    # --- ask embedding if not pre-set ---
    if [ -z "$reve_flag" ]; then
        echo ""
        echo "EEG embedding:"
        echo "  1) Band powers — 2D mood/energy (fast, interpretable)"
        echo "  2) REVE-base   — 69.2M encoder (richer, slower)"
        read -rp "Choice [1]: " emb_choice </dev/tty
        case "${emb_choice:-1}" in
            2) reve_flag="--reve" ;;
        esac
    fi

    exec python eeg_rl_clip.py --bridge $mock_flag $reve_flag

# Inference only — resume saved weights, no training
run: _vite
    exec python eeg_rl_clip.py --bridge --inference-only

# Ask whether to keep or discard session.pt for a fresh run
_fresh-check:
    #!/usr/bin/env bash
    if [ -f session.pt ]; then
        echo ""
        echo "Existing session found (session.pt)."
        read -rp "Keep previous weights? [y/N]: " keep </dev/tty
        case "${keep:-}" in
            [yY]*) echo "Resuming from existing weights." ;;
            *)
                rm -f session.pt session_log.csv
                echo "Discarded — starting with fresh weights."
                ;;
        esac
    else
        echo "No previous session found — starting fresh."
    fi

# Bake base Gemma embedding (run once before training)
bake:
    python bake_base_embedding.py

# Test REVE encoder output shape
test-reve:
    python reve.py

# MountainCar proxy sim (no image gen, fast RL loop)
mountaincar:
    python mountaincar_eeg_sim.py
