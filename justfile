default:
    just --list

# Start synthetic Muse BLE bridge (run in a separate terminal, then open bridge.html)
sim:
    cargo run --manifest-path ../elata-bio-sdk/Cargo.toml -p elata-dev-synthetic-ble-bridge

# Online RL training loop (connect real Muse or run `just sim` first)
train *args="":
    python eeg_rl_clip.py --bridge {{args}}

# Inference only — run a saved policy without updating weights
run *args="":
    python eeg_rl_clip.py --bridge --inference-only {{args}}

# Train without dreaming (pure online REINFORCE)
train-nodyna:
    python eeg_rl_clip.py --bridge --no-dyna

# Bake base Gemma embedding (run once before training)
bake:
    python bake_base_embedding.py

# Test REVE encoder output shape
test-reve:
    python reve.py

# MountainCar proxy sim (no image gen, fast RL loop)
mountaincar:
    python mountaincar_eeg_sim.py
