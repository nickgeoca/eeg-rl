default:
    just --list

# Online RL training loop with real Muse 2 EEG via browser bridge
train *args="":
    python eeg_rl_clip.py --bridge {{args}}

# Inference only — run a saved policy without updating weights (TODO: wire --inference-only flag)
run *args="":
    python eeg_rl_clip.py --bridge --no-dyna {{args}}

# Train with mock EEG (2D mood/energy coords)
train-mock:
    python eeg_rl_clip.py --mock-eeg

# Train with REVE-base EEG encoder (69.2M) + mock EEG
train-reve:
    python eeg_rl_clip.py --mock-eeg --reve

# Train with mock EEG, fullscreen display
train-fs:
    python eeg_rl_clip.py --mock-eeg --fullscreen

# Train with REVE + mock EEG, fullscreen
train-reve-fs:
    python eeg_rl_clip.py --mock-eeg --reve --fullscreen

# Train without dreaming (pure online REINFORCE)
train-nodyna:
    python eeg_rl_clip.py --mock-eeg --no-dyna

# Bake base Gemma embedding (run once before training)
bake:
    python bake_base_embedding.py

# Test REVE encoder output shape
test-reve:
    python reve.py

# MountainCar proxy sim (no image gen, fast RL loop)
mountaincar:
    python mountaincar_eeg_sim.py
