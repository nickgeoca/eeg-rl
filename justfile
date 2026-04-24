default:
    just --list

# Run neurofeedback loop (default: real EEG + band powers). Pass --mock-eeg and/or --reve to override.
run *args="":
    python eeg_rl_clip.py --bridge {{args}}

# Run the full neurofeedback loop with mock EEG (2D mood/energy coords)
sim:
    python eeg_rl_clip.py --mock-eeg

# Run with REVE-base EEG encoder (69.2M) + mock EEG
sim-reve:
    python eeg_rl_clip.py --mock-eeg --reve

# Run fullscreen
sim-fs:
    python eeg_rl_clip.py --mock-eeg --fullscreen

# Run fullscreen with REVE
sim-reve-fs:
    python eeg_rl_clip.py --mock-eeg --reve --fullscreen

# Run without dreaming (pure online REINFORCE)
sim-nodyna:
    python eeg_rl_clip.py --mock-eeg --no-dyna

# Bake base Gemma embedding (run once before sim)
bake:
    python bake_base_embedding.py

# Test REVE encoder output shape
test-reve:
    python reve.py

# MountainCar proxy sim (no image gen, fast RL loop)
mountaincar:
    python mountaincar_eeg_sim.py
