"""
MountainCar EEG-RL Simulation (PyTorch)
========================================
Full EEG→embedding→RL pipeline using MountainCar as a proxy.

Pipeline:
  1. Collect random transitions
  2. Train Encoder + WorldModel (supervised, Adam)
  3. Train Policy with REINFORCE inside the world model
  4. Evaluate in the real environment
  5. Plot loss curves in terminal with plotext
"""

import gymnasium as gym
import numpy as np
import plotext as plt
import torch
import torch.nn as nn
from torch.distributions import Categorical

SEED = 42
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

EMB_DIM   = 8
N_ACTIONS = 3
STATE_DIM = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}" + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_encoder() -> nn.Module:
    return nn.Sequential(
        nn.Linear(STATE_DIM, 32), nn.ReLU(),
        nn.Linear(32, 32),        nn.ReLU(),
        nn.Linear(32, EMB_DIM),
    ).to(DEVICE)

def make_world_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(EMB_DIM + N_ACTIONS, 64), nn.ReLU(),
        nn.Linear(64, 64),                  nn.ReLU(),
        nn.Linear(64, EMB_DIM),
    ).to(DEVICE)

def make_policy() -> nn.Module:
    return nn.Sequential(
        nn.Linear(EMB_DIM, 64), nn.ReLU(),
        nn.Linear(64, 64),      nn.ReLU(),
        nn.Linear(64, N_ACTIONS),
    ).to(DEVICE)


# ---------------------------------------------------------------------------
# Phase 1: collect transitions
# ---------------------------------------------------------------------------

def collect_transitions(n_episodes=300, max_steps=200):
    env = gym.make("MountainCar-v0")
    transitions = []
    print(f"[Phase 1] Collecting {n_episodes} episodes of random transitions…")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED + ep)
        for _ in range(max_steps):
            action = env.action_space.sample()
            next_obs, _, terminated, truncated, _ = env.step(action)
            transitions.append((obs.copy(), action, next_obs.copy()))
            obs = next_obs
            if terminated or truncated:
                break
    env.close()
    print(f"  Collected {len(transitions)} transitions")
    return transitions


# ---------------------------------------------------------------------------
# Phase 2: train encoder + world model
# ---------------------------------------------------------------------------

def train_world_model(transitions, epochs=60, batch_size=256, lr=3e-3):
    enc = make_encoder()
    wm  = make_world_model()
    optimizer = torch.optim.Adam(list(enc.parameters()) + list(wm.parameters()), lr=lr)

    states      = torch.tensor([t[0] for t in transitions], dtype=torch.float32, device=DEVICE)
    actions     = torch.tensor([t[1] for t in transitions], dtype=torch.long,    device=DEVICE)
    next_states = torch.tensor([t[2] for t in transitions], dtype=torch.float32, device=DEVICE)
    n = len(transitions)

    a_onehot_all = torch.zeros(n, N_ACTIONS, device=DEVICE)
    a_onehot_all.scatter_(1, actions.unsqueeze(1), 1.0)

    print(f"\n[Phase 2] Training encoder + world model for {epochs} epochs…")
    losses = []
    for epoch in range(epochs):
        idx = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0
        batches    = 0
        for start in range(0, n, batch_size):
            b  = idx[start:start + batch_size]
            s  = states[b]
            ns = next_states[b]
            ao = a_onehot_all[b]

            emb           = enc(s)
            next_emb_true = enc(ns).detach()
            next_emb_pred = wm(torch.cat([emb, ao], dim=-1))

            loss = nn.functional.mse_loss(next_emb_pred, next_emb_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches    += 1

        avg = epoch_loss / batches
        losses.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  MSE={avg:.5f}")

    return enc, wm, losses


# ---------------------------------------------------------------------------
# Phase 3: train policy with REINFORCE inside world model
# ---------------------------------------------------------------------------

GOAL_STATE = torch.tensor([[0.45, 0.0]], dtype=torch.float32, device=DEVICE)

def train_policy(enc, wm, epochs=200, rollout_len=20, batch_size=64, lr=1e-3):
    pol = make_policy()
    optimizer = torch.optim.Adam(pol.parameters(), lr=lr)
    env = gym.make("MountainCar-v0")

    with torch.no_grad():
        goal_emb = enc(GOAL_STATE)  # (1, EMB_DIM)

    print(f"\n[Phase 3] Training policy inside world model for {epochs} epochs…")
    losses = []
    for epoch in range(epochs):
        start_obs = torch.tensor(
            np.array([env.reset(seed=SEED + epoch * batch_size + i)[0]
                      for i in range(batch_size)], dtype=np.float32),
            device=DEVICE,
        )

        with torch.no_grad():
            emb = enc(start_obs)

        log_probs    = []
        total_reward = torch.zeros(batch_size, device=DEVICE)

        for _ in range(rollout_len):
            logits = pol(emb)
            dist   = Categorical(logits=logits)
            acts   = dist.sample()
            log_probs.append(dist.log_prob(acts))

            ao = torch.zeros(batch_size, N_ACTIONS, device=DEVICE)
            ao.scatter_(1, acts.unsqueeze(1), 1.0)
            with torch.no_grad():
                emb = wm(torch.cat([emb, ao], dim=-1))

            dist_to_goal = torch.norm(emb - goal_emb, dim=-1)
            total_reward += -dist_to_goal

        returns = total_reward / rollout_len  # (batch,)
        losses.append(-returns.mean().item())

        # REINFORCE loss: -E[log_pi * R]
        log_probs_stack = torch.stack(log_probs, dim=1)  # (batch, rollout_len)
        policy_loss = -(log_probs_stack * returns.unsqueeze(1)).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if (epoch + 1) % 40 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  avg_reward={returns.mean().item():.4f}")

    env.close()
    return pol, losses


# ---------------------------------------------------------------------------
# Phase 4: evaluate in real environment
# ---------------------------------------------------------------------------

def evaluate_policy(enc, pol, n_episodes=10, max_steps=200):
    env = gym.make("MountainCar-v0")
    successes = 0
    rewards   = []
    print(f"\n[Phase 4] Evaluating policy for {n_episodes} episodes…")
    enc.eval()
    pol.eval()
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=1000 + ep)
            total_r = 0.0
            terminated = False
            for _ in range(max_steps):
                obs_t  = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
                emb    = enc(obs_t)
                logits = pol(emb)
                action = int(logits.argmax(dim=-1).item())
                obs, r, terminated, truncated, _ = env.step(action)
                total_r += r
                if terminated or truncated:
                    break
            if terminated:
                successes += 1
            rewards.append(total_r)
            status = "SUCCESS" if terminated else "timeout"
            print(f"  Episode {ep+1:2d}: reward={total_r:.1f}  [{status}]")
    env.close()
    print(f"\n  Success rate: {successes}/{n_episodes}  |  avg reward: {np.mean(rewards):.1f}")
    return rewards


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_losses(wm_losses, pol_losses):
    plt.clf()
    plt.plot(wm_losses)
    plt.title("World Model Training Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()

    plt.clf()
    plt.plot(pol_losses)
    plt.title("Policy Training Loss (neg reward)")
    plt.xlabel("Epoch")
    plt.ylabel("-avg_reward")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transitions = collect_transitions(n_episodes=300)
    enc, wm, wm_losses = train_world_model(transitions, epochs=60)
    pol, pol_losses    = train_policy(enc, wm, epochs=200)
    rewards            = evaluate_policy(enc, pol, n_episodes=10)
    plot_losses(wm_losses, pol_losses)
    print("\nDone!")
