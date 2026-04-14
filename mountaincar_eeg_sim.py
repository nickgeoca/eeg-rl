"""
MountainCar EEG-RL Simulation (NumPy only)
===========================================
Full EEG→embedding→RL pipeline using MountainCar as a proxy.
No PyTorch — runs on Termux/Android.

Pipeline:
  1. Collect random transitions
  2. Train Encoder + WorldModel (supervised, SGD)
  3. Train Policy with REINFORCE inside the world model
  4. Evaluate in the real environment
  5. Plot loss curves in terminal with plotext
"""

import gymnasium as gym
import numpy as np
import plotext as plt

SEED = 42
rng  = np.random.default_rng(SEED)

EMB_DIM   = 8
N_ACTIONS = 3
STATE_DIM = 2

# ---------------------------------------------------------------------------
# Tiny numpy MLP helpers
# ---------------------------------------------------------------------------

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def init_layer(in_dim, out_dim):
    scale = np.sqrt(2.0 / in_dim)
    return {"W": rng.normal(0, scale, (in_dim, out_dim)),
            "b": np.zeros(out_dim)}

def linear(x, layer):
    return x @ layer["W"] + layer["b"]

def mlp_forward(x, layers):
    for layer in layers[:-1]:
        x = relu(linear(x, layer))
    return linear(x, layers[-1])


# ---------------------------------------------------------------------------
# Models (list of layer dicts)
# ---------------------------------------------------------------------------

def make_encoder():
    return [init_layer(STATE_DIM, 32),
            init_layer(32, 32),
            init_layer(32, EMB_DIM)]

def make_world_model():
    return [init_layer(EMB_DIM + N_ACTIONS, 64),
            init_layer(64, 64),
            init_layer(64, EMB_DIM)]

def make_policy():
    return [init_layer(EMB_DIM, 64),
            init_layer(64, 64),
            init_layer(64, N_ACTIONS)]

def get_params(layers):
    return [(l["W"], l["b"]) for l in layers]

def all_weights(layers):
    return [l["W"] for l in layers] + [l["b"] for l in layers]


# ---------------------------------------------------------------------------
# Numerical gradient helper (finite differences)
# ---------------------------------------------------------------------------

def numerical_grad(loss_fn, layers, eps=1e-4):
    grads = [{"W": np.zeros_like(l["W"]), "b": np.zeros_like(l["b"])}
             for l in layers]
    base  = loss_fn(layers)
    for li, layer in enumerate(layers):
        for key in ("W", "b"):
            arr = layer[key]
            for idx in np.ndindex(arr.shape):
                arr[idx] += eps
                plus  = loss_fn(layers)
                arr[idx] -= 2 * eps
                minus = loss_fn(layers)
                arr[idx] += eps
                grads[li][key][idx] = (plus - minus) / (2 * eps)
    return grads, base


# ---------------------------------------------------------------------------
# Backprop for a simple ReLU MLP (manual)
# ---------------------------------------------------------------------------

def mlp_forward_cache(x, layers):
    """Returns output and cache of (pre-act, post-act) for backprop."""
    cache = []
    h = x
    for i, layer in enumerate(layers):
        z = linear(h, layer)
        if i < len(layers) - 1:
            a = relu(z)
        else:
            a = z  # last layer: no activation
        cache.append((h, z, a))
        h = a
    return h, cache

def mlp_backward(d_out, layers, cache):
    """Returns (grads, d_input) — gradient wrt each layer and wrt the input."""
    grads = [{"W": None, "b": None} for _ in layers]
    d = d_out
    for i in reversed(range(len(layers))):
        h_in, z, a = cache[i]
        if i < len(layers) - 1:
            d = d * (z > 0)          # relu backward
        grads[i]["W"] = h_in.T @ d
        grads[i]["b"] = d.sum(axis=0)
        d = d @ layers[i]["W"].T
    return grads, d  # d is now grad wrt input

def apply_grads(layers, grads, lr):
    for layer, g in zip(layers, grads):
        layer["W"] -= lr * g["W"]
        layer["b"] -= lr * g["b"]


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

def encode(states, enc_layers):
    return mlp_forward(states, enc_layers)

def train_world_model(transitions, epochs=60, batch_size=256, lr=3e-3):
    enc = make_encoder()
    wm  = make_world_model()

    states      = np.array([t[0] for t in transitions], dtype=np.float32)
    actions     = np.array([t[1] for t in transitions], dtype=np.int32)
    next_states = np.array([t[2] for t in transitions], dtype=np.float32)
    n = len(transitions)

    a_onehot_all = np.zeros((n, N_ACTIONS), dtype=np.float32)
    a_onehot_all[np.arange(n), actions] = 1.0

    print(f"\n[Phase 2] Training encoder + world model for {epochs} epochs…")
    losses = []
    for epoch in range(epochs):
        idx        = rng.permutation(n)
        epoch_loss = 0.0
        batches    = 0
        for start in range(0, n, batch_size):
            b  = idx[start:start + batch_size]
            s  = states[b]
            ns = next_states[b]
            ao = a_onehot_all[b]

            # Forward
            emb,      enc_cache = mlp_forward_cache(s,  enc)
            next_emb_true, _    = mlp_forward_cache(ns, enc)
            wm_in               = np.concatenate([emb, ao], axis=-1)
            next_emb_pred, wm_cache = mlp_forward_cache(wm_in, wm)

            # MSE loss
            diff = next_emb_pred - next_emb_true
            loss = (diff ** 2).mean()
            epoch_loss += loss
            batches    += 1

            # Backward through world model
            d_pred = 2 * diff / diff.size
            wm_grads, d_wm_in = mlp_backward(d_pred, wm, wm_cache)
            apply_grads(wm, wm_grads, lr)

            # Backward through encoder (from prediction side)
            d_emb = d_wm_in[:, :EMB_DIM]     # only embedding part, ignore action dims
            enc_grads, _ = mlp_backward(d_emb, enc, enc_cache)
            apply_grads(enc, enc_grads, lr)

        avg = epoch_loss / batches
        losses.append(float(avg))
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  MSE={avg:.5f}")

    return enc, wm, losses


# ---------------------------------------------------------------------------
# Phase 3: train policy with REINFORCE inside world model
# ---------------------------------------------------------------------------

GOAL_STATE = np.array([[0.45, 0.0]], dtype=np.float32)

def compute_reward(emb, goal_emb):
    dist = np.linalg.norm(emb - goal_emb, axis=-1)
    return -dist

def train_policy(enc, wm, epochs=200, rollout_len=20, batch_size=64, lr=1e-3):
    pol = make_policy()
    env = gym.make("MountainCar-v0")

    goal_emb = encode(GOAL_STATE, enc)

    print(f"\n[Phase 3] Training policy inside world model for {epochs} epochs…")
    losses = []
    for epoch in range(epochs):
        start_obs = np.array([
            env.reset(seed=SEED + epoch * batch_size + i)[0]
            for i in range(batch_size)
        ], dtype=np.float32)

        emb = encode(start_obs, enc)

        # Storage for REINFORCE
        saved_log_probs = []  # list of (batch,) arrays
        total_reward    = np.zeros(batch_size, dtype=np.float32)

        # We also need policy caches for backprop
        pol_caches  = []
        action_list = []
        prob_list   = []

        for _ in range(rollout_len):
            logits, cache = mlp_forward_cache(emb, pol)
            probs         = softmax(logits)

            # Sample actions
            cum   = probs.cumsum(axis=-1)
            u     = rng.random((batch_size, 1))
            acts  = (u > cum).sum(axis=-1).clip(0, N_ACTIONS - 1)

            log_p = np.log(probs[np.arange(batch_size), acts] + 1e-8)
            saved_log_probs.append(log_p)
            pol_caches.append(cache)
            action_list.append(acts)
            prob_list.append(probs)

            # Step world model
            ao       = np.zeros((batch_size, N_ACTIONS), dtype=np.float32)
            ao[np.arange(batch_size), acts] = 1.0
            wm_in    = np.concatenate([emb, ao], axis=-1)
            next_emb = mlp_forward(wm_in, wm)

            r = compute_reward(next_emb, goal_emb)
            total_reward += r
            emb = next_emb

        returns = total_reward / rollout_len  # (batch,)
        losses.append(float(-returns.mean()))

        # REINFORCE gradient
        pol_grads = [{"W": np.zeros_like(l["W"]), "b": np.zeros_like(l["b"])}
                     for l in pol]

        for t in range(rollout_len):
            probs  = prob_list[t]           # (batch, 3)
            acts   = action_list[t]         # (batch,)
            cache  = pol_caches[t]
            R      = returns                # (batch,)

            # Gradient of -log_prob * R  wrt logits (softmax cross-entropy style)
            d_logits         = probs.copy()
            d_logits[np.arange(batch_size), acts] -= 1.0
            d_logits         *= -R[:, None] / (rollout_len * batch_size)

            step_grads, _ = mlp_backward(d_logits, pol, cache)
            for li in range(len(pol)):
                pol_grads[li]["W"] += step_grads[li]["W"]
                pol_grads[li]["b"] += step_grads[li]["b"]

        apply_grads(pol, pol_grads, lr)

        if (epoch + 1) % 40 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  avg_reward={returns.mean():.4f}")

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
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        total_r = 0.0
        terminated = False
        for _ in range(max_steps):
            emb    = encode(obs[None], enc)
            logits = mlp_forward(emb, pol)
            action = int(np.argmax(softmax(logits), axis=-1)[0])
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
