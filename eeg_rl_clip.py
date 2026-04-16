"""
EEG Neurofeedback RL — SANA-Sprint + Gemma Embedding Architecture
==================================================================
Closed-loop system: EEG mood/energy coordinates → RL policy → Gemma embedding delta
→ SANA-Sprint 0.6B image generator (one-step) → human visualisation → EEG changes → repeat.

Why SANA-Sprint 0.6B:
  - One-step generation (~0.5s) — fast enough to be a negligible part of the 10s loop
  - Lighter than 1.6B, good for prototyping
  - Accepts prompt_embeds directly, so Gemma is only needed once offline

Base embedding strategy (Option A — offline pre-compute):
  - Run bake_base_embedding.py once to encode BASE_PROMPT with Gemma2 and save base_emb.pt
  - At runtime, load base_emb.pt — Gemma never needs to be on the GPU again
  - RL delta is projected and added to this frozen base embedding each step

Pipeline per step:
  1. Read current (mood, energy) from Elata SDK
  2. Policy takes (current, goal) → low-dim delta vector
  3. Project delta into Gemma embedding space, add to pre-computed base embedding
  4. Feed embedding directly to SANA-Sprint (no text, no Gemma at runtime)
  5. Display image for STEP_SECONDS
  6. Read new (mood, energy), compute reward, store transition
  7. Update world model + policy online

Stubs are marked with: # STUB
"""

import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DELTA_DIM    = 16     # RL action dimensionality (low-dim latent delta)
GEMMA_DIM    = 2048   # Gemma2-2B hidden size (SANA-Sprint's text encoder dim)
SEQ_LEN      = 77     # token sequence length fed to the model (can go up to 300)
COORD_DIM    = 2      # (mood, energy)
POLICY_INPUT = COORD_DIM * 2  # (current_mood, current_energy, goal_mood, goal_energy)

STEP_SECONDS   = 10    # seconds to show each image before reading EEG again
GOAL_INTERVAL  = 30    # seconds before sampling a new random goal
DELTA_MAX_NORM = 2.0   # clip delta magnitude to keep images from jumping wildly
REPLAY_MAXLEN  = 2000  # sliding window replay buffer size
BATCH_SIZE     = 32
LR             = 3e-4

BASE_PROMPT    = "abstract generative art, neutral"  # used only in bake_base_embedding.py
BASE_EMB_PATH  = "base_emb.pt"                       # pre-computed embedding, loaded at runtime


def load_base_embedding() -> torch.Tensor:
    """
    Load the pre-computed Gemma base embedding from disk.
    Generate it once with bake_base_embedding.py before running this script.
    Shape: (1, SEQ_LEN, GEMMA_DIM)
    """
    return torch.load(BASE_EMB_PATH, map_location=DEVICE)


# ---------------------------------------------------------------------------
# EEG source — real hardware or synthetic mock
# Returns (mood, energy) as floats, each roughly in [-1, 1]
# ---------------------------------------------------------------------------

class EEGSource:
    """
    Wraps hardware EEG or a synthetic mock for smoke-testing.

    mood   ∈ [-1, 1]: valence axis (negative=unpleasant, positive=pleasant)
    energy ∈ [-1, 1]: arousal axis (negative=calm/sleepy, positive=excited)
    """

    def __init__(self, mock: bool = False, smooth_window: int = 3):
        self.mock = mock
        self._history: deque = deque(maxlen=smooth_window)
        self._t0 = time.time()

        if not mock:
            # STUB: initialise Elata SDK connection here, e.g.:
            #   import elata
            #   self._elata = elata.connect()
            raise NotImplementedError(
                "Real EEG not connected. Run with --mock-eeg for smoke testing."
            )

    def read(self) -> tuple[float, float]:
        """Return instantaneous (mood, energy) from hardware or mock."""
        if self.mock:
            return self._mock_read()
        # STUB: replace with actual Elata SDK call, e.g.:
        #   state = self._elata.get_state()
        #   return float(state.mood), float(state.energy)
        raise NotImplementedError("Connect Elata SDK here")

    def _mock_read(self) -> tuple[float, float]:
        """
        Synthetic EEG: slow sine-wave drift with small Gaussian noise.
        Period is long enough that 10 s steps see meaningful change.
        """
        t = time.time() - self._t0
        mood   = 0.6 * np.sin(2 * np.pi * t / 60.0) + 0.1 * np.random.randn()
        energy = 0.5 * np.cos(2 * np.pi * t / 90.0) + 0.1 * np.random.randn()
        return float(np.clip(mood, -1, 1)), float(np.clip(energy, -1, 1))

    def smooth(self) -> tuple[float, float]:
        """
        EMA-smoothed reading over the history window.
        Calls read() internally; most-recent sample has highest weight.
        """
        self._history.append(self.read())
        n       = len(self._history)
        moods   = [h[0] for h in self._history]
        energies= [h[1] for h in self._history]
        weights = np.array([0.5 ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        return float(np.dot(weights, moods)), float(np.dot(weights, energies))


# ---------------------------------------------------------------------------
# STUB: Image generator wrapper — SANA-Sprint
# Accepts a Gemma embedding tensor, returns a PIL Image
# ---------------------------------------------------------------------------

class ImageGenerator:
    def __init__(self):
        from diffusers import SanaSprintPipeline

        print("Loading SANA-Sprint 0.6B…")
        self.pipe = SanaSprintPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16,
        ).to(DEVICE)
        # Uncomment to offload heavy blocks to CPU between steps (saves VRAM):
        # self.pipe.enable_model_cpu_offload()
        print("Pipeline ready.")

    def generate(self, prompt_embeds: torch.Tensor) -> Image.Image:
        """
        Generate image from a Gemma embedding (no text involved).

        prompt_embeds: shape (1, SEQ_LEN, GEMMA_DIM)
        Returns: PIL Image
        """
        return self.pipe(
            prompt_embeds=prompt_embeds.to(torch.bfloat16),
            num_inference_steps=2,   # one-step model; 1-2 steps is sufficient
            guidance_scale=4.5,
        ).images[0]


# ---------------------------------------------------------------------------
# RL Models
# ---------------------------------------------------------------------------

class Policy(nn.Module):
    """
    Input:  (cur_mood, cur_energy, goal_mood, goal_energy)  — 4 floats
    Output: delta vector in low-dim space (DELTA_DIM floats, then projected to GEMMA_DIM)
    Uses tanh output so delta lives in [-1, 1] before norm clipping.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(POLICY_INPUT, 64), nn.ReLU(),
            nn.Linear(64, 64),           nn.ReLU(),
            nn.Linear(64, DELTA_DIM),    nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeltaProjector(nn.Module):
    """
    Projects low-dim RL delta up to Gemma embedding space.
    Output is broadcast-added to every token position in the sequence.
    Shape: (1, DELTA_DIM) → (1, 1, GEMMA_DIM) → broadcasts to (1, SEQ_LEN, GEMMA_DIM)
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(DELTA_DIM, GEMMA_DIM)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        # delta: (batch, DELTA_DIM) → (batch, 1, GEMMA_DIM) for broadcasting
        return self.proj(delta).unsqueeze(1)


class WorldModel(nn.Module):
    """
    Predicts next (mood, energy) given current (mood, energy) and delta action.
    Input:  (mood, energy, delta...)  — COORD_DIM + DELTA_DIM
    Output: (next_mood, next_energy)  — COORD_DIM
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(COORD_DIM + DELTA_DIM, 64), nn.ReLU(),
            nn.Linear(64, 64),                     nn.ReLU(),
            nn.Linear(64, COORD_DIM),
        )

    def forward(self, coord: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([coord, delta], dim=-1))


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Sliding window of (coord, delta, next_coord) transitions."""
    def __init__(self, maxlen: int = REPLAY_MAXLEN):
        self.buf = deque(maxlen=maxlen)

    def push(self, coord: np.ndarray, delta: np.ndarray, next_coord: np.ndarray):
        self.buf.append((coord.copy(), delta.copy(), next_coord.copy()))

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.random.choice(len(self.buf), size=n, replace=False)
        items = [self.buf[i] for i in idx]
        coords      = torch.tensor([i[0] for i in items], dtype=torch.float32, device=DEVICE)
        deltas      = torch.tensor([i[1] for i in items], dtype=torch.float32, device=DEVICE)
        next_coords = torch.tensor([i[2] for i in items], dtype=torch.float32, device=DEVICE)
        return coords, deltas, next_coords

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_image(image: Image.Image):
    """
    Show image to the human subject.
    STUB: replace with whatever display method fits your setup.
    Options:
      - image.show()                        simple, opens system viewer
      - pygame window                       fullscreen, more control
      - web server + browser fullscreen     if subject is remote
      - secondary monitor via pyglet        for lab setups
    """
    image.show()  # STUB


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def update_world_model(
    wm: WorldModel,
    opt: torch.optim.Optimizer,
    replay: ReplayBuffer,
) -> float:
    if len(replay) < BATCH_SIZE:
        return 0.0
    coords, deltas, next_coords = replay.sample(BATCH_SIZE)
    pred = wm(coords, deltas)
    loss = nn.functional.mse_loss(pred, next_coords)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def update_policy_reinforce(
    policy: Policy,
    opt: torch.optim.Optimizer,
    log_prob: torch.Tensor,
    reward: float,
):
    """Single-step REINFORCE update."""
    loss = -log_prob * reward
    opt.zero_grad()
    loss.backward()
    opt.step()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(explore_steps: int = 20, mock_eeg: bool = False):
    """
    explore_steps: number of random-action steps before policy takes over.
    mock_eeg:      use synthetic sine-wave EEG instead of real hardware.
    During exploration the world model accumulates data before policy training.
    """
    print(f"Device: {DEVICE}")
    if mock_eeg:
        print("Using mock EEG (sine wave). Pass mock_eeg=False for real hardware.")

    # Init components
    eeg       = EEGSource(mock=mock_eeg)
    generator = ImageGenerator()
    base_emb  = load_base_embedding()  # (1, SEQ_LEN, GEMMA_DIM), frozen, pre-computed offline

    policy    = Policy().to(DEVICE)
    projector = DeltaProjector().to(DEVICE)
    wm        = WorldModel().to(DEVICE)
    replay    = ReplayBuffer()

    pol_opt = torch.optim.Adam(list(policy.parameters()) + list(projector.parameters()), lr=LR)
    wm_opt  = torch.optim.Adam(wm.parameters(), lr=LR)

    # Sample initial goal
    goal = np.random.uniform(-1, 1, size=2).astype(np.float32)
    goal_timer = time.time()
    step = 0

    print("Starting neurofeedback loop. Ctrl-C to stop.")
    print(f"  First goal: mood={goal[0]:.2f}  energy={goal[1]:.2f}")

    while True:
        # --- 1. Read current EEG state ---
        coord = np.array(eeg.read(), dtype=np.float32)  # (mood, energy)

        # --- 2. Refresh goal if interval elapsed ---
        if time.time() - goal_timer > GOAL_INTERVAL:
            goal = np.random.uniform(-1, 1, size=2).astype(np.float32)
            goal_timer = time.time()
            print(f"\n  New goal: mood={goal[0]:.2f}  energy={goal[1]:.2f}")

        # --- 3. Choose action (delta) ---
        obs = torch.tensor(
            np.concatenate([coord, goal]), dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)  # (1, 4)

        if step < explore_steps:
            # Random exploration: uniform delta in [-1, 1]
            delta_t = torch.rand(1, DELTA_DIM, device=DEVICE) * 2 - 1
            log_prob = None
            print(f"  [explore {step+1}/{explore_steps}]", end="")
        else:
            # Policy action with Gaussian noise for exploration
            delta_mean = policy(obs)                        # (1, DELTA_DIM)
            noise      = torch.randn_like(delta_mean) * 0.1
            delta_t    = (delta_mean + noise).detach()
            # log prob for REINFORCE (treating noise as fixed std Gaussian)
            dist     = torch.distributions.Normal(delta_mean, 0.1)
            log_prob = dist.log_prob(delta_t).sum()

        # Clip delta magnitude so images don't jump wildly
        norm = delta_t.norm()
        if norm > DELTA_MAX_NORM:
            delta_t = delta_t * DELTA_MAX_NORM / norm

        # --- 4. Project delta to Gemma space, build final embedding ---
        with torch.no_grad() if step < explore_steps else torch.enable_grad():
            gemma_delta = projector(delta_t)               # (1, 1, GEMMA_DIM)
        final_emb = base_emb + gemma_delta                 # (1, SEQ_LEN, GEMMA_DIM)

        # --- 5. Generate and display image ---
        image = generator.generate(final_emb.detach())
        display_image(image)

        # --- 6. Wait, then read new EEG state ---
        time.sleep(STEP_SECONDS)
        next_coord = np.array(eeg.read(), dtype=np.float32)

        # --- 7. Compute reward: negative distance to goal ---
        reward = -float(np.linalg.norm(next_coord - goal))
        print(f"  step={step}  coord=({coord[0]:.2f},{coord[1]:.2f})"
              f"  next=({next_coord[0]:.2f},{next_coord[1]:.2f})"
              f"  reward={reward:.3f}")

        # --- 8. Store transition ---
        delta_np = delta_t.squeeze(0).cpu().detach().numpy()
        replay.push(coord, delta_np, next_coord)

        # --- 9. Update world model ---
        wm_loss = update_world_model(wm, wm_opt, replay)

        # --- 10. Update policy (once past exploration) ---
        if step >= explore_steps and log_prob is not None:
            update_policy_reinforce(policy, pol_opt, log_prob, reward)

        step += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run(explore_steps=20)
