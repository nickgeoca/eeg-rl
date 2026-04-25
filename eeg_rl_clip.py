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
  7. Update world model + SAC policy online

Stubs are marked with: # STUB
"""

import argparse
import asyncio
import base64
import csv
import json
import os
import signal
import threading
import time
from collections import deque
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Load .env if present so HF_TOKEN is available before any model download
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DELTA_DIM    = 16     # RL action dimensionality (low-dim latent delta)
GEMMA_DIM    = 2304   # Gemma2-2B hidden size (SANA-Sprint's text encoder dim)
SEQ_LEN      = 77     # token sequence length fed to the model (can go up to 300)
COORD_DIM    = 2      # (mood, energy) — overridden at runtime when --reve is used

STEP_SECONDS   = 10    # seconds to show each image before reading EEG again
GOAL_INTERVAL  = 30    # seconds before sampling a new random goal
DELTA_MAX_NORM = 2.0   # clip delta magnitude to keep images from jumping wildly
REPLAY_MAXLEN  = 2000  # sliding window replay buffer size
BATCH_SIZE     = 32
LR             = 3e-4

BASE_PROMPT    = "abstract generative art, neutral"  # used only in bake_base_embedding.py
BASE_EMB_PATH  = "base_emb.pt"                       # pre-computed embedding, loaded at runtime

# SAC hyperparameters
SAC_GAMMA          = 0.99              # discount factor
SAC_TAU            = 0.005            # soft target-network update rate
SAC_TARGET_ENTROPY = -float(DELTA_DIM) # entropy target for alpha auto-tuning
LOG_STD_MIN, LOG_STD_MAX = -5, 2

# ---------------------------------------------------------------------------
# Browser bridge — WebSocket server (Vite serves the frontend)
# ---------------------------------------------------------------------------


class MuseBridge:
    """
    WebSocket bridge between the Muse 2 browser page and Python.

    Vite serves bridge.html on port 8080. This class only runs the WebSocket
    server that receives EEG frames and sends images/status back.

    Thread-safe API for the main loop:
        get_window()      → np.ndarray (4, 400) at 200 Hz — blocks until 2 s ready
        send_image(img)   → streams PIL Image to browser as JPEG
        send_status(text) → updates the HUD overlay in the browser
    """

    CHANNELS   = ["TP9", "AF7", "AF8", "TP10"]
    _IN_HZ     = 256
    _WINDOW_IN = int(2.0 * _IN_HZ)  # 512 samples = 2 s at 256 Hz

    def __init__(self, ws_port: int = 8765):
        import subprocess, time, os, signal
        result = subprocess.run(["ss", "-Htlnp", f"sport = :{ws_port}"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if f":{ws_port}" in line and "pid=" in line:
                pid = int(line.split("pid=")[1].split(",")[0])
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.3)
                break
        self._ws_port   = ws_port
        self._buf: deque = deque(maxlen=self._WINDOW_IN * 4)
        self._buf_lock   = threading.Lock()
        self._data_ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connections: set = set()
        self._started = threading.Event()
        self._running = threading.Event()
        self._running.set()  # start in running state

        threading.Thread(target=self._run_ws, daemon=True).start()
        self._started.wait()
        print(f"  Bridge ready → open http://localhost:8080/bridge.html in Brave/Chrome")

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _run_ws(self):
        import websockets

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def _serve():
            async with websockets.serve(self._handler, "localhost", self._ws_port):
                self._started.set()
                await asyncio.Future()

        self._loop.run_until_complete(_serve())

    async def _handler(self, ws):
        self._connections.add(ws)
        # Send current running state so a freshly-opened browser tab syncs up
        await ws.send(json.dumps({"type": "train_state", "running": self._running.is_set()}))
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "eeg_frame":
                    samples = msg["samples"]  # [[tp9,af7,af8,tp10], …]
                    with self._buf_lock:
                        self._buf.extend(samples)
                        if len(self._buf) >= self._WINDOW_IN:
                            self._data_ready.set()
                elif msg.get("type") == "train_stop":
                    self._running.clear()
                    self._broadcast(json.dumps({"type": "train_state", "running": False}))
                elif msg.get("type") == "train_start":
                    self._running.set()
                    self._broadcast(json.dumps({"type": "train_state", "running": True}))
        finally:
            self._connections.discard(ws)

    def wait_if_stopped(self) -> None:
        """Block the training loop while the browser has pressed Stop."""
        self._running.wait()

    # ------------------------------------------------------------------
    # Main-thread API
    # ------------------------------------------------------------------

    def get_window(self) -> np.ndarray:
        """Return the latest 2 s EEG window resampled to 200 Hz. Shape: (4, 400)."""
        from reve import resample_to_200hz
        self._data_ready.wait()
        with self._buf_lock:
            tail = list(self._buf)[-self._WINDOW_IN:]
        arr = np.array(tail, dtype=np.float32).T  # (4, 512)
        return resample_to_200hz(arr)              # (4, 400)

    def send_image(self, img: "Image.Image") -> None:
        if not self._connections or self._loop is None:
            return
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        data = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        self._broadcast(json.dumps({"type": "image", "data": data}))

    def send_status(self, text: str) -> None:
        if not self._connections or self._loop is None:
            return
        self._broadcast(json.dumps({"type": "status", "text": text}))

    def send_metrics(
        self,
        step: int,
        coord: "np.ndarray",
        reward: float,
        wm_loss: float,
        actor_loss: "float | None",
        q_loss: "float | None",
    ) -> None:
        if not self._connections or self._loop is None:
            return
        self._broadcast(json.dumps({
            "type": "metrics",
            "step": step,
            "reward": round(float(reward), 4),
            "wm_loss": round(float(wm_loss), 4) if wm_loss is not None else None,
            "actor_loss": round(float(actor_loss), 4) if actor_loss is not None else None,
            "q_loss": round(float(q_loss), 4) if q_loss is not None else None,
            "coord": coord.tolist()[:8],
        }))

    def _broadcast(self, msg: str) -> None:
        for ws in list(self._connections):
            asyncio.run_coroutine_threadsafe(ws.send(msg), self._loop)

GOAL_RADIUS_INIT = 0.3   # initial goal sampling radius around current EEG state
GOAL_RADIUS_MAX  = 1.0   # maximum radius (expanded after each resample)
GOAL_RADIUS_GROW = 0.05  # radius added per resample as system gains confidence


def sample_goal_near(current: np.ndarray, radius: float) -> np.ndarray:
    """
    Sample a goal within `radius` of `current`, clipped to [-1, 1].
    Starting near the current state avoids unachievable early goals.
    """
    offset = np.random.uniform(-radius, radius, size=current.shape).astype(np.float32)
    return np.clip(current + offset, -1.0, 1.0)


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

    mood   ∈ [-1, 1]: alpha dominance (relaxed/pleasant vs tense)
    energy ∈ [-1, 1]: beta dominance (alert/excited vs calm)

    With a MuseBridge: derives mood/energy from band powers of the raw EEG window.
    Without a bridge:  mock sine-wave drift.
    """

    def __init__(self, mock: bool = False, smooth_window: int = 3, bridge=None):
        self.mock = mock
        self._bridge = bridge
        self._history: deque = deque(maxlen=smooth_window)
        self._t0 = time.time()

        if not mock and bridge is None:
            raise NotImplementedError(
                "Pass a MuseBridge (--bridge) or use --mock-eeg for smoke testing."
            )

    def read(self) -> np.ndarray:
        """Return instantaneous (mood, energy) as a 1-D float32 array."""
        if self.mock:
            return self._mock_read()
        return self._band_power_read()

    def _band_power_read(self) -> np.ndarray:
        """Derive (mood, energy) from alpha/beta band powers of the EEG window."""
        from scipy.signal import welch
        window = self._bridge.get_window()  # (4, 400) at 200 Hz
        window = np.clip(window - window.mean(axis=1, keepdims=True), -100.0, 100.0)
        freqs, psd = welch(window, fs=200, axis=1, nperseg=min(256, window.shape[1]))
        psd_mean = psd.mean(axis=0)
        alpha = psd_mean[(freqs >= 8)  & (freqs <= 13)].mean()
        beta  = psd_mean[(freqs >= 13) & (freqs <= 30)].mean()
        theta = psd_mean[(freqs >= 4)  & (freqs <= 8)].mean()
        total = alpha + beta + theta + 1e-9
        mood   = float(np.clip(alpha / total * 3 - 1, -1, 1))
        energy = float(np.clip(beta  / total * 3 - 1, -1, 1))
        return np.array([mood, energy], dtype=np.float32)

    def _mock_read(self) -> np.ndarray:
        t = time.time() - self._t0
        mood   = 0.6 * np.sin(2 * np.pi * t / 60.0) + 0.1 * np.random.randn()
        energy = 0.5 * np.cos(2 * np.pi * t / 90.0) + 0.1 * np.random.randn()
        return np.array([np.clip(mood, -1, 1), np.clip(energy, -1, 1)], dtype=np.float32)

    def smooth(self) -> np.ndarray:
        self._history.append(self.read())
        n       = len(self._history)
        stack   = np.stack(list(self._history))         # (n, coord_dim)
        weights = np.array([0.5 ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        return (weights[:, None] * stack).sum(axis=0).astype(np.float32)

    @property
    def coord_dim(self) -> int:
        return 2


class REVEEEGSource:
    """
    EEG source that encodes raw signals through REVE-base (69.2M) and returns
    a mean-pooled flat embedding as the state coordinate.

    The coord_dim is determined on the first read() call (lazy REVE load).
    Pass a MuseBridge instance for real hardware; pass mock=True for smoke testing.
    """

    def __init__(
        self,
        mock: bool = False,
        smooth_window: int = 3,
        bridge: "MuseBridge | None" = None,
    ):
        from reve import REVEEncoder, DEFAULT_ELECTRODES, MUSE_CHANNELS, SAMPLE_RATE
        self._bridge     = bridge
        self._mock       = mock
        self._electrodes = MUSE_CHANNELS if bridge else DEFAULT_ELECTRODES
        self._encoder    = REVEEncoder(electrode_names=self._electrodes)
        self._sample_rate = SAMPLE_RATE
        self._history: deque = deque(maxlen=smooth_window)
        self._coord_dim: int | None = None

        if not mock and bridge is None:
            raise NotImplementedError(
                "Provide a MuseBridge (--bridge) or use --mock-eeg for smoke testing."
            )

    def _raw_eeg(self) -> np.ndarray:
        """Return (channels, time_points) float32 array at 200 Hz."""
        if self._mock:
            n_ch = len(self._electrodes)
            n_t  = int(2.0 * self._sample_rate)
            return np.random.randn(n_ch, n_t).astype(np.float32)
        return self._bridge.get_window()  # (4, 400) at 200 Hz

    def read(self) -> np.ndarray:
        """Encode one EEG segment and return mean-pooled flat vector."""
        eeg = self._raw_eeg()
        eeg = (eeg - eeg.mean(axis=1, keepdims=True)) / (eeg.std(axis=1, keepdims=True) + 1e-6)
        emb = self._encoder.encode(eeg)             # (1, tokens, hidden)
        flat = emb.squeeze(0).mean(0).cpu().numpy() # (hidden,)
        self._coord_dim = flat.shape[0]
        return flat.astype(np.float32)

    def smooth(self) -> np.ndarray:
        self._history.append(self.read())
        n       = len(self._history)
        stack   = np.stack(list(self._history))
        weights = np.array([0.5 ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        return (weights[:, None] * stack).sum(axis=0).astype(np.float32)

    @property
    def coord_dim(self) -> int:
        if self._coord_dim is None:
            # probe to determine output dim before the main loop
            self.read()
        return self._coord_dim


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
        self.pipe.vae.enable_tiling(tile_sample_min_width=512, tile_sample_min_height=512)
        self.pipe.enable_model_cpu_offload()
        print("Pipeline ready.")

    def generate(self, prompt_embeds: torch.Tensor) -> Image.Image:
        """
        Generate image from a Gemma embedding (no text involved).

        prompt_embeds: shape (1, SEQ_LEN, GEMMA_DIM)
        Returns: PIL Image
        """
        attn_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=prompt_embeds.device)
        return self.pipe(
            prompt_embeds=prompt_embeds.to(torch.bfloat16),
            prompt_attention_mask=attn_mask,
            num_inference_steps=2,   # one-step model; 1-2 steps is sufficient
            guidance_scale=4.5,
        ).images[0]


# ---------------------------------------------------------------------------
# RL Models
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """
    Stochastic actor for SAC.
    Input:  obs = concat(coord, goal) — coord_dim * 2 floats
    Output: tanh-squashed action in (-1, 1)^DELTA_DIM
    """
    def __init__(self, coord_dim: int = COORD_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(coord_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, 64),            nn.ReLU(),
        )
        self.mean_head    = nn.Linear(64, DELTA_DIM)
        self.log_std_head = nn.Linear(64, DELTA_DIM)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action + log_prob with tanh correction. Returns (action, log_prob)."""
        mean, log_std = self(obs)
        std    = log_std.exp()
        u      = torch.distributions.Normal(mean, std).rsample()  # reparameterized
        action = torch.tanh(u)
        log_prob = (
            torch.distributions.Normal(mean, std).log_prob(u)
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)
        return action, log_prob

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean)


class Critic(nn.Module):
    """
    Q-function: Q(obs, action) → scalar.
    obs = concat(coord, goal) — coord_dim * 2 floats
    """
    def __init__(self, coord_dim: int = COORD_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim * 2 + DELTA_DIM, 64), nn.ReLU(),
            nn.Linear(64, 64),                         nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


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
    Predicts next coord given current coord and delta action.
    Input:  (coord, delta)  — coord_dim + DELTA_DIM
    Output: next_coord      — coord_dim
    """
    def __init__(self, coord_dim: int = COORD_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim + DELTA_DIM, 64), nn.ReLU(),
            nn.Linear(64, 64),                    nn.ReLU(),
            nn.Linear(64, coord_dim),
        )

    def forward(self, coord: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([coord, delta], dim=-1))


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Sliding window of (coord, delta, reward, next_coord, goal) transitions."""
    def __init__(self, maxlen: int = REPLAY_MAXLEN):
        self.buf = deque(maxlen=maxlen)

    def push(
        self,
        coord: np.ndarray,
        delta: np.ndarray,
        reward: float,
        next_coord: np.ndarray,
        goal: np.ndarray,
    ):
        self.buf.append((coord.copy(), delta.copy(), float(reward), next_coord.copy(), goal.copy()))

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx   = np.random.choice(len(self.buf), size=n, replace=False)
        items = [self.buf[i] for i in idx]
        coords      = torch.tensor([i[0] for i in items], dtype=torch.float32, device=DEVICE)
        deltas      = torch.tensor([i[1] for i in items], dtype=torch.float32, device=DEVICE)
        rewards     = torch.tensor([i[2] for i in items], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_coords = torch.tensor([i[3] for i in items], dtype=torch.float32, device=DEVICE)
        goals       = torch.tensor([i[4] for i in items], dtype=torch.float32, device=DEVICE)
        return coords, deltas, rewards, next_coords, goals

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_image(image: Image.Image, fullscreen: bool = False):
    """
    Show image to the human subject.
    fullscreen=True opens a fullscreen pygame window (requires: pip install pygame).
    fullscreen=False uses the OS image viewer — fine for a first smoke test.
    """
    if not fullscreen:
        image.show()
        return

    import pygame  # noqa: deferred import — only needed in fullscreen mode

    if not pygame.get_init():
        pygame.init()
        pygame.display.set_caption("EEG Neurofeedback")

    info   = pygame.display.Info()
    screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
    surface = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
    surface = pygame.transform.scale(surface, (info.current_w, info.current_h))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    # pygame stays open; the STEP_SECONDS sleep in the main loop acts as display time.


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
    coords, deltas, _, next_coords, _ = replay.sample(BATCH_SIZE)
    pred = wm(coords, deltas)
    loss = nn.functional.mse_loss(pred, next_coords)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def update_sac(
    actor: Actor,
    q1: Critic,
    q2: Critic,
    q1_tgt: Critic,
    q2_tgt: Critic,
    log_alpha: torch.Tensor,
    actor_opt: torch.optim.Optimizer,
    q_opt: torch.optim.Optimizer,
    alpha_opt: torch.optim.Optimizer,
    obs: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    next_obs: torch.Tensor,
) -> None:
    alpha = log_alpha.exp().detach()

    # Critic update — Bellman target with entropy bonus
    with torch.no_grad():
        next_act, next_lp = actor.sample(next_obs)
        q_next   = torch.min(q1_tgt(next_obs, next_act), q2_tgt(next_obs, next_act))
        q_target = reward + SAC_GAMMA * (q_next - alpha * next_lp)

    q_loss = (
        nn.functional.mse_loss(q1(obs, action), q_target)
        + nn.functional.mse_loss(q2(obs, action), q_target)
    )
    q_opt.zero_grad()
    q_loss.backward()
    q_opt.step()

    # Actor update — maximise Q minus entropy cost
    new_act, log_prob = actor.sample(obs)
    q_val      = torch.min(q1(obs, new_act), q2(obs, new_act))
    actor_loss = (alpha * log_prob - q_val).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    # Alpha update — keep entropy near target
    alpha_loss = -(log_alpha * (log_prob.detach() + SAC_TARGET_ENTROPY)).mean()
    alpha_opt.zero_grad()
    alpha_loss.backward()
    alpha_opt.step()

    # Soft-update frozen target critics
    for p, pt in zip(q1.parameters(), q1_tgt.parameters()):
        pt.data.mul_(1 - SAC_TAU).add_(SAC_TAU * p.data)
    for p, pt in zip(q2.parameters(), q2_tgt.parameters()):
        pt.data.mul_(1 - SAC_TAU).add_(SAC_TAU * p.data)

    return actor_loss.item(), q_loss.item()


def dyna_sac_update(
    actor: Actor,
    q1: Critic,
    q2: Critic,
    q1_tgt: Critic,
    q2_tgt: Critic,
    log_alpha: torch.Tensor,
    wm: WorldModel,
    actor_opt: torch.optim.Optimizer,
    q_opt: torch.optim.Optimizer,
    alpha_opt: torch.optim.Optimizer,
    replay: ReplayBuffer,
    n_steps: int = 5,
) -> None:
    """
    Dyna-style: generate n_steps imagined transitions via the world model and
    run SAC updates on each. More data-efficient than real steps alone — the
    critics learn faster, which lowers actor gradient variance.
    """
    if len(replay) < BATCH_SIZE or n_steps == 0:
        return

    coords, _, _, _, goals = replay.sample(BATCH_SIZE)
    coord  = coords[torch.randint(len(coords), (1,))].detach()
    goal_t = goals[torch.randint(len(goals), (1,))].detach()

    for _ in range(n_steps):
        obs = torch.cat([coord, goal_t], dim=-1)
        with torch.no_grad():
            action, _ = actor.sample(obs)
            next_coord = wm(coord, action).clamp(-1.0, 1.0)
        next_obs = torch.cat([next_coord, goal_t], dim=-1)
        reward   = -torch.norm(next_coord - goal_t, dim=-1, keepdim=True)

        update_sac(
            actor, q1, q2, q1_tgt, q2_tgt, log_alpha,
            actor_opt, q_opt, alpha_opt,
            obs, action, reward, next_obs,
        )
        coord = next_coord.detach()


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402 — grouped with session helpers


def save_session(
    path: str,
    actor: Actor,
    projector: DeltaProjector,
    wm: WorldModel,
    q1: Critic,
    q2: Critic,
    q1_tgt: Critic,
    q2_tgt: Critic,
    log_alpha: torch.Tensor,
    step: int,
):
    torch.save({
        "actor":     actor.state_dict(),
        "projector": projector.state_dict(),
        "wm":        wm.state_dict(),
        "q1":        q1.state_dict(),
        "q2":        q2.state_dict(),
        "q1_tgt":    q1_tgt.state_dict(),
        "q2_tgt":    q2_tgt.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
        "step":      step,
    }, path)
    print(f"  [saved → {path}  step={step}]")


def load_session(
    path: str,
    actor: Actor,
    projector: DeltaProjector,
    wm: WorldModel,
    q1: Critic,
    q2: Critic,
    q1_tgt: Critic,
    q2_tgt: Critic,
    log_alpha: torch.Tensor,
) -> int:
    ckpt = torch.load(path, map_location=DEVICE)
    actor.load_state_dict(ckpt["actor"])
    projector.load_state_dict(ckpt["projector"])
    wm.load_state_dict(ckpt["wm"])
    q1.load_state_dict(ckpt["q1"])
    q2.load_state_dict(ckpt["q2"])
    q1_tgt.load_state_dict(ckpt["q1_tgt"])
    q2_tgt.load_state_dict(ckpt["q2_tgt"])
    log_alpha.data.copy_(ckpt["log_alpha"].to(DEVICE))
    step = ckpt.get("step", 0)
    print(f"  [loaded {path}  resuming at step {step}]")
    return step


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    explore_steps: int = 20,
    mock_eeg: bool = False,
    use_reve: bool = False,
    save_path: str | None = "session.pt",
    save_interval: int = 10,
    dyna: bool = True,
    fullscreen: bool = False,
    bridge: "MuseBridge | None" = None,
    inference_only: bool = False,
):
    """
    explore_steps:  number of random-action steps before SAC policy takes over.
    mock_eeg:       use synthetic sine-wave EEG instead of real hardware.
    use_reve:       encode raw EEG through REVE-base instead of (mood, energy) coords.
    save_path:      checkpoint file for session persistence; None disables saving.
    save_interval:  save checkpoint every N steps.
    dyna:           whether to use world-model dreaming between real steps.
    fullscreen:     display images fullscreen via pygame instead of image.show().
    inference_only: freeze all weights — skip replay, world-model, and SAC updates.
    """
    print(f"Device: {DEVICE}")
    if mock_eeg:
        print("Using mock EEG (sine wave). Pass mock_eeg=False for real hardware.")

    # Init components
    if use_reve:
        print("EEG encoder: REVE-base (69.2M)")
        eeg = REVEEEGSource(mock=mock_eeg, bridge=bridge)
    else:
        eeg = EEGSource(mock=mock_eeg, bridge=bridge)

    coord_dim = eeg.coord_dim
    if use_reve:
        print(f"REVE coord dim: {coord_dim}")

    generator = ImageGenerator()
    base_emb  = load_base_embedding()  # (1, SEQ_LEN, GEMMA_DIM), frozen, pre-computed offline

    actor     = Actor(coord_dim=coord_dim).to(DEVICE)
    projector = DeltaProjector().to(DEVICE)
    wm        = WorldModel(coord_dim=coord_dim).to(DEVICE)
    q1        = Critic(coord_dim=coord_dim).to(DEVICE)
    q2        = Critic(coord_dim=coord_dim).to(DEVICE)
    q1_tgt    = Critic(coord_dim=coord_dim).to(DEVICE)
    q2_tgt    = Critic(coord_dim=coord_dim).to(DEVICE)
    q1_tgt.load_state_dict(q1.state_dict())
    q2_tgt.load_state_dict(q2.state_dict())
    for p in (*q1_tgt.parameters(), *q2_tgt.parameters()):
        p.requires_grad_(False)
    log_alpha = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    replay    = ReplayBuffer()

    actor_opt = torch.optim.Adam(list(actor.parameters()) + list(projector.parameters()), lr=LR)
    q_opt     = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=LR)
    alpha_opt = torch.optim.Adam([log_alpha], lr=LR)
    wm_opt    = torch.optim.Adam(wm.parameters(), lr=LR)

    # Load prior session if checkpoint exists
    step = 0
    if save_path and Path(save_path).exists():
        step = load_session(save_path, actor, projector, wm, q1, q2, q1_tgt, q2_tgt, log_alpha)

    # CSV metrics log — one row per step, appended so resumed sessions accumulate
    log_path = save_path.replace(".pt", "_log.csv") if save_path else "session_log.csv"
    _log_file = open(log_path, "a", newline="")
    _log_writer = csv.writer(_log_file)
    if _log_file.tell() == 0:
        _log_writer.writerow(["step", "timestamp", "coord", "goal", "reward", "wm_loss", "actor_loss", "q_loss"])

    if inference_only:
        actor.eval()
        projector.eval()
        print("Inference-only mode: weights frozen, no replay or gradient updates.")

    # Sample initial goal near the user's current EEG state
    first_coord = eeg.read()
    goal_radius = GOAL_RADIUS_INIT
    goal        = sample_goal_near(first_coord, goal_radius)
    goal_timer  = time.time()

    # Graceful Ctrl-C: save checkpoint and quit pygame before exit
    def _handle_sigint(sig, frame):
        print("\nSession ended by user.")
        if save_path and not inference_only:
            save_session(save_path, actor, projector, wm, q1, q2, q1_tgt, q2_tgt, log_alpha, step)
        _log_file.flush()
        _log_file.close()
        print(f"  [log → {log_path}]")
        if fullscreen:
            import pygame  # noqa
            pygame.quit()
        raise SystemExit(0)
    signal.signal(signal.SIGINT, _handle_sigint)

    print("Starting neurofeedback loop. Ctrl-C to stop.")
    print(f"  First goal (norm={np.linalg.norm(goal):.2f})  radius={goal_radius:.2f}")

    _prev_coord = None

    while True:
        # Block here while browser has pressed Stop
        if bridge:
            bridge.wait_if_stopped()

        # --- 1. Read current EEG state ---
        coord = eeg.read()  # (coord_dim,)
        if _prev_coord is not None and np.allclose(coord, _prev_coord, atol=1e-4):
            print("  WARNING: EEG coord identical to previous step — signal may be stale/disconnected")

        # --- 2. Refresh goal if interval elapsed ---
        if time.time() - goal_timer > GOAL_INTERVAL:
            goal_radius = min(GOAL_RADIUS_MAX, goal_radius + GOAL_RADIUS_GROW)
            goal        = sample_goal_near(coord, goal_radius)
            goal_timer  = time.time()
            print(f"\n  New goal (norm={np.linalg.norm(goal):.2f})  radius={goal_radius:.2f}")

        # --- 3. Choose action (delta) ---
        obs = torch.tensor(
            np.concatenate([coord, goal]), dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)  # (1, coord_dim * 2)

        if not inference_only and step < explore_steps:
            # Random exploration: scale down to 0.3 so images stay visually coherent
            delta_t = (torch.rand(1, DELTA_DIM, device=DEVICE) * 2 - 1) * 0.3
            print(f"  [explore {step+1}/{explore_steps}]", end="")
        else:
            with torch.no_grad():
                delta_t, _ = actor.sample(obs)  # SAC explores via entropy, no manual noise needed

        # Clip delta magnitude so images don't jump wildly
        norm = delta_t.norm()
        if norm > DELTA_MAX_NORM:
            delta_t = delta_t * DELTA_MAX_NORM / norm

        # --- 4. Project delta to Gemma space, build final embedding ---
        with torch.no_grad():
            gemma_delta = projector(delta_t)               # (1, 1, GEMMA_DIM)
        final_emb = base_emb + gemma_delta                 # (1, SEQ_LEN, GEMMA_DIM)

        # --- 5. Generate and display image ---
        image = generator.generate(final_emb.detach())
        if bridge:
            bridge.send_image(image)
        else:
            display_image(image, fullscreen=fullscreen)

        # --- 6. Wait, then read new EEG state ---
        time.sleep(STEP_SECONDS)
        next_coord = eeg.read()  # (coord_dim,)

        # --- 7. Compute reward: negative distance to goal ---
        reward = -float(np.linalg.norm(next_coord - goal))
        print(f"  step={step}  |coord|={np.linalg.norm(coord):.2f}"
              f"  |next|={np.linalg.norm(next_coord):.2f}"
              f"  reward={reward:.3f}"
              f"  α={log_alpha.exp().item():.3f}")

        if bridge:
            bridge.send_status(
                f"step={step}  reward={reward:.3f}  |coord|={np.linalg.norm(coord):.2f}"
            )

        wm_loss, actor_loss, q_loss = None, None, None
        if not inference_only:
            # --- 8. Store transition ---
            delta_np = delta_t.squeeze(0).cpu().detach().numpy()
            replay.push(coord, delta_np, reward, next_coord, goal)

            # --- 9. Update world model ---
            wm_loss = update_world_model(wm, wm_opt, replay)

            # --- 10. SAC policy update (once past exploration and buffer is ready) ---
            if step >= explore_steps and len(replay) >= BATCH_SIZE:
                coords_b, deltas_b, rewards_b, next_coords_b, goals_b = replay.sample(BATCH_SIZE)
                obs_b      = torch.cat([coords_b, goals_b],      dim=-1)
                next_obs_b = torch.cat([next_coords_b, goals_b], dim=-1)
                actor_loss, q_loss = update_sac(
                    actor, q1, q2, q1_tgt, q2_tgt, log_alpha,
                    actor_opt, q_opt, alpha_opt,
                    obs_b, deltas_b, rewards_b, next_obs_b,
                )
                if dyna:
                    dyna_sac_update(
                        actor, q1, q2, q1_tgt, q2_tgt, log_alpha,
                        wm, actor_opt, q_opt, alpha_opt, replay,
                    )

        if bridge:
            bridge.send_metrics(step, coord, reward, wm_loss, actor_loss, q_loss)

        # --- 11. Log metrics row ---
        _log_writer.writerow([
            step,
            round(time.time(), 3),
            json.dumps(coord.tolist()),
            json.dumps(goal.tolist()),
            round(reward, 6),
            round(wm_loss, 6) if wm_loss is not None else "",
            round(actor_loss, 6) if actor_loss is not None else "",
            round(q_loss, 6) if q_loss is not None else "",
        ])
        _log_file.flush()
        _prev_coord = next_coord.copy()

        # --- 12. Periodic checkpoint ---
        if not inference_only and save_path and step % save_interval == 0:
            save_session(save_path, actor, projector, wm, q1, q2, q1_tgt, q2_tgt, log_alpha, step)

        step += 1




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EEG Neurofeedback RL with SANA-Sprint")
    p.add_argument("--explore-steps", type=int, default=20,
                   help="Random exploration steps before SAC policy takes over (default: 20).")
    p.add_argument("--mock-eeg", action="store_true", default=False,
                   help="Use synthetic sine-wave EEG instead of real hardware.")
    p.add_argument("--reve", action="store_true", default=False,
                   help="Encode raw EEG through REVE-base (69.2M) instead of (mood, energy) coords.")
    p.add_argument("--save-path", type=str, default="session.pt",
                   help="Checkpoint file for session persistence (default: session.pt).")
    p.add_argument("--no-save", action="store_true", default=False,
                   help="Disable session persistence entirely.")
    p.add_argument("--save-interval", type=int, default=10,
                   help="Save checkpoint every N steps (default: 10).")
    p.add_argument("--no-dyna", action="store_true", default=False,
                   help="Disable world-model dreaming (SAC on real transitions only).")
    p.add_argument("--fullscreen", action="store_true", default=False,
                   help="Display images fullscreen via pygame (requires: pip install pygame).")
    p.add_argument("--bridge", action="store_true", default=False,
                   help="Use browser bridge: serves bridge.html + receives Muse 2 EEG via WebSocket.")
    p.add_argument("--ws-port", type=int, default=8765, help="WebSocket port (default: 8765).")
    p.add_argument("--inference-only", action="store_true", default=False,
                   help="Freeze weights: skip replay, world-model, and SAC updates.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    bridge = MuseBridge(ws_port=args.ws_port) if args.bridge else None
    run(
        explore_steps=args.explore_steps,
        mock_eeg=args.mock_eeg,
        use_reve=args.reve,
        save_path=None if args.no_save else args.save_path,
        save_interval=args.save_interval,
        dyna=not args.no_dyna,
        fullscreen=args.fullscreen,
        bridge=bridge,
        inference_only=args.inference_only,
    )
