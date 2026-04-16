"""
reve.py — REVE EEG Foundation Model encoder

Processes EEG data through the REVE foundation model and outputs embeddings.

Input:  np.ndarray or torch.Tensor of shape (channels, time_points)
        Must be sampled at 200 Hz.
Output: torch.Tensor of shape (1, num_tokens, hidden_dim)

Reference: https://arxiv.org/abs/2510.21585
"""

import argparse

import numpy as np
import torch
from transformers import AutoModel

# Standard 10-20 electrode names (19 channels)
DEFAULT_ELECTRODES = [
    "Fp1", "Fp2",
    "F7",  "F3", "Fz", "F4", "F8",
    "T7",  "C3", "Cz", "C4", "T8",
    "P7",  "P3", "Pz", "P4", "P8",
    "O1",  "O2",
]

SAMPLE_RATE = 200  # REVE requires input at exactly 200 Hz


def get_eeg() -> np.ndarray:
    """
    Return a (channels, time_points) EEG array at 200 Hz.

    STUB: replace with real hardware read, e.g.:
        import elata
        device = elata.connect()
        return device.read_segment(duration_s=2.0)
    """
    n_ch = len(DEFAULT_ELECTRODES)
    n_t  = int(2.0 * SAMPLE_RATE)  # 2-second window
    return np.random.randn(n_ch, n_t).astype(np.float32)


class REVEEncoder:
    """
    Wraps the REVE EEG foundation model for embedding extraction.
    Models are loaded lazily on first call to encode().

    Example:
        encoder = REVEEncoder()
        eeg = get_eeg()           # (channels, time_points)
        emb = encoder.encode(eeg) # (1, num_tokens, hidden_dim)
    """

    def __init__(
        self,
        electrode_names: list[str] = DEFAULT_ELECTRODES,
        device: torch.device | None = None,
    ):
        self.electrode_names = electrode_names
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model    = None
        self._pos_bank = None

    def _load(self) -> None:
        if self._model is not None:
            return
        print("Loading REVE position bank (brain-bzh/reve-positions)…")
        self._pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions", trust_remote_code=True
        ).to(self.device)
        print("Loading REVE base model (brain-bzh/reve-base)…")
        self._model = AutoModel.from_pretrained(
            "brain-bzh/reve-base", trust_remote_code=True
        ).to(self.device)
        self._model.eval()

    def encode(self, eeg: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Encode a single EEG segment.

        Args:
            eeg: shape (channels, time_points) at 200 Hz.

        Returns:
            Embedding tensor of shape (1, num_tokens, hidden_dim).
        """
        if isinstance(eeg, np.ndarray):
            eeg = torch.tensor(eeg, dtype=torch.float32)
        return self._encode_batch(eeg.unsqueeze(0))

    def _encode_batch(self, eeg: torch.Tensor) -> torch.Tensor:
        self._load()
        eeg = eeg.to(self.device)
        with torch.no_grad():
            positions = self._pos_bank(self.electrode_names)       # (channels, 3)
            positions = positions.expand(eeg.size(0), -1, -1)      # (batch, channels, 3)
            return self._model(eeg, positions)


# ---------------------------------------------------------------------------
# CLI — smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REVE EEG encoder — extract embeddings")
    parser.add_argument("--seconds", type=float, default=2.0,
                        help="Duration of mock EEG segment in seconds (default: 2.0)")
    args = parser.parse_args()

    # Override stub window size from CLI
    n_ch = len(DEFAULT_ELECTRODES)
    n_t  = int(args.seconds * SAMPLE_RATE)
    print(f"Mock EEG: {n_ch} channels × {n_t} time-points  ({args.seconds}s @ {SAMPLE_RATE} Hz)")

    eeg = np.random.randn(n_ch, n_t).astype(np.float32)

    encoder = REVEEncoder()
    emb = encoder.encode(eeg)

    print(f"Embedding shape : {tuple(emb.shape)}")
    print(f"Embedding dtype : {emb.dtype}")
    print(f"Embedding norm  : {emb.norm().item():.4f}")
    print(f"Embedding device: {emb.device}")
