"""
bake_base_embedding.py
======================
Run this once, offline, to encode BASE_PROMPT with SANA-Sprint's Gemma2 text encoder
and save the result to base_emb.pt.

After this, Gemma never needs to be loaded at runtime — eeg_rl_clip.py just loads
the saved tensor.

Usage:
    python bake_base_embedding.py
"""

import torch
from diffusers import SanaSprintPipeline

BASE_PROMPT   = "abstract generative art, neutral"
SEQ_LEN       = 77
OUTPUT_PATH   = "base_emb.pt"
MODEL_ID      = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

print(f"Loading pipeline (text encoder only needed)…")
pipe = SanaSprintPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.text_encoder = pipe.text_encoder.to(DEVICE)

print(f"Encoding: '{BASE_PROMPT}'")
with torch.no_grad():
    tokens = pipe.tokenizer(
        BASE_PROMPT,
        return_tensors="pt",
        padding="max_length",
        max_length=SEQ_LEN,
        truncation=True,
    ).to(DEVICE)
    base_emb = pipe.text_encoder(**tokens).last_hidden_state  # (1, SEQ_LEN, 2048)

print(f"Embedding shape: {base_emb.shape}  dtype: {base_emb.dtype}")
torch.save(base_emb.cpu(), OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
