"""
PixArt-Σ image generation script
Requirements: pip install diffusers transformers accelerate sentencepiece safetensors torch

Model downloads automatically from HuggingFace on first run (~2.3GB DiT + ~4.5GB T5).
Subsequent runs use cached weights.
"""

import torch
from diffusers import PixArtSigmaPipeline
from pathlib import Path

# --- Config ---
MODEL_ID = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"  # 1024px variant
OUTPUT_DIR = Path("./outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16  # use bfloat16 if you hit NaN issues

PROMPTS = [
    "A misty mountain valley at dawn, cinematic lighting, hyperrealistic photography",
    "A neon-lit Tokyo alley in the rain, reflections on wet pavement, 35mm film",
]

# --- Inference settings ---
NUM_STEPS = 20        # 20 is a good default; can go down to 14 for speed
GUIDANCE_SCALE = 4.5  # 4.5 works well for PixArt; don't go too high
HEIGHT = 1024
WIDTH = 1024
SEED = 42


def load_pipeline(offload_text_encoder: bool = True) -> PixArtSigmaPipeline:
    """
    Load PixArt-Σ pipeline.
    
    offload_text_encoder: moves T5 to CPU after encoding, freeing ~4.5GB VRAM.
    Highly recommended on 8GB cards.
    """
    print(f"Loading pipeline on {DEVICE} ({DTYPE})...")

    pipe = PixArtSigmaPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True,
    )

    if offload_text_encoder:
        # Encode on GPU, then offload T5 to CPU — keeps VRAM usage low
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(DEVICE)

    # Optional: speed up attention on Ampere+ GPUs (RTX 30xx/40xx/50xx)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers memory-efficient attention enabled.")
    except Exception:
        print("xformers not available, using default attention.")

    return pipe


def generate(pipe: PixArtSigmaPipeline, prompt: str, idx: int) -> Path:
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + idx)

    print(f"\nGenerating [{idx}]: {prompt[:60]}...")

    image = pipe(
        prompt=prompt,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"output_{idx:03d}.png"
    image.save(out_path)
    print(f"Saved → {out_path}")
    return out_path


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    pipe = load_pipeline(offload_text_encoder=True)

    for idx, prompt in enumerate(PROMPTS):
        generate(pipe, prompt, idx)

    print("\nDone.")


if __name__ == "__main__":
    main()
