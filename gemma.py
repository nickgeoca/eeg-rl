"""
Gemma 4 E4B conversational chat (GGUF)
Model: unsloth/gemma-4-E4B-it-GGUF / gemma-4-E4B-it-UD-Q6_K_XL.gguf

First run downloads ~4.5GB to ~/.cache/huggingface/. Subsequent runs are instant.
Requires llama-cpp-python built with CUDA: uv sync --extra cuda
"""

import re
from pathlib import Path
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODEL_REPO = "unsloth/gemma-4-E4B-it-GGUF"
MODEL_FILE = "gemma-4-E4B-it-UD-Q6_K_XL.gguf"

SYSTEM_PROMPT = "You are a helpful assistant."

# Gemma 4 recommended sampling params
SAMPLING = dict(
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    max_tokens=1024,
)


def _read_token() -> str | None:
    token_file = Path(__file__).parent / "token"
    return token_file.read_text().strip() if token_file.exists() else None


def load_model() -> Llama:
    print("Locating model (downloads on first run ~4.5GB)...")
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, token=_read_token())
    print(f"Loading from {path}")
    return Llama(
        model_path=path,
        n_gpu_layers=-1,  # offload all layers to GPU
        n_ctx=8192,
        verbose=False,
    )


def strip_thinking(text: str) -> str:
    """Remove <|channel>thought\\n...<channel|> blocks before saving to history."""
    return re.sub(r"<\|channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL).strip()


def chat(model: Llama) -> None:
    # Per docs: thoughts must not appear in multi-turn history,
    # so we keep two lists — full output for display, clean for history.
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\nGemma 4 E4B — type 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        history.append({"role": "user", "content": user_input})

        print("\nGemma: ", end="", flush=True)
        full_reply = ""
        for chunk in model.create_chat_completion(messages=history, stream=True, **SAMPLING):
            token = chunk["choices"][0]["delta"].get("content", "")
            print(token, end="", flush=True)
            full_reply += token
        print("\n")

        history.append({"role": "assistant", "content": strip_thinking(full_reply)})


if __name__ == "__main__":
    model = load_model()
    chat(model)
