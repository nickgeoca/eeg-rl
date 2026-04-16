# AI Image Generation — Model Assessments

Scratchpad for evaluating generative image models. Focus: **under 4GB** (roughly ≤2B params in FP16, or larger models quantized).

Source: https://carlofkl.github.io/dreamlite/

---

## Under-4GB Shortlist

| Model | Params | GenEval ↑ | DPG ↑ | Available | Notes |
|---|---|---|---|---|---|
| DreamLite | 0.39B | 0.72 | 85.8 | **Not yet released** | Best sub-1B on paper; also has editing scores |
| SANA-Sprint 0.6B | 0.6B | 0.72 | — | HuggingFace | Best released sub-1B; 1-step inference, fast on laptop GPU |
| SANA-Sprint 1.6B | 1.6B | 0.74 | — | HuggingFace | Beats FLUX.1-schnell (12B) on GenEval + FID at 1-step |
| PixArt-Sigma XL/2 | 0.6B | ~0.66 | — | HuggingFace | Mature ecosystem, good for LoRA/fine-tuning |
| SANA-1.6B | 1.6B | 0.67 | 84.8 | HuggingFace | Original (non-sprint) teacher model |
| SnapGen++ (small) | 0.4B | 0.66 | 85.2 | Paper only | No public weights as of early 2026 |
| SD 3.5 Medium | 2.5B | ~0.73 | — | HuggingFace | Slightly over 2B but quantizes to ~2.8GB; best aesthetics in tier |

**Best available backup for DreamLite: SANA-Sprint 1.6B or 0.6B** — both on HuggingFace, strong benchmark numbers, fast inference.

---

## Personal Evaluations

<!-- Add notes after running models locally -->

### SANA-Sprint 1.6B
`Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers`

- [ ] Ran locally
- Notes:

### SANA-Sprint 0.6B
`Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`

- [ ] Ran locally
- Notes:

### DreamLite
`carlofkl/dreamlite` *(not yet released)*

- [ ] Ran locally
- Notes:

---

## Links & Resources

- DreamLite paper/project: https://carlofkl.github.io/dreamlite/
- SANA-Sprint project page: https://nvlabs.github.io/Sana/Sprint/
- SANA HuggingFace collection: https://huggingface.co/collections/Efficient-Large-Model/sana-673efba2a57ed99843f11f9e
- PixArt-Sigma: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS
- SD 3.5 Medium: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
