# Modal Vision Server

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-pink.svg)](https://github.com/sponsors/kylebrodeur)

High-performance GPU-accelerated computer vision system deployed on Modal. This server provides specialized capabilities for biological taxon classification and precision specimen segmentation.

## The Vision Pillars

### Classification
Driven by the vision model (ViT-L/14, TreeOfLife-200M).
- **Species Precision:** State-of-the-art taxonomic classification with emergent intra-species variation separation.
- **Few-Shot Matching:** Calibrated reference-matching that allows for high-precision discrimination within confusable morphological clusters without steamrolling decisive text evidence.

### Segmentation
Powered by `facebook/sam2.1-hiera-tiny` (SAM 2.1).
- **Adaptive Masking:** A fast-path classification determines if a specimen is isolated enough to skip segmentation. SAM is invoked only as a tie-breaker for ambiguous cases.
- **Specimen Isolation:** Hard sanity gates prevent background texture (e.g., flooring) from being classified as the primary specimen.

### Sync
Optimized for cold-start latency and resource persistence.
- **Weight Persistence:** Uses `modal.Volume` to store model weights, eliminating redundant Hugging Face downloads.
- **Kernel Warming:** Performs a dummy encode pass during initialization to pre-compile CUDA kernels, ensuring the first request hits peak performance.

## Deployment

### Deploy to Modal
To deploy this server to your Modal account:

```bash
modal deploy server/app.py
```

### Environment Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MODAL_VISION_GPU` | `T4` | GPU hardware target (T4, A10G, etc.) |
| `MODAL_VISION_MODEL` | `hf-hub:imageomics/bioclip-2` | Vision model identifier |
| `MODAL_VISION_ADAPTIVE_SEGMENT` | `false` | Enable adaptive SAM segmentation |
| `MODAL_VISION_REFERENCE_SCALE` | `60` | Cosine scale for reference matching |
| `MODAL_VISION_RATE_LIMIT_MAX` | `30` | Maximum requests per window |

## Part of the Modal Ecosystem

This repo is one of three standalone Modal utilities from the same author. Each is extractable and deployable on its own.

- **[modal-embedding-server](https://github.com/kylebrodeur/modal-embedding-server):** GPU-backed embeddings with a monotonic sync protocol for local-first search.
- **[modal-inference-server](https://github.com/kylebrodeur/modal-inference-server):** OpenAI-compatible LLM inference with hot-set routing and scale-to-zero.

## Examples

See [`examples/`](examples/) for a minimal, stdlib-only client (`classify_example.py`) you can copy directly into your own stack.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for ground rules and workflow.

## License

Apache-2.0 — see [LICENSE](LICENSE).
