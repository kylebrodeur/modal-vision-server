# Contributing

Thanks for your interest in contributing. This project is extracted from production work and kept intentionally lean, so contributions should follow the same spirit.

## Ground Rules

- **Keep it boring.** Prefer straightforward code over clever abstractions. This is a utility, not a framework.
- **No new heavyweight deps** unless absolutely required for the model pipeline. The local test env must stay fast to `uv sync`.
- **Config, not code.** New vision models, thresholds, and label sets belong in model cards under [`registry/`](registry/) (consumed by `server/model_card.py`), not sprinkled through the endpoint logic. `server/config.py` only carries the env-resolved service constants (app identity, secrets, rate limits).
- **Env prefixes** follow `MODAL_VISION_*`. Never reintroduce project-specific branding.
- **No AI slop.** Comments and docs should describe *why* the code exists, not restate what it does.

## Workflow

1. Fork and create a feature branch.
2. Use `uv sync --group dev` inside `server/` to set up a local dev environment.
3. Ensure `uv run pytest tests -q` passes for any code change (run from `server/`).
4. Run `uv run ruff check .` from `server/` before submitting.
5. Submit a pull request against `main` with a clear description of what and why.

The project is licensed under Apache 2.0. By contributing you agree that your contributions will be licensed under the same terms.