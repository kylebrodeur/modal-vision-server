---
name: vision-server-upgrade
description: Upgrade + deploy-boundary discipline for the modal-vision-server checkout: tag upgrades, overlay-only deploys, model-card and weights-volume cautions. Use when upgrading/redeploying this repo or before modal commands inside this checkout.
license: Apache-2.0
metadata:
  author: kylebrodeur
  family: modal-toolkit
  repo: modal-vision-server
---

# vision-server: upgrade + deploy boundary (lane agents)

See AGENTS.md for the full rules; this is the working version:

```bash
# 0. provenance preflight (system workspace):
tools/guards/deploy-provenance.sh <overlay-dir>

# 1. upgrade by tag + verify
git fetch --tags && git checkout <tag>
uv run --project server pytest server/tests -q
```

Vision-specific cautions:

- The weights Volume (`modal-vision-weights`, shared name in-repo) holds
  the BioCLIP/SAM weights - a plain-repo deploy WOULD attach that same
  Volume. Overlays compose their own volume names via deploy.json env;
  never let a lane attach the shared volume.
- Model behavior comes from the model card (`registry/*.json` + env
  overrides). When a lane needs a different model, that is a CARD in
  the overlay (registry file + env), not a source edit.
- Hooks: `request.pre/post` + `identify.post` in `server/app.py`;
  register the boot-provenance observer from
  `tools/guards/boot-provenance.py` for auditability.
- Adaptive-segmentation + fast-gate thresholds are card fields; do not
  retune them in source to "fix" a lane's observations.

## Operating with mtk

Source of truth: the [quickstart](../../docs/quickstart.md) and
[how-to](../../docs/how-to.md) in this repo, and the toolkit's `mtk` skill.

```bash
mtk init <project-id>      # from the project directory; checks prerequisites and the account
mtk pull vision             # clone this repo into the repos root
mtk deploy vision           # preflight the overlay, confirm secrets, run deploys/vision/deploy.sh
mtk doctor --pkg vision     # health and drift
```

- Overlay: `deploys/vision/deploy.json` must name the app `<project>-vision`.
- Secrets this package declares: modal-vision-secret (API_TOKEN, generated). `mtk deploy` asks before creating any.
- Never run `modal deploy` inside this checkout.
