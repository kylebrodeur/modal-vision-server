# Quickstart: modal-vision-server

Vision classification on Modal. You choose the model, the segmenter, and the fast
gate through one model card. The BioCLIP plant stack ships as the example card.
Current release: v1.2.0.

Start with the [toolkit quickstart](https://github.com/kylebrodeur/modal-toolkit/blob/main/docs/quickstart.md) if `mtk` is not
installed yet.

## 1. Create a project

```bash
mtk init vision-lab
cd vision-lab
```

## 2. Pull the package

```bash
mtk pull vision --tag <current-release>
```

The package is vendored into `vendor/vision-server/`: a git checkout pinned to
the exact release tag, read-only and drift-guarded.

## 3. Write the overlay

`deploys/vision-server/deploy.json` names the app `<project>-vision`:

```json
{
  "pkg": "vision",
  "app_name": "vision-lab-vision",
  "upstream": { "repo": "modal-vision-server", "tag": "<current-release>" }
}
```

`deploys/vision-server/deploy.sh` deploys the VENDORED copy under
`vendor/vision-server/`: never a live checkout. `mtk deploy vision` runs it
after the preflight, drift, and secrets gates.

## 4. Deploy

```bash
mtk deploy vision
```

One secret is declared: `modal-vision-secret` (`API_TOKEN`, generated as hex32).
If your model card needs a HuggingFace token, declare it in your overlay, not in
the public repo.

## 5. Verify

```bash
mtk doctor --pkg vision
curl -s https://<your-workspace>--<app>-serve.modal.run/health
```

`/health` reports the model, the segmenter, and the fast gate.

## Next

- Classify, segment, and tune the model card: [how-to](./how-to.md)

