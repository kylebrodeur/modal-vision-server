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
mtk pull vision
```

## 3. Write the overlay

`deploys/vision/deploy.json` names the app `<project>-vision`:

```json
{ "pkg": "vision", "app_name": "vision-lab-vision" }
```

`deploys/vision/deploy.sh` runs `modal deploy server/app.py` from the pulled
checkout by path. Keep it in your project.

## 4. Deploy

```bash
mtk deploy vision
```

One secret is declared: `modal-vision-secret` (`API_TOKEN`, generated as hex32).
If your model card needs a HuggingFace token, declare it in your overlay, not in
the public repo.

`mtk vision deploy` runs a deploy inside the package checkout with no overlay. Do
not use it. Use `mtk deploy vision`.

## 5. Verify

```bash
mtk doctor --pkg vision
curl -s https://<your-workspace>--<app>-serve.modal.run/health
```

`/health` reports the model, the segmenter, and the fast gate.

## Next

- Classify, segment, and tune the model card: [how-to](./how-to.md)
