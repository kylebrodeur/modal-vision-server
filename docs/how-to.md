# How to operate modal-vision-server

Current release: v1.2.0. Start with the [quickstart](./quickstart.md) if you
have not deployed yet.

## Choose the model card

The model, the segmenter, and the fast gate are set by one model card. The
BioCLIP plant stack is the shipped example. Changing the card is a repo-side
change, followed by a redeploy through your overlay.

Segmenter options: SAM 2.1, or none. Fast gate options: self, cheap CLIP,
a deterministic script, or an external endpoint.

## Warm

`mtk vision warm` probes the lane. A cold boot starts on the first request,
so warming only matters when you need a response without a cold start.

## Health

`/health` is the liveness and configuration check. Retry once on a cold start.

## Scale to zero

The app idles to zero containers. The first request after idle pays the cold
start.

## Related

- Quickstart: [quickstart.md](./quickstart.md)
