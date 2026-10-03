"""
BioCLIP vs Pl@ntNet accuracy harness.

Sources N images per taxon from Wikimedia Commons (botanically labeled),
then scores each against both the deployed Modal BioCLIP endpoint and the
Pl@ntNet identify API. Reports top-1 accuracy per model per taxon.

Usage (uv):
    cd backends/modal-bioclip
    uv run python eval/harness.py --per-taxon 3 [--limit-taxa 4]

Requires (env):
    BIOCLIP_URL      deployed /v1/identify URL (default: kylebrodeur workspace)
    BIOCLIP_TOKEN    shared bearer token (1Password: PlantFluent Modal BioCLIP)
    PLANTNET_API_KEY Pl@ntNet v2 key (1Password: PlantNet API)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_TAXA = [
    "Alocasia zebrina", "Alocasia micholitziana", "Alocasia longiloba",
    "Alocasia sanderiana", "Alocasia baginda", "Alocasia reginula",
    "Monstera deliciosa", "Monstera adansonii", "Monstera dubia",
    "Philodendron hederaceum", "Philodendron billietiae", "Philodendron verrucosum",
    "Epipremnum aureum", "Scindapsus pictus", "Ficus elastica", "Ficus lyrata",
]

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
PHOTO_DIR = Path(__file__).parent / "photos"
RESULTS_PATH = Path(__file__).parent / "results.json"

UA = "PlantFluentBioClipEval/1.0 (development; kyle@brodeur.me)"


def http_json(url: str, params: dict) -> dict:
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{qs}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def http_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_commons_images(taxon: str, per_taxon: int) -> list[tuple[str, bytes]]:
    """Return [(file_title, jpeg_bytes)] for a taxon from Commons."""
    search = http_json(COMMONS_API, {
        "action": "query", "list": "search", "format": "json",
        "srsearch": taxon, "srnamespace": "6", "srlimit": per_taxon * 3,
    })
    titles = [r["title"] for r in search["query"]["search"] if ".jpg" in r["title"].lower()][: per_taxon * 2]
    if not titles:
        return []
    # Resolve direct thumb URLs (<= 1024px keeps payloads sane).
    meta = http_json(COMMONS_API, {
        "action": "query", "format": "json", "titles": "|".join(titles),
        "prop": "imageinfo", "iiprop": "url", "iiurlwidth": "1024",
    })
    out: list[tuple[str, bytes]] = []
    for page in meta["query"]["pages"].values():
        ii = page.get("imageinfo")
        if not ii:
            continue
        thumb = ii[0].get("thumburl") or ii[0].get("url")
        if not thumb:
            continue
        try:
            out.append((page["title"], http_bytes(thumb)))
        except Exception as exc:  # noqa: BLE001 - network best-effort
            print(f"    skip {page['title']}: {exc}", file=sys.stderr)
            if len(out) >= per_taxon:
                break
    return out[:per_taxon]


def classify_bioclip(img_bytes: bytes, url: str, token: str) -> list[tuple[str, float]]:
    body = json.dumps({
        "images": ["data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()],
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        "origin": "https://app.plantfluent.com",
    })
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read())
    return [(p["name"], p["score"]) for p in data.get("predictions", [])]


def classify_plantnet(img_bytes: bytes, api_key: str) -> list[tuple[str, float]]:
    boundary = "pf-eval-boundary"
    parts = []
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="images"; filename="photo.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode() + img_bytes + b"\r\n")
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="organs"\r\n\r\nleaf\r\n'.encode())
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    url = f"https://my-api.plantnet.org/v2/identify/all?api-key={api_key}&no-reject=true&include-related-images=false"
    req = urllib.request.Request(url, data=body, headers={
        "content-type": f"multipart/form-data; boundary={boundary}",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    scored = data.get("results", [])
    out = []
    for r in scored:
        sci = r.get("species", {}).get("scientificNameWithoutAuthor", "")
        score = r.get("score", 0.0)
        if sci:
            out.append((sci, score))
    return out


def normalize(name: str) -> str:
    """Compare on binomial-ish lowercase, no authors."""
    return " ".join(name.lower().split()[:2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-taxon", type=int, default=3)
    parser.add_argument("--limit-taxa", type=int, default=0, help="only first N taxa")
    parser.add_argument("--reuse-photos", action="store_true", help="skip downloads if photos exist")
    args = parser.parse_args()

    bioclip_url = os.environ.get("BIOCLIP_URL", "https://kylebrodeur--plantfluent-bioclip-bioclipservice-web.modal.run/v1/identify")
    bioclip_token = os.environ.get("BIOCLIP_TOKEN", "")
    plantnet_key = os.environ.get("PLANTNET_API_KEY", "")
    if not bioclip_token:
        sys.exit("BIOCLIP_TOKEN not set (1Password: PlantFluent Modal BioCLIP)")
    if not plantnet_key:
        sys.exit("PLANTNET_API_KEY not set (1Password: PlantNet API)")

    taxa = DEFAULT_TAXA[: args.limit_taxa] if args.limit_taxa else DEFAULT_TAXA

    results = {"per_image": [], "summary": {}}
    for taxon in taxa:
        print(f"\n=== {taxon} ===")
        taxon_dir = PHOTO_DIR / taxon.replace(" ", "_")
        taxon_dir.mkdir(parents=True, exist_ok=True)
        if args.reuse_photos and any(taxon_dir.glob("*.jpg")):
            photos = [(p.name, p.read_bytes()) for p in sorted(taxon_dir.glob("*.jpg"))][: args.per_taxon]
        else:
            photos = fetch_commons_images(taxon, args.per_taxon)
            for title, blob in photos:
                safe = title.replace("File:", "").replace(" ", "_").replace("/", "_")
                (taxon_dir / safe).write_bytes(blob)
        if not photos:
            print("  no photos found on Commons; skipping taxon")
            continue

        for idx, (title, blob) in enumerate(photos):
            row = {"taxon": taxon, "photo": title, "bioclip": None, "plantnet": None}
            try:
                bc = classify_bioclip(blob, bioclip_url, bioclip_token)
                row["bioclip"] = bc[:3]
                top = bc[0][0] if bc else ""
                hit = "HIT" if normalize(top) == normalize(taxon) else "MISS"
                print(f"  bio  [{hit}] {top} ({bc[0][1] if bc else 0:.3f})")
            except Exception as exc:  # noqa: BLE001
                print(f"  bio  ERROR {exc}", file=sys.stderr)
            try:
                pn = classify_plantnet(blob, plantnet_key)
                row["plantnet"] = pn[:3]
                top = pn[0][0] if pn else ""
                hit = "HIT" if normalize(top) == normalize(taxon) else "MISS"
                print(f"  pn   [{hit}] {top} ({pn[0][1] if pn else 0:.2f})")
            except Exception as exc:  # noqa: BLE001
                print(f"  pn   ERROR {exc}", file=sys.stderr)
            results["per_image"].append(row)

    # Summary
    for model in ("bioclip", "plantnet"):
        rows = [r for r in results["per_image"] if r[model]]
        hits = sum(1 for r in rows if r[model] and normalize(r[model][0][0]) == normalize(r["taxon"]))
        results["summary"][model] = {
            "n": len(rows), "top1_hits": hits,
            "accuracy": round(hits / len(rows), 3) if rows else None,
        }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(results["summary"], indent=2))
    print(f"\nfull results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()