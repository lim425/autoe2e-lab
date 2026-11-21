"""Find which scenes have a route, without downloading them.

For each scene, fetch only the first ~2 MB of its shard (.tar) from Hugging Face, find the
first sample's navigation_meta.json, and read its route_valid flag. That flag is exactly
what the model's input pipeline uses to switch the route input on or off.

    python scripts/scan_routes.py          # writes outputs/<config>/route_scan.csv

Note: only the FIRST moment of each scene is checked. Verify a scene fully after downloading.
"""
import csv
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from autoe2e_lab.config import load_config
from autoe2e_lab.data import list_parts

REPO_URL = "https://huggingface.co/AutowareFoundation/auto_e2e/resolve/main"
CHUNK = 2 * 1024 * 1024  # bytes fetched per request

cfg = load_config()
prefix = cfg.dataset_dir.relative_to(cfg.data_root)


def fetch(url: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}",
                                               "User-Agent": "autoe2e-lab-route-scan"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def first_navigation_meta(part: str) -> dict:
    """Walk the tar headers until the first *.navigation_meta.json member."""
    url = f"{REPO_URL}/{prefix}/shards/{part}-train-000000.tar"
    data = fetch(url, 0, CHUNK)
    offset = 0
    while True:
        if offset + 512 > len(data):           # need more bytes for the next header
            data += fetch(url, len(data), CHUNK)
        header = data[offset:offset + 512]
        name = header[:100].rstrip(b"\0").decode()
        if not name:
            raise ValueError("reached end of archive without navigation_meta.json")
        size = int(header[124:136].rstrip(b"\0 ").decode() or "0", 8)
        start = offset + 512
        if name.endswith(".navigation_meta.json"):
            while start + size > len(data):
                data += fetch(url, len(data), CHUNK)
            return json.loads(data[start:start + size])
        offset = start + (size + 511) // 512 * 512  # tar pads each file to 512 bytes


def scan(p: dict) -> dict:
    try:
        meta = first_navigation_meta(p["part"])
        return {**p,
                "route_valid": bool(meta.get("route_valid", False)),
                "map_valid": bool(meta.get("map_valid", False)),
                "maneuver": meta.get("route_maneuver", ""),
                "intersection": bool(meta.get("route_intersection", False)),
                "failure_reasons": meta.get("route_quality_failure_reasons", "")}
    except Exception as error:  # keep going; report the failure in the table
        return {**p, "route_valid": None, "map_valid": None, "maneuver": "",
                "intersection": None, "failure_reasons": f"SCAN ERROR: {error}"}


parts = list_parts(cfg.dataset_dir)
print(f"Scanning {len(parts)} scenes (first moment of each)...")
with ThreadPoolExecutor(max_workers=8) as pool:
    rows = list(pool.map(scan, parts))

rows.sort(key=lambda r: (r["route_valid"] is not True, -r["moments"]))
out = cfg.output_dir / "route_scan.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

with_route = [r for r in rows if r["route_valid"]]
errors = [r for r in rows if r["route_valid"] is None]
print(f"\nRoute available: {len(with_route)}/{len(rows)} scenes   (scan errors: {len(errors)})")
print(f"\nLongest scenes WITH a route:")
print(f"  {'part':26s} {'seconds':>7s} {'download':>9s}  maneuver / intersection")
for r in with_route[:15]:
    print(f"  {r['part']:26s} {r['seconds']:7.1f} {r['download_mb']:7.0f} MB  "
          f"{r['maneuver']} / {r['intersection']}")
reasons = {}
for r in rows:
    if r["route_valid"] is False:
        reasons[r["failure_reasons"] or "(none given)"] = reasons.get(r["failure_reasons"] or "(none given)", 0) + 1
print("\nWhy the others have no route:", reasons)
print(f"\nFull table: {out}")
