"""List the scenes in the dataset, or download one scene from Hugging Face.

    python scripts/download_scene.py --list                 # 15 longest scenes + sizes
    python scripts/download_scene.py part-xxxxxxxxxxxxxxxx  # download one scene

Needs shards/manifest.json, which the README's first download step fetches.
"""
import argparse
import subprocess
import sys

from autoe2e_lab.config import load_config
from autoe2e_lab.data import list_parts

REPO_ID = "AutowareFoundation/auto_e2e"

parser = argparse.ArgumentParser()
parser.add_argument("part", nargs="?")
parser.add_argument("--list", action="store_true")
args = parser.parse_args()

cfg = load_config()
parts = sorted(list_parts(cfg.dataset_dir), key=lambda p: -p["moments"])

if args.list or not args.part:
    print(f"{len(parts)} scenes. Longest 15:")
    print(f"  {'part':26s} {'moments':>8s} {'seconds':>8s} {'download':>10s}")
    for p in parts[:15]:
        print(f"  {p['part']:26s} {p['moments']:8d} {p['seconds']:8.1f} {p['download_mb']:8.0f} MB")
    sys.exit(0)

if args.part not in {p["part"] for p in parts}:
    sys.exit(f"Unknown part {args.part!r}. Use --list to see the available scenes.")
prefix = cfg.dataset_dir.relative_to(cfg.data_root)
# One --include per pattern: with several patterns after a single --include, the hf CLI
# treats the extra ones as literal filenames and fails with "File not found".
cmd = ["hf", "download", REPO_ID,
       "--include", f"{prefix}/shards/{args.part}-*",
       "--include", f"{prefix}/pool/{args.part}/*",
       "--local-dir", str(cfg.data_root)]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
