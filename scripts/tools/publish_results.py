"""Copy the small, shareable result files from outputs/ (not committed) to results/ (committed).

    outputs/  = workshop: everything the scripts write (big, on your laptop only)
    results/  = display shelf: small tables and charts that go to GitHub

Only files matching ALLOWLIST are copied: CSV tables and charts without camera images
or dataset pictures. Videos, camera pictures and cached predictions never leave outputs/.
The folder layout is kept, e.g.
    outputs/kitscenes_val/route_on_off/<part>/results_10hz.csv
 -> results/kitscenes_val/route_on_off/<part>/results_10hz.csv

    python scripts/tools/publish_results.py            # copy new/changed files
    python scripts/tools/publish_results.py --dry-run  # only show what would be copied

Then review with `git status` and commit results/ yourself.
To publish a new kind of result, add its pattern to ALLOWLIST.
"""
import argparse
import filecmp
import shutil
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS, RESULTS = LAB_ROOT / "outputs", LAB_ROOT / "results"

# Patterns relative to outputs/: <dataset>/<experiment>/<part>/<file>.
# Any experiment that uses these standard file names is published automatically.
ALLOWLIST = [
    "*/route_scan.csv",                          # tools/scan_routes.py: which scenes have a route
    "*/*/summary_*.csv",                         # experiment: all parts together
    "*/*/part-*/results_*.csv",                  # experiment: one row per moment
    "*/*/part-*/error_over_time_*.png",          # chart only, no camera/map image
    "*/*/part-*/average_by_setup_*.png",         # chart only, no camera/map image
]
MAX_BYTES = 2_000_000  # safety net: nothing big should ever match

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="show what would be copied, copy nothing")
args = parser.parse_args()

files = sorted({p for pattern in ALLOWLIST for p in OUTPUTS.glob(pattern) if p.is_file()})
new = changed = same = 0
for src in files:
    rel = src.relative_to(OUTPUTS)
    dst = RESULTS / rel
    if src.stat().st_size > MAX_BYTES:
        raise SystemExit(f"Refusing to publish {rel}: {src.stat().st_size / 1e6:.1f} MB is too big")
    if dst.exists() and filecmp.cmp(src, dst, shallow=False):
        same += 1
        continue
    status = "changed" if dst.exists() else "new"
    new += status == "new"
    changed += status == "changed"
    print(f"  {status:8s} {rel}  ({src.stat().st_size / 1e3:.0f} KB)")
    if not args.dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

verb = "Would copy" if args.dry_run else "Copied"
print(f"\n{verb} {new} new and {changed} changed file(s); {same} already up to date.")
if not args.dry_run and new + changed:
    print("Next: git status results/   then commit if it looks right.")
