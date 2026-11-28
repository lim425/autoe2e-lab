"""Sanity checks: break things on purpose and confirm the results react as they should.

If the green line were faked, or the setup were wrong, these tests would come out
differently than expected.

    python experiments/sanity_checks/run.py                              # 3 moments, default drive
    python experiments/sanity_checks/run.py --part part-xxxx --moments 0 50 100

Outputs (outputs/<config>/sanity_checks/<part>/):
  results_checks.csv   error of every test at every moment
  <moment>.png         top-down view with every test's prediction

Tests per moment:
  1. normal           trained model, real inputs                 -> small error
  2. run again        identical second run                       -> exactly the same answer
  3. empty brain      same model structure, random weights        -> nonsense
  4. black cameras    every camera image replaced by black        -> answer changes, worse
  5. cameras shuffled the 6 camera images in a different order     -> worse
  6. first step       first predicted point vs speed x 0.1 s     -> about equal
"""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from autoe2e_lab.config import load_config  # noqa: E402
from autoe2e_lab.data import PackedScene  # noqa: E402
from autoe2e_lab.experiment import ade3, experiment_dir  # noqa: E402
from autoe2e_lab.model import load_model, predict, to_batch  # noqa: E402
from autoe2e_lab.render import GEOM, bev_background  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--part", default="part-d646a7f28b51e27f")
parser.add_argument("--moments", type=int, nargs="+", default=[0, 100, 200])
args = parser.parse_args()

cfg = load_config()
scene = PackedScene(cfg.dataset_dir, args.part)
out_dir = experiment_dir(cfg, "sanity_checks") / args.part
out_dir.mkdir(parents=True, exist_ok=True)

# ImageNet colour normalisation used by the data loader: a black pixel (0) becomes -mean/std.
BLACK = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(3, 1, 1)
SHUFFLE = [3, 4, 5, 0, 1, 2]  # front <-> rear-left, left <-> rear, right <-> rear-right


def black_cameras(sample: dict) -> dict:
    s = dict(sample)
    for key in ("visual_tiles", "front_camera_tile", "front_camera_fpn_tile", "camera_history_tiles"):
        if key in s:
            s[key] = BLACK.expand_as(s[key]).clone()
    return s


def shuffled_cameras(sample: dict) -> dict:
    """Swap the camera IMAGES but keep each slot's calibration: images and geometry disagree."""
    s = dict(sample)
    s["visual_tiles"] = s["visual_tiles"][SHUFFLE]
    s["camera_history_tiles"] = s["camera_history_tiles"][:, SHUFFLE]
    # The native front image and its companion must follow slot 0's new image (now cam 3).
    if "front_camera_tile" in s:
        s["front_camera_tile"] = torch.nn.functional.interpolate(
            s["visual_tiles"][0:1], size=s["front_camera_tile"].shape[-2:], mode="bilinear"
        )[0]
    if "front_camera_fpn_tile" in s:
        s["front_camera_fpn_tile"] = s["visual_tiles"][0].clone()
    return s


trained = load_model(cfg)
torch.manual_seed(0)
untrained = load_model(cfg, trained=False)

results = {name: [] for name in ["normal", "run again", "empty brain", "black cameras", "cameras shuffled"]}
first_step = []
for m in args.moments:
    sample = scene.decoded(m)
    gt_xy, valid = sample["trajectory_xy_m"].numpy(), sample["trajectory_valid"].numpy()
    runs = {
        "normal": predict(trained, to_batch(sample), cfg),
        "run again": predict(trained, to_batch(sample), cfg),
        "empty brain": predict(untrained, to_batch(sample), cfg),
        "black cameras": predict(trained, to_batch(black_cameras(sample)), cfg),
        "cameras shuffled": predict(trained, to_batch(shuffled_cameras(sample)), cfg),
    }
    for name, r in runs.items():
        results[name].append((m, ade3(r["pred_xy"].numpy(), gt_xy, valid), r))
    same = torch.equal(runs["normal"]["controls"], runs["run again"]["controls"])
    v0 = float(sample["initial_speed_mps"])
    step1 = float(np.linalg.norm(runs["normal"]["pred_xy"][0].numpy()))
    first_step.append((m, v0 * 0.1, step1))
    print(f"moment {m:3d} (t = {m / 10:4.1f} s) done   identical second run: {same}", flush=True)

    # Picture: every test's prediction on the top-down map for this moment.
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(bev_background(sample["map_context"].numpy(), sample["route_mask"].numpy()))
    rc = GEOM.ego_to_pixel(gt_xy[valid])
    ax.plot(rc[:, 1], rc[:, 0], color="purple", lw=5, label="human")
    colors = {"normal": "limegreen", "empty brain": "red", "black cameras": "black",
              "cameras shuffled": "darkorange"}
    for name, color in colors.items():
        xy = runs[name]["pred_xy"].numpy()
        rc = GEOM.ego_to_pixel(xy)
        ax.plot(rc[:, 1], rc[:, 0], color=color, lw=2.5,
                label=f"{name} ({ade3(xy, gt_xy, valid):.2f} m)")
    ax.plot(GEOM.ego_anchor_col, GEOM.ego_anchor_row, "r^", ms=10)
    ax.set_xlim(GEOM.ego_anchor_col - 70, GEOM.ego_anchor_col + 70)
    ax.set_ylim(GEOM.ego_anchor_row + 30, GEOM.ego_anchor_row - 200)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"Sanity checks, t = {m / 10:.1f} s\n(error over the next 3 s)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{m:04d}.png", dpi=90)
    plt.close(fig)

with (out_dir / "results_checks.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["moment", *results])
    for k, m in enumerate(args.moments):
        w.writerow([m, *(f"{results[name][k][1]:.4f}" for name in results)])

print("\nAVERAGE ERROR OVER THE NEXT 3 s (lower = better)")
for name, rows in results.items():
    errs = [e for _, e, _ in rows]
    print(f"  {name:17s} {np.mean(errs):6.2f} m    per moment: {', '.join(f'{e:.2f}' for e in errs)}")
print("\nFIRST PREDICTED STEP (0.1 s ahead): expected ~ speed x 0.1 s")
for m, expected, got in first_step:
    print(f"  moment {m:3d}: expected {expected:.2f} m, model {got:.2f} m")
print(f"\nPictures: {out_dir}")
