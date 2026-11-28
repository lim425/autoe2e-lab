"""Camera count: how much worse does AutoE2E get with fewer cameras?

The checkpoint always needs 6 camera images. To simulate a vehicle with fewer cameras,
the missing cameras are replaced by black images: now AND in all 7 past frames (and, for
the front camera, also its two extra high-resolution copies). Map, route and motion
history stay exactly as in the dataset.

Setups (camera slots: 0 front, 1 front-left, 2 front-right, 3 rear-left, 4 rear, 5 rear-right):
  6 cameras   all real                         -> the normal model
  3 front     0, 1, 2 real; rear 3 black
  front only  0 real; other 5 black
  no cameras  all 6 black (reference: what map + route + motion alone give)

Each setup needs its own full camera pass (~24 s on a laptop GPU), so 4 setups take
~1.6 min per moment. --step 10 on a 34 s drive = 34 moments = ~55 min. Resumable.

    python experiments/camera_count/run.py --part part-d646a7f28b51e27f
    python experiments/camera_count/run.py --part part-d646a7f28b51e27f part-8028dd4a50d07877

Outputs (outputs/<config>/camera_count/):
  <part>/results_<rate>.csv           per-moment error for every setup
  <part>/error_over_time_<rate>.png   all setups along the drive
  <part>/average_by_setup_<rate>.png  average error per setup
  <part>/biggest_gap_<rate>.png       top-down view of the moment where fewer cameras hurt most
  summary_<rate>.csv                  one row per part + all parts together

Note: this is still open-loop on KIT car data with a model trained on 6 cameras. It tells
you how much THIS model leans on each camera group, not how another vehicle would drive.
"""
import csv
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from autoe2e_lab.config import load_config  # noqa: E402
from autoe2e_lab.data import PackedScene  # noqa: E402
from autoe2e_lab.experiment import ade3, experiment_dir, part_args, rate_tag, write_summary  # noqa: E402
from autoe2e_lab.model import load_model, predict, to_batch  # noqa: E402
from autoe2e_lab.render import GEOM, bev_background  # noqa: E402

NAME = "camera_count"
SETUPS = {                      # name -> camera slots that stay REAL
    "6 cameras": [0, 1, 2, 3, 4, 5],
    "3 front": [0, 1, 2],
    "front only": [0],
    "no cameras": [],
}
COLORS = {"6 cameras": "limegreen", "3 front": "royalblue", "front only": "darkorange",
          "no cameras": "black"}
# ImageNet colour normalisation used by the data loader: a black pixel (0) becomes -mean/std.
BLACK = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(3, 1, 1)

args = part_args(__doc__.splitlines()[0], default_step=10)
rate = rate_tag(args.step)
cfg = load_config()
exp_dir = experiment_dir(cfg, NAME)
_model = None


def keep_cameras(sample: dict, keep: list[int]) -> dict:
    """Black out every camera slot NOT in `keep`, in the current image and all past frames."""
    s = dict(sample)
    drop = [i for i in range(6) if i not in keep]
    if not drop:
        return s
    s["visual_tiles"] = s["visual_tiles"].clone()
    s["visual_tiles"][drop] = BLACK
    s["camera_history_tiles"] = s["camera_history_tiles"].clone()
    s["camera_history_tiles"][:, drop] = BLACK
    if 0 in drop:  # the front camera is also given at 1024 px and as a 512 px companion
        for key in ("front_camera_tile", "front_camera_fpn_tile"):
            if key in s:
                s[key] = BLACK.expand_as(s[key]).clone()
    return s


def get_model():
    """Load the model once (only if needed) and check that blacking out works as intended."""
    global _model
    if _model is None:
        _model = load_model(cfg)
        s0 = PackedScene(cfg.dataset_dir, args.part[0]).decoded(0)
        s1 = keep_cameras(s0, [0])
        assert torch.equal(s1["visual_tiles"][0], s0["visual_tiles"][0])
        assert torch.equal(s1["front_camera_tile"], s0["front_camera_tile"])
        assert torch.equal(s1["visual_tiles"][3], BLACK.expand(3, 512, 512))
        assert torch.equal(s1["camera_history_tiles"][:, 5], BLACK.expand(7, 3, 512, 512))
        assert torch.equal(keep_cameras(s0, SETUPS["6 cameras"])["visual_tiles"], s0["visual_tiles"])
        print("Self-check OK: kept cameras untouched, dropped cameras black in all frames.")
    return _model


def run_part(part: str) -> None:
    scene = PackedScene(cfg.dataset_dir, part)
    out_dir = exp_dir / part
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    indices = list(range(0, len(scene), args.step))
    print(f"\n=== {part}: {len(scene)} moments, using {len(indices)} (every {args.step}), "
          f"{len(SETUPS)} setups")

    # Stage 1: run every setup on every moment and save the answers (resumable)
    todo = [i for i in indices if not (pred_dir / f"{i:04d}.npz").exists()]
    start = time.time()
    for n, i in enumerate(todo, 1):
        model = get_model()
        sample = scene.decoded(i)
        preds = {name: predict(model, to_batch(keep_cameras(sample, keep)), cfg)["pred_xy"].numpy()
                 for name, keep in SETUPS.items()}
        np.savez(pred_dir / f"{i:04d}.npz", **preds)
        per = (time.time() - start) / n
        print(f"  [{n}/{len(todo)}] moment {i} (t = {i / 10:.1f} s) done, "
              f"~{per * (len(todo) - n) / 60:.0f} min left", flush=True)

    # Stage 2: errors, table, plots
    rows, truth = [], {}
    for i in indices:
        sample = scene.decoded(i)
        gt_xy, valid = sample["trajectory_xy_m"].numpy(), sample["trajectory_valid"].numpy()
        truth[i] = (gt_xy, valid)
        preds = np.load(pred_dir / f"{i:04d}.npz")
        rows.append({"moment": i, "time_s": i / 10,
                     **{name: ade3(preds[name], gt_xy, valid) for name in SETUPS}})

    with (out_dir / f"results_{rate}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["moment", "time_s", *SETUPS])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    t = np.array([r["time_s"] for r in rows])
    errs = {name: np.array([r[name] for r in rows]) for name in SETUPS}

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name in SETUPS:
        ax.plot(t, errs[name], "o-", ms=3, lw=1.5, color=COLORS[name],
                label=f"{name} (mean {np.nanmean(errs[name]):.2f} m)")
    ax.set_xlabel("time in drive (s)")
    ax.set_ylabel("error over next 3 s (m)")
    ax.set_title(f"Fewer cameras: {part} ({len(rows)} moments, {rate})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"error_over_time_{rate}.png", dpi=90)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    means = [np.nanmean(errs[n]) for n in SETUPS]
    bars = ax.bar(list(SETUPS), means, color=[COLORS[n] for n in SETUPS])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.2f} m", ha="center", va="bottom")
    ax.set_ylabel("mean error over next 3 s (m)")
    ax.set_title("Average error by number of cameras (lower = better)")
    fig.tight_layout()
    fig.savefig(out_dir / f"average_by_setup_{rate}.png", dpi=90)
    plt.close(fig)

    # The moment where dropping to the front camera hurt most
    gap = errs["front only"] - errs["6 cameras"]
    k = int(np.nanargmax(gap))
    i = rows[k]["moment"]
    gt_xy, valid = truth[i]
    preds = np.load(pred_dir / f"{i:04d}.npz")
    sample = scene.decoded(i)
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(bev_background(sample["map_context"].numpy(), sample["route_mask"].numpy()))
    rc = GEOM.ego_to_pixel(gt_xy[valid])
    ax.plot(rc[:, 1], rc[:, 0], color="purple", lw=5, label="human")
    for name in SETUPS:
        rc = GEOM.ego_to_pixel(preds[name])
        ax.plot(rc[:, 1], rc[:, 0], color=COLORS[name], lw=2.5, label=f"{name} ({errs[name][k]:.2f} m)")
    ax.plot(GEOM.ego_anchor_col, GEOM.ego_anchor_row, "r^", ms=10)
    ax.set_xlim(GEOM.ego_anchor_col - 70, GEOM.ego_anchor_col + 70)
    ax.set_ylim(GEOM.ego_anchor_row + 30, GEOM.ego_anchor_row - 200)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"Biggest loss from fewer cameras, t = {i / 10:.1f} s\n(error over the next 3 s)")
    fig.tight_layout()
    fig.savefig(out_dir / f"biggest_gap_{rate}.png", dpi=90)
    plt.close(fig)
    print(f"Saved in {out_dir}")


def summarize(rows: list[dict]) -> dict:
    out = {"moments": len(rows)}
    base = np.array([r["6 cameras"] for r in rows])
    for name in SETUPS:
        e = np.array([r[name] for r in rows])
        out[f"err_{name.replace(' ', '_')}_m"] = float(np.nanmean(e))
        if name != "6 cameras":
            out[f"worse_pct_{name.replace(' ', '_')}"] = float(np.mean(e - base > 0.1) * 100)
    return out


for part in args.part:
    run_part(part)
write_summary(exp_dir, rate, summarize)
