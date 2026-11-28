"""Map on/off: does AutoE2E actually use the map?

For every chosen moment, ask the model twice (the route stays on in both):
  WITH map     = the dataset's map, as normal                                   -> green
  WITHOUT map  = the same moment, map switched off by the model's own validity
                 gate (map_valid = False multiplies the map by 0)               -> blue
Then compare both with where the human drove (purple).

The camera top-down picture is built from the cameras only, before the map is mixed in,
so it is computed once per moment and reused for the second answer. Self-checks at the
start confirm that reusing it changes nothing and that the gate equals an all-zero map.

    python experiments/map_on_off/run.py --part part-d646a7f28b51e27f
    python experiments/map_on_off/run.py --part part-d646a7f28b51e27f part-8028dd4a50d07877

Outputs (outputs/<config>/map_on_off/):
  <part>/results_<rate>.csv           per-moment errors with and without map
  <part>/error_over_time_<rate>.png   both errors along the drive
  <part>/average_by_setup_<rate>.png  average error with and without map
  <part>/map_on_off_<rate>.mp4        camera + map with both paths and the human
  <part>/without_map_vs_human_<rate>.mp4  only the model without map vs the human
  summary_<rate>.csv                  one row per part + all parts together

Note: the model was trained with a map, so "without map" is an input it never practised.
A large error without the map shows that it depends on the map, and also that it never
learned to drive without one.
"""
import csv
import time

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from autoe2e_lab.config import load_config  # noqa: E402
from autoe2e_lab.data import PackedScene  # noqa: E402
from autoe2e_lab.experiment import ade3, experiment_dir, part_args, rate_tag, write_summary  # noqa: E402
from autoe2e_lab.model import load_model, predict, predict_navigation_ablation, to_batch  # noqa: E402
from autoe2e_lab.render import camera_view, plot_bev  # noqa: E402

NAME = "map_on_off"
SETUPS = {"with map": {"map": True, "route": True},
          "without map": {"map": False, "route": True}}
BLUE = (65, 105, 225)

args = part_args(__doc__.splitlines()[0], default_step=1)
rate = rate_tag(args.step)
cfg = load_config()
exp_dir = experiment_dir(cfg, NAME)
_model = None


def get_model():
    """Load the model once (only if needed) and run the self-checks."""
    global _model
    if _model is None:
        _model = load_model(cfg)
        _self_check(_model)
    return _model


def _self_check(model):
    """(a) reused camera picture, with map    == the normal predict()
    (b) reused camera picture, without map == a full run with the map off
    (c) map gate off == an all-zero map with the gate on (the gate really removes the map)
    """
    sample = PackedScene(cfg.dataset_dir, args.part[0]).decoded(0)
    if not bool(sample["map_valid"]):
        raise SystemExit(f"{args.part[0]} has no map; nothing to switch off.")
    batch = to_batch(sample)
    abl = predict_navigation_ablation(model, batch, cfg, SETUPS)
    normal = predict(model, batch, cfg)["controls"]
    map_gate_off = predict(model, {**batch, "map_valid": torch.zeros_like(batch["map_valid"])}, cfg)["controls"]
    map_zero = predict(model, {**batch, "map_context": torch.zeros_like(batch["map_context"])}, cfg)["controls"]
    diffs = {
        "(a) reuse, with map     vs normal run    ": (abl["with map"]["controls"] - normal).abs().max().item(),
        "(b) reuse, without map  vs full run      ": (abl["without map"]["controls"] - map_gate_off).abs().max().item(),
        "(c) map gate off        vs all-zero map  ": (map_gate_off - map_zero).abs().max().item(),
    }
    for label, d in diffs.items():
        print(f"Check {label}: max difference {d:.2e}")
    if max(diffs.values()) > 1e-4:
        raise SystemExit("A self-check failed; stopping.")


def run_part(part: str) -> None:
    scene = PackedScene(cfg.dataset_dir, part)
    out_dir = exp_dir / part
    pred_dir, frame_dir = out_dir / "predictions", out_dir / "frames"
    pred_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(exist_ok=True)
    indices = list(range(0, len(scene), args.step))
    print(f"\n=== {part}: {len(scene)} moments, using {len(indices)} (every {args.step})")

    # Stage 1: run the model (twice per moment) and save the answers (resumable)
    todo = [i for i in indices if not (pred_dir / f"{i:04d}.npz").exists()]
    start = time.time()
    for n, i in enumerate(todo, 1):
        model = get_model()
        sample = scene.decoded(i)
        result = predict_navigation_ablation(model, to_batch(sample), cfg, SETUPS)
        np.savez(pred_dir / f"{i:04d}.npz",
                 with_xy=result["with map"]["pred_xy"].numpy(),
                 without_xy=result["without map"]["pred_xy"].numpy(),
                 with_controls=result["with map"]["controls"].numpy(),
                 without_controls=result["without map"]["controls"].numpy(),
                 map_valid=bool(sample["map_valid"]), route_valid=bool(sample["route_valid"]))
        per = (time.time() - start) / n
        print(f"  [{n}/{len(todo)}] moment {i:3d} done, ~{per * (len(todo) - n) / 60:.0f} min left",
              flush=True)

    # Stage 2: compare, plot, and make the video
    rows, frames = [], {"both": [], "without": []}
    for i in indices:
        sample, raw = scene.decoded(i), scene.raw(i)
        saved = np.load(pred_dir / f"{i:04d}.npz")
        gt_xy, valid = sample["trajectory_xy_m"].numpy(), sample["trajectory_valid"].numpy()
        row = {"moment": i, "t_s": i / 10,
               "speed_kmh": round(float(sample["initial_speed_mps"]) * 3.6, 1),
               "map_valid": bool(saved["map_valid"]), "route_valid": bool(saved["route_valid"]),
               "maneuver": sample["navigation_metadata"].get("route_maneuver", ""),
               "err_with_map_m": ade3(saved["with_xy"], gt_xy, valid),
               "err_without_map_m": ade3(saved["without_xy"], gt_xy, valid)}
        row["map_helps_m"] = row["err_without_map_m"] - row["err_with_map_m"]
        rows.append(row)

        # Video 1: both answers. Video 2: only the answer without map vs the human.
        for name in ("both", "without"):
            both = name == "both"
            fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(16, 8), dpi=80,
                                                 gridspec_kw={"width_ratios": [1.6, 1]})
            if both:
                ax_cam.imshow(camera_view(raw, saved["with_xy"], gt_xy, valid, cfg.ground_z_m,
                                          cfg.front_camera_index, extra=[(saved["without_xy"], BLUE)]))
                plot_bev(ax_bev, sample, saved["with_xy"], gt_xy, valid, pred_label="with map",
                         extra=[(saved["without_xy"], BLUE, "without map")])
                title = (f"t = {i / 10:4.1f} s   speed {row['speed_kmh']:3.0f} km/h   "
                         f"error next 3 s:  with map {row['err_with_map_m']:.2f} m  |  "
                         f"without map {row['err_without_map_m']:.2f} m\n"
                         "purple = human,  green = AutoE2E with map,  blue = AutoE2E without map")
            else:
                ax_cam.imshow(camera_view(raw, saved["without_xy"], gt_xy, valid, cfg.ground_z_m,
                                          cfg.front_camera_index, pred_color=BLUE))
                plot_bev(ax_bev, sample, saved["without_xy"], gt_xy, valid,
                         pred_label="AutoE2E without map", pred_color=BLUE)
                title = (f"t = {i / 10:4.1f} s    speed {row['speed_kmh']:3.0f} km/h    "
                         f"avg error next 3 s: {row['err_without_map_m']:.2f} m\n"
                         "purple = human driver,  blue = AutoE2E without map")
            ax_cam.axis("off")
            ax_bev.legend(loc="lower right")
            ax_bev.set_title("map drawn for reference; the blue answer did not see it", fontsize=9)
            fig.suptitle(title, fontsize=13)
            fig.text(0.01, 0.01, cfg.attribution, fontsize=9, color="gray")
            fig.tight_layout()
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)
            if both:
                imageio.imwrite(frame_dir / f"{i:04d}.png", frame)
            frames[name].append(frame)

    with (out_dir / f"results_{rate}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fps = 10 / args.step
    for name, video in (("both", f"{NAME}_{rate}.mp4"), ("without", f"without_map_vs_human_{rate}.mp4")):
        imageio.mimwrite(out_dir / video, frames[name], fps=fps, codec="libx264", quality=8,
                         macro_block_size=16, ffmpeg_params=["-g", str(max(1, round(fps)))])

    t = np.array([r["t_s"] for r in rows])
    w = np.array([r["err_with_map_m"] for r in rows])
    wo = np.array([r["err_without_map_m"] for r in rows])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(t, w, color="limegreen", lw=2, label=f"with map (mean {np.nanmean(w):.2f} m)")
    ax.plot(t, wo, color="royalblue", lw=2, label=f"without map (mean {np.nanmean(wo):.2f} m)")
    ax.set_xlabel("time in the drive [s]")
    ax.set_ylabel("avg error next 3 s [m]")
    ax.set_title(f"{part}: does the map help? ({len(rows)} moments, {rate})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"error_over_time_{rate}.png", dpi=90)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    means = [np.nanmean(w), np.nanmean(wo)]
    bars = ax.bar(["with map", "without map"], means, color=["limegreen", "royalblue"])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.2f} m", ha="center", va="bottom")
    ax.set_ylabel("mean error over next 3 s (m)")
    ax.set_title("Average error with and without map (lower = better)")
    fig.tight_layout()
    fig.savefig(out_dir / f"average_by_setup_{rate}.png", dpi=90)
    plt.close(fig)
    print(f"Saved in {out_dir}")


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if np.isfinite(r["err_with_map_m"]) and np.isfinite(r["err_without_map_m"])]
    d = np.array([r["map_helps_m"] for r in ok])
    return {
        "moments": len(rows),
        "err_with_map_m": float(np.mean([r["err_with_map_m"] for r in ok])),
        "err_without_map_m": float(np.mean([r["err_without_map_m"] for r in ok])),
        "helped_pct": float(np.mean(d > 0.05) * 100),
        "hurt_pct": float(np.mean(d < -0.05) * 100),
        "same_pct": float(np.mean(np.abs(d) <= 0.05) * 100),
    }


for part in args.part:
    run_part(part)
write_summary(exp_dir, rate, summarize)
