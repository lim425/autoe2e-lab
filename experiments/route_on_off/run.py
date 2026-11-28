"""Route on/off: does AutoE2E actually use the route?

For every chosen moment, ask the model twice:
  WITH route    = the dataset's route, as normal                  -> green
  WITHOUT route = the same moment, route switched off by the model's
                  own validity gate (the authors' "route-zero" mode) -> orange
Then compare both with where the human drove (purple).

The camera top-down picture is computed once and reused for the second answer, so the
experiment costs about the same as a normal run (~24 s per moment on a laptop GPU).

    python experiments/route_on_off/run.py --part part-d646a7f28b51e27f
    python experiments/route_on_off/run.py --part part-d646a7f28b51e27f part-8028dd4a50d07877

Outputs (outputs/<config>/route_on_off/):
  <part>/results_<rate>.csv              per-moment errors with and without route
  <part>/error_over_time_<rate>.png      both errors along the drive
  <part>/route_on_off_<rate>.mp4         camera + map with all three paths
  <part>/with_route_vs_human_<rate>.mp4  only the model (with route) vs the human
  summary_<rate>.csv                     one row per part + all parts together
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
from autoe2e_lab.model import (  # noqa: E402
    apply_input_rules, load_model, predict, predict_route_ablation, to_batch,
)
from autoe2e_lab.render import ORANGE, camera_view, plot_bev  # noqa: E402

NAME = "route_on_off"
args = part_args(__doc__.splitlines()[0], default_step=1)
rate = rate_tag(args.step)
cfg = load_config()
exp_dir = experiment_dir(cfg, NAME)
_model = None


def get_model():
    """Load the model once, only if some part still needs predictions."""
    global _model
    if _model is None:
        _model = load_model(cfg)
        _self_check(_model)
    return _model


def _self_check(model):
    """One-time check that reusing the camera picture changes nothing:
    (a) pass 1 of the ablation == the normal predict()
    (b) pass 2 with the route switched back ON == pass 1
    """
    scene = PackedScene(cfg.dataset_dir, args.part[0])
    check = to_batch(scene.decoded(0))
    normal = predict(model, check, cfg)["controls"]
    ablation = predict_route_ablation(model, check, cfg)
    check = apply_input_rules(check, cfg)  # same dataset rules as every real model call
    with torch.no_grad():  # inference only; the model refuses BEV reuse while tracking gradients
        reused_on = model(check["visual_tiles"], check["map_context"], check["visual_history"],
                          check["egomotion_history"], route_mask=check["route_mask"],
                          route_valid=check["route_valid"], map_valid=check["map_valid"],
                          precomputed_image_bev=ablation["image_bev"], mode="infer",
                          compute_bev_segmentation=False, compute_route_reconstruction=False)
    reused_on = (reused_on[0] if isinstance(reused_on, tuple) else reused_on).reshape(64, 2).cpu()
    diff_a = (ablation["with_route"]["controls"] - normal).abs().max().item()
    diff_b = (reused_on - ablation["with_route"]["controls"]).abs().max().item()
    print(f"Check (a) ablation pass 1 vs normal run:      max difference {diff_a:.2e}")
    print(f"Check (b) reused camera picture, route on:    max difference {diff_b:.2e}")
    if max(diff_a, diff_b) > 1e-4:
        raise SystemExit("Reusing the camera picture changed the answer; stopping.")


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
    for n, i in enumerate(todo, 1):
        model = get_model()
        t0 = time.perf_counter()
        sample = scene.decoded(i)
        result = predict_route_ablation(model, to_batch(sample), cfg)
        np.savez(pred_dir / f"{i:04d}.npz",
                 with_xy=result["with_route"]["pred_xy"].numpy(),
                 without_xy=result["without_route"]["pred_xy"].numpy(),
                 with_controls=result["with_route"]["controls"].numpy(),
                 without_controls=result["without_route"]["controls"].numpy(),
                 route_valid=bool(sample["route_valid"]))
        print(f"  [{n}/{len(todo)}] moment {i:3d} done in {time.perf_counter() - t0:4.1f} s", flush=True)

    # Stage 2: compare, plot, and make the videos
    rows, frames = [], {"three": [], "two": []}
    for i in indices:
        sample, raw = scene.decoded(i), scene.raw(i)
        saved = np.load(pred_dir / f"{i:04d}.npz")
        gt_xy, valid = sample["trajectory_xy_m"].numpy(), sample["trajectory_valid"].numpy()
        meta = sample["navigation_metadata"]
        row = {"moment": i, "t_s": i / 10,
               "speed_kmh": round(float(sample["initial_speed_mps"]) * 3.6, 1),
               "route_valid": bool(saved["route_valid"]),
               "maneuver": meta.get("route_maneuver", ""),
               "intersection": bool(meta.get("route_intersection", False)),
               "err_with_route_m": ade3(saved["with_xy"], gt_xy, valid),
               "err_without_route_m": ade3(saved["without_xy"], gt_xy, valid)}
        row["route_helps_m"] = row["err_without_route_m"] - row["err_with_route_m"]
        rows.append(row)

        # Video 1: all three paths. Video 2: only the model with route vs the human.
        for name, show_without in (("three", True), ("two", False)):
            extra_cam = [(saved["without_xy"], ORANGE)] if show_without else []
            extra_bev = [(saved["without_xy"], ORANGE, "without route")] if show_without else []
            fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(16, 8), dpi=80,
                                                 gridspec_kw={"width_ratios": [1.6, 1]})
            ax_cam.imshow(camera_view(raw, saved["with_xy"], gt_xy, valid, cfg.ground_z_m,
                                      cfg.front_camera_index, extra=extra_cam))
            ax_cam.axis("off")
            plot_bev(ax_bev, sample, saved["with_xy"], gt_xy, valid,
                     pred_label="AutoE2E (with route)" if not show_without else "with route",
                     extra=extra_bev)
            ax_bev.legend(loc="lower right")
            if show_without:
                title = (f"t = {i / 10:4.1f} s   speed {row['speed_kmh']:3.0f} km/h   "
                         f"error next 3 s:  with route {row['err_with_route_m']:.2f} m  |  "
                         f"without route {row['err_without_route_m']:.2f} m\n"
                         "purple = human,  green = AutoE2E with route,  orange = AutoE2E without route")
            else:
                title = (f"t = {i / 10:4.1f} s    speed {row['speed_kmh']:3.0f} km/h    "
                         f"avg error next 3 s: {row['err_with_route_m']:.2f} m\n"
                         "purple = human driver,  green = AutoE2E (with route)")
            fig.suptitle(title, fontsize=13)
            fig.text(0.01, 0.01, cfg.attribution, fontsize=9, color="gray")
            fig.tight_layout()
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)
            if show_without:
                imageio.imwrite(frame_dir / f"{i:04d}.png", frame)
            frames[name].append(frame)

    with (out_dir / f"results_{rate}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fps = 10 / args.step
    for name, video in (("three", f"{NAME}_{rate}.mp4"), ("two", f"with_route_vs_human_{rate}.mp4")):
        imageio.mimwrite(out_dir / video, frames[name], fps=fps, codec="libx264", quality=8,
                         macro_block_size=16, ffmpeg_params=["-g", str(max(1, round(fps)))])

    t = np.array([r["t_s"] for r in rows])
    w = np.array([r["err_with_route_m"] for r in rows])
    wo = np.array([r["err_without_route_m"] for r in rows])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(t, w, color="limegreen", lw=2, label="with route")
    ax.plot(t, wo, color="orange", lw=2, label="without route")
    ax.set_xlabel("time in the drive [s]")
    ax.set_ylabel("avg error next 3 s [m]")
    ax.set_title(f"{part}: does the route help?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"error_over_time_{rate}.png", dpi=90)
    plt.close(fig)
    print(f"Saved in {out_dir}")


def summarize(rows: list[dict]) -> dict:
    """Only moments where the dataset has a route can show a route effect."""
    ok = [r for r in rows if r["route_valid"]
          and np.isfinite(r["err_with_route_m"]) and np.isfinite(r["err_without_route_m"])]
    d = np.array([r["route_helps_m"] for r in ok])
    return {
        "moments": len(rows),
        "moments_with_route": len(ok),
        "err_with_route_m": float(np.mean([r["err_with_route_m"] for r in ok])) if ok else float("nan"),
        "err_without_route_m": float(np.mean([r["err_without_route_m"] for r in ok])) if ok else float("nan"),
        "helped_pct": float(np.mean(d > 0.05) * 100) if ok else float("nan"),
        "hurt_pct": float(np.mean(d < -0.05) * 100) if ok else float("nan"),
        "same_pct": float(np.mean(np.abs(d) <= 0.05) * 100) if ok else float("nan"),
    }


for part in args.part:
    run_part(part)
write_summary(exp_dir, rate, summarize)
