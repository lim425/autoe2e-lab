"""Run the model on every moment of a scene, then make a video (autoe2e_realtime_<rate>.mp4).

Stage 1 (slow, ~25 s per moment on an RTX 5060 laptop): run the model and SAVE
each prediction. Re-running skips moments that are already saved.
Stage 2 (fast): draw one picture per moment and join them into videos.

    python scripts/04_make_video.py                        # default scene, every moment
    python scripts/04_make_video.py --part part-xxxx --step 5
"""
import argparse
import time

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoe2e_lab.config import load_config  # noqa: E402
from autoe2e_lab.data import PackedScene  # noqa: E402
from autoe2e_lab.model import load_model, predict, to_batch  # noqa: E402
from autoe2e_lab.render import camera_view, plot_bev  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None, help="dataset config (default: kitscenes_val)")
parser.add_argument("--part", default=None, help="scene to use (default: from config)")
parser.add_argument("--step", type=int, default=1, help="use every Nth moment")
args = parser.parse_args()
rate = f"{round(10 / args.step, 2):g}hz".replace(".", "p")  # e.g. 10hz, 2hz: in every output name

cfg = load_config(args.config) if args.config else load_config()
part = args.part or cfg.default_part
scene = PackedScene(cfg.dataset_dir, part)
out_dir = cfg.output_dir / "video" / part
pred_dir, frame_dir = out_dir / "predictions", out_dir / "frames"
pred_dir.mkdir(parents=True, exist_ok=True)
frame_dir.mkdir(exist_ok=True)
indices = list(range(0, len(scene), args.step))
print(f"Scene {part}: {len(scene)} moments, using {len(indices)} (every {args.step})")

# Stage 1: run the model and save predictions
todo = [i for i in indices if not (pred_dir / f"{i:04d}.npz").exists()]
if todo:
    model = load_model(cfg)
    for n, i in enumerate(todo, 1):
        t0 = time.perf_counter()
        result = predict(model, to_batch(scene.decoded(i)), cfg)
        np.savez(pred_dir / f"{i:04d}.npz",
                 pred_xy=result["pred_xy"].numpy(), controls=result["controls"].numpy())
        print(f"  [{n}/{len(todo)}] moment {i:3d} done in {time.perf_counter() - t0:4.1f} s", flush=True)

# Stage 2: draw frames and join them into videos
frames, errors_3s = [], []
for i in indices:
    sample, raw = scene.decoded(i), scene.raw(i)
    pred_xy = np.load(pred_dir / f"{i:04d}.npz")["pred_xy"]
    gt_xy, valid = sample["trajectory_xy_m"].numpy(), sample["trajectory_valid"].numpy()
    n3 = int(valid[:30].sum())
    err3 = np.linalg.norm(pred_xy - gt_xy, axis=1)[:n3].mean() if n3 else float("nan")
    errors_3s.append(err3)

    fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(16, 8), dpi=80,
                                         gridspec_kw={"width_ratios": [1.6, 1]})  # 1280x640 px
    ax_cam.imshow(camera_view(raw, pred_xy, gt_xy, valid, cfg.ground_z_m, cfg.front_camera_index))
    ax_cam.axis("off")
    plot_bev(ax_bev, sample, pred_xy, gt_xy, valid)
    ax_bev.legend(loc="lower right")
    fig.suptitle(f"t = {i * 0.1:4.1f} s    speed {sample['initial_speed_mps'] * 3.6:3.0f} km/h    "
                 f"avg error next 3 s: {err3:.2f} m\npurple = human driver,  green = AutoE2E", fontsize=13)
    fig.text(0.01, 0.01, cfg.attribution, fontsize=9, color="gray")
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    imageio.imwrite(frame_dir / f"{i:04d}.png", frame)
    frames.append(frame)

fps = 10 / args.step  # plays at the real speed of the drive
path = out_dir / f"autoe2e_realtime_{rate}.mp4"
# -g N = a full keyframe at least every N frames (here: every second of video).
# Without it, H.264 stores one full picture and then only "changes"; at 2 fps almost
# everything changes, and players like VLC show ghost/broken lines when seeking.
imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=16,
                 ffmpeg_params=["-g", str(max(1, round(fps)))])
print(f"Saved {path}  ({len(frames) / fps:.1f} s long)")

e = np.array(errors_3s)
e = e[np.isfinite(e)]
print(f"\nAverage error over the next 3 s, {len(e)} moments: "
      f"mean {e.mean():.2f} m, best {e.min():.2f} m, worst {e.max():.2f} m")
