"""Run the model on ONE sample: print its error and draw camera + map views.

    python scripts/basics/03_infer_one_sample.py          # sample 0 of the default scene
    python scripts/basics/03_infer_one_sample.py 10       # sample 10
"""
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from autoe2e_lab.config import load_config  # noqa: E402
from autoe2e_lab.data import PackedScene  # noqa: E402
from autoe2e_lab.model import load_model, predict, to_batch  # noqa: E402
from autoe2e_lab.render import camera_view, plot_bev  # noqa: E402

cfg = load_config()
scene = PackedScene(cfg.dataset_dir, cfg.default_part)
index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
raw, sample = scene.raw(index), scene.decoded(index)
print(f"Sample #{index}/{len(scene) - 1}: {sample['sample_uid']}")
print(f"Speed now (v0): {sample['initial_speed_mps']:.1f} m/s")

model = load_model(cfg)
torch.cuda.synchronize()
t0 = time.perf_counter()
result = predict(model, to_batch(sample), cfg)
torch.cuda.synchronize()
print(f"Inference: {time.perf_counter() - t0:.1f} s, peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

pred_xy, controls = result["pred_xy"].numpy(), result["controls"].numpy()
gt_xy, valid = sample["trajectory_xy_m"].numpy(), sample["trajectory_valid"].numpy()
err = np.linalg.norm(pred_xy - gt_xy, axis=1)
n3 = int(valid[:30].sum())
print(f"\nValid future steps: {valid.sum()}/64")
print(f"Average error over first 3 s: {err[:n3].mean():.2f} m")
print(f"Error at the 3 s point:       {err[n3 - 1]:.2f} m")
print("First 5 predicted steps [accel m/s², curvature 1/m]:")
print(np.round(controls[:5], 3))

out = cfg.output_dir / "basics" / "single"
out.mkdir(parents=True, exist_ok=True)
fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1.6, 1]})
ax_cam.imshow(camera_view(raw, pred_xy, gt_xy, valid, cfg.ground_z_m, cfg.front_camera_index))
ax_cam.axis("off")
plot_bev(ax_bev, sample, pred_xy, gt_xy, valid)
ax_bev.legend(loc="lower right")
fig.suptitle(f"{sample['sample_uid']}   avg error first 3 s: {err[:n3].mean():.2f} m\n"
             f"purple = human, green = AutoE2E    {cfg.attribution}")
fig.tight_layout()
path = out / f"{cfg.default_part}_{index:03d}.png"
fig.savefig(path, dpi=80)
print(f"\nSaved {path}")
