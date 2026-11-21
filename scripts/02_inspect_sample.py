"""Open ONE sample and look at it (no model). Pictures go to outputs/<config>/inspect/.

    python scripts/02_inspect_sample.py            # sample 0 of the default scene
    python scripts/02_inspect_sample.py 10         # sample 10
"""
import io
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from autoe2e_lab.config import load_config  # noqa: E402
from autoe2e_lab.data import PackedScene  # noqa: E402
from autoe2e_lab.render import GEOM, bev_background  # noqa: E402
from navigation.geometry import MapChannel, RouteChannel  # noqa: E402

cfg = load_config()
scene = PackedScene(cfg.dataset_dir, cfg.default_part)
index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
raw, s = scene.raw(index), scene.decoded(index)
out = cfg.output_dir / "inspect"
out.mkdir(parents=True, exist_ok=True)
print(f"Scene has {len(scene)} samples. Showing #{index}: {s['sample_uid']}\n")

cams = [np.asarray(Image.open(io.BytesIO(raw[f"cam_{i}.jpg"])).convert("RGB")) for i in range(6)]
ego = np.frombuffer(raw["ego.npy"], dtype="<f4")  # raw float32, NOT a real .npy file
ego_hist, ego_future = ego[:256].reshape(64, 4), ego[256:].reshape(64, 2)
gt_xy, valid = s["trajectory_xy_m"].numpy(), s["trajectory_valid"].numpy()

print("MODEL INPUTS")
for k in ["visual_tiles", "front_camera_tile", "camera_history_tiles", "map_context",
          "route_mask", "egomotion_history", "camera_projection_matrix"]:
    print(f"  {k:26s} {tuple(s[k].shape)}")
print("ANSWER KEY (never fed to the model)")
print(f"  trajectory_xy_m            {tuple(gt_xy.shape)}  valid steps: {valid.sum()}/64")
print(f"\nSpeed now: {ego_hist[-1, 0]:.1f} m/s = {ego_hist[-1, 0] * 3.6:.0f} km/h")
print(f"History rows that are zero padding: {int((ego_hist == 0).all(1).sum())}/64")

# Cameras
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(cams[i]); ax.set_title(f"cam_{i}  {cams[i].shape[1]}x{cams[i].shape[0]}"); ax.axis("off")
fig.suptitle(f"{s['sample_uid']}\n{cfg.attribution}"); fig.tight_layout()
fig.savefig(out / f"cameras_{index:02d}.png", dpi=80); plt.close(fig)

# Front camera history (the 7 past frames + now)
history = json.loads(raw["bev_history_index.json"])
fig, axes = plt.subplots(1, 8, figsize=(24, 3.6))
for ax, row, t in zip(axes, history + [None], range(-7, 1)):
    img = cams[0] if row is None else Image.open(io.BytesIO(scene.pool.extractfile(f"{row[0]}.jpg").read()))
    ax.imshow(img); ax.set_title(f"t = {t * 0.5:+.1f} s"); ax.axis("off")
fig.suptitle("Front camera history: 7 past frames + now, 0.5 s apart"); fig.tight_layout()
fig.savefig(out / f"front_history_{index:02d}.png", dpi=70); plt.close(fig)

# All 16 map/route channels
m, r = s["map_context"].numpy(), s["route_mask"].numpy()
layers = [(m[c], MapChannel(c).name) for c in range(14)] + [(r[c], "ROUTE_" + RouteChannel(c).name) for c in range(2)]
fig, axes = plt.subplots(2, 8, figsize=(24, 9))
for ax, (img, name) in zip(axes.flat, layers):
    ax.imshow(img); ax.plot(GEOM.ego_anchor_col, GEOM.ego_anchor_row, "r^")
    ax.set_title(name.replace("_", "\n", 1), fontsize=9); ax.axis("off")
fig.suptitle("Map raster: 14 map + 2 route channels (red triangle = our car)"); fig.tight_layout()
fig.savefig(out / f"map_channels_{index:02d}.png", dpi=70); plt.close(fig)

# Bird's-eye overview with the human's real path
rc = GEOM.ego_to_pixel(gt_xy[valid])
fig, ax = plt.subplots(figsize=(6, 9))
ax.imshow(bev_background(m, r)); ax.plot(rc[:, 1], rc[:, 0], color="purple", lw=3, label="human (answer key)")
ax.plot(GEOM.ego_anchor_col, GEOM.ego_anchor_row, "r^", ms=10, label="our car"); ax.legend(); ax.axis("off")
fig.tight_layout(); fig.savefig(out / f"bev_overview_{index:02d}.png", dpi=90); plt.close(fig)

# Egomotion: past (input) and future controls (answer key)
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
for i, (ax, label) in enumerate(zip(axes, ["speed [m/s]", "accel [m/s²]", "yaw rate [rad/s]", "curvature [1/m]"])):
    ax.plot(np.arange(-63, 1) * 0.1, ego_hist[:, i], "b", label="history (input)")
    if i in (1, 3):
        ax.plot(np.arange(1, 65) * 0.1, ego_future[:, 0 if i == 1 else 1], "m--", label="future (answer key)")
    ax.axvline(0, color="k", lw=0.8); ax.set_ylabel(label); ax.grid(alpha=0.3)
axes[1].legend(); axes[-1].set_xlabel("time [s] (0 = now)"); fig.tight_layout()
fig.savefig(out / f"egomotion_{index:02d}.png", dpi=80); plt.close(fig)

print(f"\nSaved pictures to {out}")
