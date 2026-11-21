"""Step 6, part 1: what does the model's own top-down picture look like?

Before the map and route are mixed in, the model turns the 6 cameras (+ 3.5 s of camera
history) into its own top-down picture ("image BEV"): a 300 x 200 grid, 0.6 m per cell,
with 256 numbers in every cell. Nobody can look at 256 numbers per cell, so we use PCA to
squeeze them into the 3 most important ones and show those as red / green / blue.
Cells the model considers similar get similar colours.

We show it for the real cameras and for black cameras (same colour scale), next to the
HD map of the same area for reference.

    python scripts/07_look_inside.py                         # moments 0 and 100
    python scripts/07_look_inside.py --part part-xxxx --moments 50

Note: the model also has a small "BEV segmentation" helper head meant to label this
picture (road, vehicle, ...). In the released checkpoint it was never trained (its
training loss is 0.0 and its layers are still at their starting values), so its output is
shown once for completeness but is NOT meaningful.
"""
import argparse
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from autoe2e_lab.config import load_config  # noqa: E402
from autoe2e_lab.data import PackedScene  # noqa: E402
from autoe2e_lab.model import load_model, predict_with_internals, to_batch  # noqa: E402
from autoe2e_lab.render import GEOM, bev_background  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--part", default="part-d646a7f28b51e27f")
parser.add_argument("--moments", type=int, nargs="+", default=[0, 100])
args = parser.parse_args()

cfg = load_config()
scene = PackedScene(cfg.dataset_dir, args.part)
model = load_model(cfg)
out_dir = cfg.output_dir / "look_inside"
out_dir.mkdir(parents=True, exist_ok=True)

BLACK = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(3, 1, 1)
# Image-BEV grid: same area as the map (x -60..120 m forward, y -60..60 m), 0.6 m per cell.
BEV_EGO_ROW, BEV_EGO_COL = 120 / 0.6 - 0.5, 60 / 0.6 - 0.5


def black_cameras(sample: dict) -> dict:
    s = dict(sample)
    for key in ("visual_tiles", "front_camera_tile", "front_camera_fpn_tile", "camera_history_tiles"):
        if key in s:
            s[key] = BLACK.expand_as(s[key]).clone()
    return s


def pca_basis(bev: torch.Tensor):
    """Top-3 directions of variation across all cells of a [C, H, W] picture."""
    x = bev.flatten(1).T.double()                  # [cells, 256]
    mean = x.mean(0)
    _, _, v = torch.linalg.svd(x - mean, full_matrices=False)
    proj = (x - mean) @ v[:3].T
    lo, hi = torch.quantile(proj, 0.02, dim=0), torch.quantile(proj, 0.98, dim=0)
    return mean, v[:3], lo, hi


def to_rgb(bev: torch.Tensor, basis) -> np.ndarray:
    mean, v, lo, hi = basis
    proj = (bev.flatten(1).T.double() - mean) @ v.T
    rgb = ((proj - lo) / (hi - lo)).clamp(0, 1)
    return rgb.T.reshape(3, *bev.shape[1:]).permute(1, 2, 0).numpy()


for m in args.moments:
    sample, raw = scene.decoded(m), scene.raw(m)
    real = predict_with_internals(model, to_batch(sample), cfg)
    dark = predict_with_internals(model, to_batch(black_cameras(sample)), cfg)
    basis = pca_basis(real["image_bev"])          # one colour scale for both pictures
    change = float((real["image_bev"] - dark["image_bev"]).norm() / real["image_bev"].norm())
    print(f"moment {m:3d}: image BEV {tuple(real['image_bev'].shape)}; "
          f"black cameras change it by {change:.0%}")

    fig, axes = plt.subplots(1, 5, figsize=(26, 8), gridspec_kw={"width_ratios": [1.5, 1, 1, 1, 1]})
    axes[0].imshow(Image.open(io.BytesIO(raw["cam_0.jpg"])))
    axes[0].set_title(f"front camera, t = {m / 10:.1f} s")
    for ax, picture, title in (
        (axes[1], to_rgb(real["image_bev"], basis), "model's own top-down picture\n(cameras only, PCA colours)"),
        (axes[2], to_rgb(dark["image_bev"], basis), f"same, BLACK cameras\n(changed by {change:.0%})"),
    ):
        ax.imshow(picture)
        ax.plot(BEV_EGO_COL, BEV_EGO_ROW, "w^", ms=10, mec="k")
        ax.set_title(title)
    axes[3].imshow(bev_background(sample["map_context"].numpy(), sample["route_mask"].numpy()))
    axes[3].plot(GEOM.ego_anchor_col, GEOM.ego_anchor_row, "r^", ms=10)
    axes[3].set_title("HD map of the same area\n(reference, NOT used for the left pictures)")
    axes[4].imshow(real["bev_probabilities"][0].numpy(), vmin=0, vmax=1, cmap="viridis")
    axes[4].plot(GEOM.ego_anchor_col, GEOM.ego_anchor_row, "r^", ms=10)
    axes[4].set_title("helper head 'drivable area' (0..1)\nUNTRAINED in this checkpoint: ignore")
    for ax in axes:
        ax.axis("off")
    fig.text(0.01, 0.01, cfg.attribution, fontsize=9, color="gray")
    fig.tight_layout()
    path = out_dir / f"{args.part}_{m:04d}.png"
    fig.savefig(path, dpi=70)
    plt.close(fig)
    print(f"  saved {path}")
