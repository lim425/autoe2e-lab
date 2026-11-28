"""Drawing: top-down map view and camera view with green (model) / purple (human) paths."""
import io
import json

import numpy as np
from PIL import Image, ImageDraw

from . import upstream  # noqa: F401  (makes AutoE2E importable)
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY as GEOM  # noqa: E402
from navigation.geometry import MapChannel, RouteChannel  # noqa: E402

GREEN = (60, 220, 120)
PURPLE = (150, 80, 230)
CAR_HALF_WIDTH_M = 0.9  # each path is drawn as a ~1.8 m wide band


# ---------------------------------------------------------------------------
# Camera maths: metres on the road -> pixels in the photo
# ---------------------------------------------------------------------------
def project_to_pixels(matrix_3x4: np.ndarray, xy_m: np.ndarray, ground_z: float):
    """Pinhole camera model.

    A 3D point (x forward, y left, z up) in metres becomes a pixel (u, v):
        [a, b, depth] = M @ [x, y, z, 1]
        u = a / depth,  v = b / depth
    Points with depth <= 0 are behind the camera and cannot be drawn.
    """
    n = len(xy_m)
    points = np.column_stack([xy_m, np.full(n, ground_z), np.ones(n)])
    a, b, depth = matrix_3x4 @ points.T
    in_front = depth > 1e-5
    safe = np.where(in_front, depth, 1.0)
    return np.column_stack([a / safe, b / safe]), in_front


def band_edges(xy_m: np.ndarray, half_width: float):
    """Left and right edges of a band around a path (offset along the path normal)."""
    tangent = np.gradient(xy_m, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True).clip(1e-6)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return xy_m + half_width * normal, xy_m - half_width * normal


def draw_band(img: Image.Image, matrix, xy_m, color, ground_z: float):
    """Draw a translucent band + centre line for one path onto an RGBA camera image."""
    w, h = img.size
    xy_m = np.vstack([[0.0, 0.0], xy_m])  # start the band at the car
    left, right = band_edges(xy_m, CAR_HALF_WIDTH_M)
    uv_l, ok_l = project_to_pixels(matrix, left, ground_z)
    uv_r, ok_r = project_to_pixels(matrix, right, ground_z)
    uv_c, ok_c = project_to_pixels(matrix, xy_m, ground_z)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for i in range(len(xy_m) - 1):
        if ok_l[i] and ok_l[i + 1] and ok_r[i] and ok_r[i + 1]:
            quad = [tuple(uv_l[i]), tuple(uv_l[i + 1]), tuple(uv_r[i + 1]), tuple(uv_r[i])]
            d.polygon(quad, fill=(*color, 90))
    line = [tuple(p) for p, ok in zip(uv_c, ok_c) if ok and -w < p[0] < 2 * w and -h < p[1] < 2 * h]
    if len(line) >= 2:
        d.line(line, fill=(0, 0, 0, 255), width=7, joint="curve")
        d.line(line, fill=(*color, 255), width=4, joint="curve")
    img.alpha_composite(overlay)


ORANGE = (255, 150, 30)


def camera_view(raw_sample: dict, pred_xy, gt_xy, valid, ground_z: float, cam: int = 0,
                extra=(), pred_color=GREEN) -> Image.Image:
    """Front camera (native resolution) with human (purple) and model (green by default) bands.

    extra: optional [(xy_m, rgb_color), ...] drawn first (underneath), e.g. a second prediction.
    """
    img = Image.open(io.BytesIO(raw_sample[f"cam_{cam}.jpg"])).convert("RGBA")
    calib = json.loads(raw_sample["calib.json"])
    matrix = np.asarray(calib["front_projection"]["matrix"][0], dtype=np.float64)
    for xy, color in extra:
        draw_band(img, matrix, xy, color, ground_z)
    draw_band(img, matrix, gt_xy[valid], PURPLE, ground_z)
    draw_band(img, matrix, pred_xy, pred_color, ground_z)
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Top-down map view
# ---------------------------------------------------------------------------
def bev_background(map_context: np.ndarray, route_mask: np.ndarray) -> np.ndarray:
    """RGB picture of the map raster: grey road, blue route, yellow crosswalks."""
    m, r = map_context, route_mask
    bev = np.ones((*m.shape[1:], 3))
    bev[m[MapChannel.DRIVABLE_AREA] > 0] = (0.80, 0.80, 0.80)
    bev[r[RouteChannel.SELECTED_CORRIDOR] > 0] = (0.75, 0.88, 1.0)
    bev[m[MapChannel.LANE_BOUNDARY] > 0] = (0.35, 0.35, 0.35)
    bev[m[MapChannel.CROSSWALK] > 0] = (1.0, 0.85, 0.4)
    return bev


def plot_bev(ax, sample: dict, pred_xy, gt_xy, valid, zoom: bool = True,
             pred_label: str = "AutoE2E", extra=(), pred_color=GREEN):
    """Draw the map + paths onto a matplotlib axis. extra: [(xy_m, rgb_color, label), ...]."""
    ax.imshow(bev_background(sample["map_context"].numpy(), sample["route_mask"].numpy()))
    gt_rc = GEOM.ego_to_pixel(gt_xy[valid])
    pred_rc = GEOM.ego_to_pixel(pred_xy)
    ax.plot(gt_rc[:, 1], gt_rc[:, 0], color=np.array(PURPLE) / 255, lw=5, label="human")
    for xy, color, label in extra:
        rc = GEOM.ego_to_pixel(xy)
        ax.plot(rc[:, 1], rc[:, 0], color=np.array(color) / 255, lw=2.5, label=label)
    ax.plot(pred_rc[:, 1], pred_rc[:, 0], color=np.array(pred_color) / 255, lw=2.5, label=pred_label)
    ax.plot(GEOM.ego_anchor_col, GEOM.ego_anchor_row, "r^", ms=10)
    if zoom:
        ax.set_xlim(GEOM.ego_anchor_col - 45, GEOM.ego_anchor_col + 45)
        ax.set_ylim(GEOM.ego_anchor_row + 20, GEOM.ego_anchor_row - 190)
    ax.axis("off")
