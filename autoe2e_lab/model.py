"""Load the AutoE2E checkpoint and run it the same way the authors evaluate it.

`predict()` mirrors `evaluate_reactive_xy()` in the AutoE2E repo
(Model/training/reactive_stage_runner.py) step by step.
"""
import torch

from . import upstream  # noqa: F401  (makes AutoE2E importable)
from model_components.auto_e2e import AutoE2E  # noqa: E402
from training.losses.control_rollout import integrate_controls_torch  # noqa: E402
from training.reactive_multitask import ReactiveTrainingStage, reactive_model_kwargs  # noqa: E402
from training.reactive_stage_runner import (  # noqa: E402
    apply_reactive_egomotion_policy,
    reactive_training_policy,
    resolve_reactive_batch_projection,
    resolve_reactive_camera_history,
    resolve_reactive_front_projection,
)

from .config import Config


def _build(payload: dict) -> AutoE2E:
    """The v1.0 architecture with RANDOM weights (nothing learned yet)."""
    config = payload["config"]
    return AutoE2E(
        backbone=config["backbone"],
        embed_dim=config["embed_dim"],
        is_pretrained=False,
        **reactive_model_kwargs(
            ReactiveTrainingStage(config["training_stage"]),
            num_views=config["num_views"],
        ),
    )


def load_model(cfg: Config, device: str = "cuda", trained: bool = True) -> AutoE2E:
    """Build the v1.0 architecture and copy in the trained weights (strict match).

    trained=False skips the weights: same structure, random numbers ("empty brain").
    """
    payload = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    model = _build(payload)
    if trained:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def to_batch(sample: dict, device: str = "cuda") -> dict:
    """Add a batch dimension of 1 to every tensor and move it to the GPU."""
    return {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v for k, v in sample.items()}


def apply_input_rules(batch: dict, cfg: Config) -> dict:
    """Dataset-specific input rules, applied before EVERY model call.

    For KITScenes this hides the newest acceleration value in the motion history,
    exactly as the authors do in training and evaluation.
    """
    stage = ReactiveTrainingStage(cfg.training_stage)
    return apply_reactive_egomotion_policy(batch, reactive_training_policy(stage))


@torch.no_grad()
def predict(model: AutoE2E, batch: dict, cfg: Config, device: str = "cuda") -> dict:
    """One forward pass -> 64 x (acceleration, curvature) and the integrated XY path."""
    # 1. Dataset-specific input rules (e.g. KITScenes hides the newest acceleration).
    batch = apply_input_rules(batch, cfg)
    # 2. Camera geometry: wrap the 3x4 calibration matrices in projection operators.
    dev = torch.device(device)
    projection, geometry_type = resolve_reactive_batch_projection(batch, None, "pseudo", device=dev)
    front_projection = resolve_reactive_front_projection(batch, geometry_type, device=dev, required=True)
    history_tiles, history_projections = resolve_reactive_camera_history(
        batch, geometry_type, device=dev, required=True
    )
    # 3. The forward pass. The answer key (trajectory_xy_m) is never passed in.
    controls = model(
        batch["visual_tiles"],
        batch["map_context"],
        batch["visual_history"],
        batch["egomotion_history"],
        route_mask=batch["route_mask"],
        map_valid=batch["map_valid"],
        route_valid=batch["route_valid"],
        projection=projection,
        geometry_type=geometry_type,
        camera_history_tiles=history_tiles,
        history_projections=history_projections,
        front_camera_tile=batch.get("front_camera_tile"),
        front_projection=front_projection,
        mode="infer",
        compute_bev_segmentation=False,
        compute_route_reconstruction=False,
    )
    if isinstance(controls, tuple):
        controls = controls[0]
    controls = controls.reshape(1, 64, 2)
    # 4. Integrate controls into an XY path, starting at the current speed v0.
    pred_xy, _, _ = integrate_controls_torch(controls, batch["initial_speed_mps"])
    return {"controls": controls[0].cpu(), "pred_xy": pred_xy[0].cpu()}


@torch.no_grad()
def predict_with_internals(model: AutoE2E, batch: dict, cfg: Config, device: str = "cuda") -> dict:
    """Like predict(), but also return what the model builds inside.

    image_bev:        the model's own top-down picture, built ONLY from the cameras
                      (before the map and route are mixed in): [C, H, W] numbers per cell
    bev_probabilities: the built-in helper head's reading of that picture: for each of the
                      8 categories (BEV_CLASSES), the chance (0..1) that a cell contains it
    """
    batch = apply_input_rules(batch, cfg)
    dev = torch.device(device)
    projection, geometry_type = resolve_reactive_batch_projection(batch, None, "pseudo", device=dev)
    front_projection = resolve_reactive_front_projection(batch, geometry_type, device=dev, required=True)
    history_tiles, history_projections = resolve_reactive_camera_history(
        batch, geometry_type, device=dev, required=True
    )
    controls, aux = model(
        batch["visual_tiles"], batch["map_context"], batch["visual_history"], batch["egomotion_history"],
        route_mask=batch["route_mask"], map_valid=batch["map_valid"], route_valid=batch["route_valid"],
        projection=projection, geometry_type=geometry_type,
        camera_history_tiles=history_tiles, history_projections=history_projections,
        front_camera_tile=batch.get("front_camera_tile"), front_projection=front_projection,
        mode="infer", return_auxiliary=True, return_image_bev=True,
        compute_bev_segmentation=True, compute_route_reconstruction=False,
    )
    out = _to_path(controls, batch["initial_speed_mps"])
    out["image_bev"] = aux["_evaluation_image_bev"][0].float().cpu()
    out["bev_probabilities"] = aux["bev_segmentation_logits"][0].float().sigmoid().cpu()
    return out


def _to_path(controls: torch.Tensor, initial_speed: torch.Tensor) -> dict:
    controls = controls.reshape(1, 64, 2)
    pred_xy, _, _ = integrate_controls_torch(controls, initial_speed)
    return {"controls": controls[0].cpu(), "pred_xy": pred_xy[0].cpu()}


@torch.no_grad()
def predict_route_ablation(model: AutoE2E, batch: dict, cfg: Config, device: str = "cuda") -> dict:
    """Same moment, two answers: WITH the route and WITHOUT it.

    Pass 1 runs the full model and also returns the camera top-down picture (image BEV).
    Pass 2 reuses that picture (the cameras did not change) and only switches the route
    off with the model's own validity gate (route_valid=False multiplies the route by 0).
    This is the authors' "route-zero" counterfactual; it skips the slow camera part.
    """
    batch = apply_input_rules(batch, cfg)
    dev = torch.device(device)
    projection, geometry_type = resolve_reactive_batch_projection(batch, None, "pseudo", device=dev)
    front_projection = resolve_reactive_front_projection(batch, geometry_type, device=dev, required=True)
    history_tiles, history_projections = resolve_reactive_camera_history(
        batch, geometry_type, device=dev, required=True
    )
    common = dict(map_valid=batch["map_valid"], mode="infer",
                  compute_bev_segmentation=False, compute_route_reconstruction=False)
    inputs = (batch["visual_tiles"], batch["map_context"], batch["visual_history"], batch["egomotion_history"])

    # Pass 1: full model, route as given by the dataset; keep the camera top-down picture.
    with_route, aux = model(
        *inputs, route_mask=batch["route_mask"], route_valid=batch["route_valid"],
        projection=projection, geometry_type=geometry_type,
        camera_history_tiles=history_tiles, history_projections=history_projections,
        front_camera_tile=batch.get("front_camera_tile"), front_projection=front_projection,
        return_auxiliary=True, return_image_bev=True, **common,
    )
    image_bev = aux["_evaluation_image_bev"]

    # Pass 2: same camera picture, route switched off.
    route_off = torch.zeros_like(batch["route_valid"], dtype=torch.bool)
    without_route = model(
        *inputs, route_mask=batch["route_mask"], route_valid=route_off,
        precomputed_image_bev=image_bev, **common,
    )
    if isinstance(without_route, tuple):
        without_route = without_route[0]
    speed = batch["initial_speed_mps"]
    return {"with_route": _to_path(with_route, speed),
            "without_route": _to_path(without_route, speed),
            "image_bev": image_bev}
