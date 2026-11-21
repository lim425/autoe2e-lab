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


def load_model(cfg: Config, device: str = "cuda") -> AutoE2E:
    """Build the v1.0 architecture and copy in the trained weights (strict match)."""
    payload = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    config = payload["config"]
    model = AutoE2E(
        backbone=config["backbone"],
        embed_dim=config["embed_dim"],
        is_pretrained=False,
        **reactive_model_kwargs(
            ReactiveTrainingStage(config["training_stage"]),
            num_views=config["num_views"],
        ),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def to_batch(sample: dict, device: str = "cuda") -> dict:
    """Add a batch dimension of 1 to every tensor and move it to the GPU."""
    return {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v for k, v in sample.items()}


@torch.no_grad()
def predict(model: AutoE2E, batch: dict, cfg: Config, device: str = "cuda") -> dict:
    """One forward pass -> 64 x (acceleration, curvature) and the integrated XY path."""
    stage = ReactiveTrainingStage(cfg.training_stage)
    # 1. Dataset-specific input rules (e.g. KITScenes hides the newest acceleration).
    batch = apply_reactive_egomotion_policy(batch, reactive_training_policy(stage))
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
