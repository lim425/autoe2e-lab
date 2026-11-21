"""Load the released checkpoint and check it fits the code exactly (strict=True).

    python scripts/01_load_checkpoint.py
"""
import torch

from autoe2e_lab.config import load_config
from autoe2e_lab.model import load_model

cfg = load_config()
payload = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
print("Checkpoint contains:", list(payload))
print("Training stage:", payload["config"]["training_stage"])

model = load_model(cfg)  # raises if any layer name or shape differs
print("load_state_dict: all keys matched (strict=True)")
print(f"Tensors in checkpoint: {len(payload['model_state_dict'])}")
print(f"Parameters in model:   {sum(p.numel() for p in model.parameters()):,}")
print(f"GPU memory used by weights: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
