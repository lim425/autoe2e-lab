"""Load a dataset config file (configs/*.yaml) and resolve its paths."""
from dataclasses import dataclass
from pathlib import Path

import yaml

from .upstream import LAB_ROOT

DEFAULT_CONFIG = LAB_ROOT / "configs/kitscenes_val.yaml"


@dataclass(frozen=True)
class Config:
    name: str
    data_root: Path
    checkpoint: Path
    dataset_dir: Path
    default_part: str
    training_stage: str
    ground_z_m: float
    front_camera_index: int
    attribution: str

    @property
    def output_dir(self) -> Path:
        return LAB_ROOT / "outputs" / self.name


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    data_root = (LAB_ROOT / raw["data_root"]).resolve()
    return Config(
        name=raw["name"],
        data_root=data_root,
        checkpoint=data_root / raw["checkpoint"],
        dataset_dir=data_root / raw["dataset_dir"],
        default_part=raw["default_part"],
        training_stage=raw["training_stage"],
        ground_z_m=float(raw["ground_z_m"]),
        front_camera_index=int(raw["front_camera_index"]),
        attribution=raw["attribution"],
    )
