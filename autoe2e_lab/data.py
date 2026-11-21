"""Read scenes stored in the AutoE2E packed format (WebDataset .tar files).

One "part" = one driving scene:
  shards/<part>-train-000000.tar   samples: cameras, map, egomotion, answer key, ...
  pool/<part>/frame_pool.tar       past camera frames used for the 3.5 s history
"""
import json
import tarfile
from pathlib import Path

from . import upstream  # noqa: F401  (makes AutoE2E importable)
from data_parsing.pre_extracted import _decode_sample  # noqa: E402


class PackedScene:
    def __init__(self, dataset_dir: Path, part: str):
        self.part = part
        self.shard = tarfile.open(dataset_dir / f"shards/{part}-train-000000.tar")
        self.pool = tarfile.open(dataset_dir / f"pool/{part}/frame_pool.tar")
        self.keys = sorted({m.name.split(".")[0] for m in self.shard.getmembers()})

    def __len__(self) -> int:
        return len(self.keys)

    def raw(self, index: int) -> dict:
        """All files of one sample as {"cam_0.jpg": bytes, ...}."""
        key = self.keys[index]
        sample = {
            m.name[len(key) + 1:]: self.shard.extractfile(m).read()
            for m in self.shard.getmembers()
            if m.name.startswith(key + ".")
        }
        sample["__key__"] = key
        return sample

    def decoded(self, index: int) -> dict:
        """The authors' decoder: raw files -> model-ready tensors."""
        def pool(frame_id: str) -> bytes:
            return self.pool.extractfile(f"{frame_id}.jpg").read()
        return _decode_sample(self.raw(index), pool, decode_future_frames=False)


def list_parts(dataset_dir: Path) -> list[dict]:
    """Every scene in the dataset manifest with its length and download size."""
    manifest = json.loads((dataset_dir / "shards/manifest.json").read_text())
    shard_bytes = {e["name"].split("-train-")[0]: e["byte_size"] for e in manifest["shard_entries"]}
    parts = []
    for p in manifest["partitions"]:
        pid = p["partition_id"]
        parts.append({
            "part": pid,
            "moments": p["sample_count"],
            "seconds": p["sample_count"] / manifest["hz"],
            "download_mb": (shard_bytes.get(pid, 0) + p["pool"]["byte_size"]) / 1e6,
        })
    return parts
