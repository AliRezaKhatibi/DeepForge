"""Explicit checkpoint configuration for training scripts and frontends."""
import json
from pathlib import Path
from .checkpoint import CheckpointPolicy, CheckpointStore


def load_checkpoint_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            data = yaml.safe_load(stream)
        else:
            data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("configuration must be a mapping")
    if set(data) - {"root", "run_id", "policy"}:
        raise ValueError("unknown checkpoint configuration keys")
    return CheckpointStore(data["root"], data["run_id"],
                           CheckpointPolicy(**data.get("policy", {})))
