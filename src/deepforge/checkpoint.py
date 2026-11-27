"""Checkpoint storage primitives. Payload serialization belongs to adapters."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointPolicy:
    keep_last: int = 2
    max_bytes: int | None = None
    enabled: bool = True
    save_every_epochs: int = 1

    def __post_init__(self):
        if self.keep_last < 2 or self.save_every_epochs < 1:
            raise ValueError("retain at least two checkpoints; interval must be positive")
        if self.max_bytes is not None and self.max_bytes < 1:
            raise ValueError("max_bytes must be positive")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def verify_payload(path: Path, manifest: dict) -> bool:
    """Verify accidental corruption, not authenticity of untrusted files."""
    try:
        return (path.is_file() and not path.is_symlink()
                and path.stat().st_size == manifest["size"]
                and sha256_file(path) == manifest["sha256"])
    except (OSError, KeyError, TypeError):
        return False


def require_capacity(root: Path, incoming_bytes: int,
                     policy: CheckpointPolicy, used_bytes: int) -> None:
    if incoming_bytes < 0 or used_bytes < 0:
        raise ValueError("byte counts must be nonnegative")
    if shutil.disk_usage(root).free < incoming_bytes:
        raise OSError("insufficient free storage for safe checkpoint staging")
    if policy.max_bytes is not None and used_bytes + incoming_bytes > policy.max_bytes:
        raise OSError("checkpoint quota exceeded; existing checkpoints preserved")
