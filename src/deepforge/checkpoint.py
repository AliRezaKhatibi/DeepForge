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


class CheckpointStore:
    """Single-writer local store; immutable payloads and manifest-last publication."""

    def __init__(self, root, run_id: str, policy: CheckpointPolicy | None = None):
        import re
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
            raise ValueError("invalid run identifier")
        self.root = Path(root) / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = policy or CheckpointPolicy()

    def _records(self):
        records = []
        for manifest in self.root.glob("epoch-*.json"):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                epoch = data["epoch"]
                if type(epoch) is not int or epoch < 0 or manifest.name != f"epoch-{epoch}.json":
                    continue
                payload = self.root / f"epoch-{epoch}.bin"
                if verify_payload(payload, data):
                    records.append((epoch, payload, manifest, data))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(records, key=lambda item: item[0], reverse=True)

    def latest(self):
        records = self._records()
        return records[0][1] if records else None

    def save(self, epoch: int, payload: bytes, *, best: bool = False):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("invalid epoch")
        if not self.policy.enabled or (epoch % self.policy.save_every_epochs and not best):
            return None
        lock = self.root / ".writer.lock"
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.close(fd)
            target = self.root / f"epoch-{epoch}.bin"
            manifest = self.root / f"epoch-{epoch}.json"
            if target.exists() or manifest.exists():
                raise FileExistsError("epoch already exists")
            used = sum(p.stat().st_size for p in self.root.glob("epoch-*.bin"))
            require_capacity(self.root, len(payload), self.policy, used)
            fd, name = tempfile.mkstemp(prefix=".pending-", dir=self.root)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, target)
                atomic_json(manifest, {"epoch": epoch, "size": len(payload),
                                      "sha256": hashlib.sha256(payload).hexdigest(),
                                      "best": bool(best)})
            finally:
                Path(name).unlink(missing_ok=True)
            records = self._records()
            protected = {item[0] for item in records[:self.policy.keep_last]}
            best_record = next((item for item in records if item[3].get("best")), None)
            if best_record:
                protected.add(best_record[0])
            for old_epoch, old_payload, old_manifest, _ in records:
                if old_epoch not in protected:
                    old_manifest.unlink()
                    old_payload.unlink()
            return target
        finally:
            lock.unlink(missing_ok=True)
