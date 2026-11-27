"""Local, atomic training snapshots for an independent graphical viewer."""
import json
import math
import os
import tempfile
import time
from pathlib import Path
from .diagnostics import diagnose, class_distribution


def _count(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class TrainingMonitor:
    """Single writer per snapshot path. Record completed batches, not optimizer steps."""
    def __init__(self, path, *, epochs, batches_per_epoch, history_limit=500,
                 publish_interval=1.0, archive=None, diagnostics=None):
        self.path = Path(path)
        self.epochs = _count(epochs, "epochs", 1)
        self.batches = _count(batches_per_epoch, "batches_per_epoch", 1)
        self.limit = _count(history_limit, "history_limit", 1)
        if not math.isfinite(publish_interval) or publish_interval < 0:
            raise ValueError("publish_interval must be finite and nonnegative")
        self.interval = publish_interval
        self.started = time.monotonic()
        self.last_write = -math.inf
        self.completed = self.samples = 0
        self.history = []
        self.metrics = {}
        self.epoch_history = []
        self.resources = {}
        self.artifacts = []
        self.status = "running"
        self.diagnostic_options = dict(diagnostics or {})
        diagnose([], **self.diagnostic_options)
        self.batch_distribution = {}
        self.preview = None
        self.archive = Path(archive) if archive is not None else None
        if self.archive is not None:
            if self.archive.resolve() == self.path.resolve():
                raise ValueError("Archive and snapshot must be different files")
            with self.archive.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": "start", "epochs": self.epochs, "batches_per_epoch": self.batches}) + "\n")
        self._publish(force=True)

    def batch(self, *, epoch, batch, samples=0, metrics=None, labels=None):
        if self.status != "running":
            raise RuntimeError("Monitor already finished")
        _count(epoch, "epoch", 1)
        _count(batch, "batch", 1)
        _count(samples, "samples")
        if epoch > self.epochs or batch > self.batches:
            raise ValueError("Batch position outside configured schedule")
        step = (epoch - 1) * self.batches + batch
        if step != self.completed + 1:
            raise ValueError("Record each completed batch exactly once in order")
        values = {}
        for key, raw in (metrics or {}).items():
            value = float(raw.detach().item() if hasattr(raw, "detach") else raw)
            values[str(key)] = value if math.isfinite(value) else None
        distribution = class_distribution(labels) if labels is not None else {}
        self.batch_distribution = distribution
        self.completed = step
        self.samples += samples
        self.metrics = values
        self.history.append({"step": step, "metrics": values})
        self.history = self.history[-self.limit:]
        self._append({"event": "batch", "step": step, "samples": samples, "metrics": values, "labels": distribution})
        self._publish(force=step == self.epochs * self.batches)

    def epoch_end(self, epoch, metrics):
        """Record finite scalar validation metrics after a completed epoch."""
        _count(epoch, "epoch", 1)
        if self.status != "running" or epoch > self.completed // self.batches:
            raise ValueError("Epoch has not completed or monitor has stopped")
        if self.epoch_history and epoch <= self.epoch_history[-1]["epoch"]:
            raise ValueError("Epoch metrics must be recorded once in order")
        values = {}
        for key, raw in metrics.items():
            value = float(raw.detach().item() if hasattr(raw, "detach") else raw)
            values[str(key)] = value if math.isfinite(value) else None
        self.epoch_history.append({"epoch": epoch, "metrics": values})
        self._append({"event": "epoch", "epoch": epoch, "metrics": values})
        self._publish(force=True)

    def _append(self, event):
        if self.archive is not None:
            with self.archive.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")

    def record_preview(self, path):
        """Explicitly publish an image preview; do not archive image content."""
        from .preview import image_preview
        self.preview = image_preview(path)
        self._publish(force=True)

    def record_resources(self, resources):
        """Record caller-supplied, JSON-compatible resource measurements."""
        json.dumps(resources, allow_nan=False)
        self.resources = json.loads(json.dumps(resources))
        self._publish(force=True)

    def artifact(self, path):
        """Register an explicitly selected output; never inspect file contents."""
        p = Path(path)
        item = {"name": p.name, "exists": p.is_file(),
                "bytes": p.stat().st_size if p.is_file() else None}
        self.artifacts = [a for a in self.artifacts if a["name"] != item["name"]]
        self.artifacts = (self.artifacts + [item])[-100:]
        self._publish(force=True)

    def finish(self, *, failed=False):
        self.status = "failed" if failed else ("completed" if self.completed == self.epochs * self.batches else "stopped")
        self._append({"event": "finish", "status": self.status})
        self._publish(force=True)

    def snapshot(self):
        elapsed = max(0.0, time.monotonic() - self.started)
        total = self.epochs * self.batches
        rate = self.completed / elapsed if elapsed else 0.0
        return {"schema_version": 1, "status": self.status,
                "epochs": self.epochs, "batches_per_epoch": self.batches,
                "epoch": (self.completed - 1) // self.batches + 1 if self.completed else 0,
                "batch": (self.completed - 1) % self.batches + 1 if self.completed else 0,
                "completed_batches": self.completed, "total_batches": total,
                "progress": self.completed / total, "samples": self.samples,
                "elapsed_seconds": elapsed, "batches_per_second": rate,
                "samples_per_second": self.samples / elapsed if elapsed else 0.0,
                "eta_seconds": (total - self.completed) / rate if rate and self.status == "running" else (0 if self.status == "completed" else None),
                "metrics": dict(self.metrics), "history": list(self.history),
                "artifacts": list(self.artifacts), "epoch_history": list(self.epoch_history),
                "resources": dict(self.resources), "diagnostics": diagnose(self.epoch_history, **self.diagnostic_options),
                "batch_distribution": dict(self.batch_distribution), "preview_png": self.preview}

    def _publish(self, *, force=False):
        now = time.monotonic()
        if not force and now - self.last_write < self.interval:
            return
        payload = json.dumps(self.snapshot(), allow_nan=False)
        # Same-directory replacement prevents the reader seeing partial JSON.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".deepforge-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        self.last_write = now
