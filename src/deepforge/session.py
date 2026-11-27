"""Cooperative epoch-boundary control for user-owned training loops."""
import json
from pathlib import Path
from threading import Lock
from .checkpoint import CheckpointStore, CheckpointPolicy, atomic_json, verify_payload
from .checkpoint_torch import TorchCheckpoint


class TrainingSession:
    def __init__(self, monitor, checkpoint, config):
        self.monitor = monitor
        self.config = json.loads(json.dumps(config, allow_nan=False))
        self.root = checkpoint.store.root
        self.recovery = TorchCheckpoint(
            CheckpointStore(self.root, "recovery", CheckpointPolicy()),
            checkpoint.model, checkpoint.optimizer, **checkpoint.components)
        self.target = None
        self.lock = Lock()
        self.state_path = self.root / "session.json"
        self.metadata = {"config": self.config, "epochs": monitor.epochs,
                         "batches_per_epoch": monitor.batches}
        if not self.state_path.exists():
            atomic_json(self.state_path, {"status": "running", **self.metadata})

    def request_stop(self, *, after_next=True):
        with self.lock:
            if self.monitor.status != "running":
                raise RuntimeError("Session is not running")
            current = max(1, (self.monitor.completed - 1) // self.monitor.batches + 1)
            self.target = min(self.monitor.epochs, current + int(after_next))
            return self.target

    def cancel_stop(self):
        with self.lock:
            self.target = None

    def epoch_boundary(self, epoch, *, global_step):
        """Call after validation and scheduler updates; break loop if True."""
        with self.lock:
            if self.target is None or epoch < self.target:
                return False
            if self.monitor.completed != epoch * self.monitor.batches:
                raise RuntimeError("Not at requested epoch boundary")
            if "validation" not in self.monitor.epoch_records.get(epoch, {}):
                raise RuntimeError("Record validation before safe stop")
            path = self.recovery.save_epoch(epoch, global_step=global_step,
                                            metadata=self.metadata)
            if not verify_payload(path, json.loads(path.with_suffix(".json").read_text())):
                raise OSError("Recovery verification failed")
            atomic_json(self.state_path, {"status": "paused", **self.metadata,
                        "epoch": epoch, "global_step": global_step,
                        "snapshot": self.monitor.snapshot()})
            self.monitor.pause()
            self.target = None
            return True

    def restore(self):
        """Restore into a fresh, identically configured model and monitor."""
        saved = json.loads(self.state_path.read_text(encoding="utf-8"))
        if saved["status"] != "paused" or any(saved[k] != v for k,v in self.metadata.items()):
            raise ValueError("No matching paused session")
        if self.monitor.completed:
            raise RuntimeError("Use a fresh monitor")
        result = self.recovery.restore_latest(expected_metadata=self.metadata)
        if result["epoch"] != saved["epoch"]:
            raise ValueError("Recovery epoch does not match saved history")
        snap = saved["snapshot"]
        self.monitor.completed = snap["completed_batches"]
        self.monitor.samples = snap["samples"]
        self.monitor.metrics = snap["metrics"]
        self.monitor.history = snap["history"]
        self.monitor.epoch_history = snap["epoch_history"]
        self.monitor.epoch_records = {int(k): v for k,v in snap["epoch_records"].items()}
        self.monitor._publish(force=True)
        atomic_json(self.state_path, {"status": "running", **self.metadata})
        return result

    def controls(self):
        """Optional ipywidgets controls; frontend must deliver callbacks."""
        import ipywidgets as w
        from IPython.display import display
        stop = w.Button(description="Stop after next epoch", button_style="danger")
        cancel = w.Button(description="Cancel stop")
        label = w.Label(value="Running")
        def request(_):
            label.value = f"Stop requested after epoch {self.request_stop()}"
        def reset(_):
            self.cancel_stop()
            label.value = "Running"
        stop.on_click(request)
        cancel.on_click(reset)
        display(w.VBox([w.HBox([stop, cancel]), label]))
        return stop, cancel, label
