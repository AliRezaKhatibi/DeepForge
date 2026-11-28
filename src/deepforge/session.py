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
            checkpoint.model, checkpoint.optimizer, monitor=monitor, refresh=checkpoint.refresh, **checkpoint.components)
        from .control import StopControl
        self.control = StopControl(self.root / "control.json")
        self.target = None
        self.lock = Lock()
        self.state_path = self.root / "session.json"
        self.manual_path = self.root / "manual-save.json"
        self.desktop_metadata = {}
        self.desktop_process = None
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
            self._desktop_status()
            return self.target

    def cancel_stop(self):
        with self.lock:
            self.target = None
            self._desktop_status()

    def poll_control(self):
        command = self.control.consume()
        if command:
            if command["action"] == "cancel":
                self.cancel_stop()
            else:
                self.request_stop(after_next=command.get("after_next", True))

    def request_checkpoint(self):
        """Queue an explicit save at the next completed epoch boundary."""
        if self.monitor.status != "running":
            raise RuntimeError("Session is not running")
        atomic_json(self.manual_path, {"requested": True})

    def _desktop_status(self, **extra):
        if (self.root / "desktop.json").is_file():
            atomic_json(self.root / "desktop-status.json", {"target_epoch": self.target, **extra})

    def open_desktop(self, *, theme="jade", interval=0.5, allow_resume=False, readiness=None):
        """Open the optional independent desktop monitor using this Python environment."""
        from .desktop_monitor import launch_desktop, model_metadata
        if self.desktop_process is not None and self.desktop_process.poll() is None:
            return self.desktop_process
        settings = dict(self.desktop_metadata)
        settings.setdefault("run_id", self.root.name)
        settings.update(model_metadata(self.recovery.model))
        settings["readiness"] = {**settings.get("readiness", {}), **(readiness or {})}
        settings["controls"] = True
        settings["allow_resume"] = bool(allow_resume)
        settings["config"] = self.config.get("setup", {})
        atomic_json(self.root / "desktop.json", settings)
        self._desktop_status()
        self.desktop_process = launch_desktop(self.monitor.path, theme=theme, interval=interval)
        return self.desktop_process

    def wait_for_resume_request(self, *, timeout=None, interval=.25):
        """Opt-in caller-side wait after pausing; caller must rebuild/restore and resume its loop."""
        import math
        import time
        if self.monitor.status != "paused":
            raise RuntimeError("Resume requests require a paused session")
        if not math.isfinite(interval) or interval <= 0 or (timeout is not None and (not math.isfinite(timeout) or timeout < 0)):
            raise ValueError("Invalid resume wait interval or timeout")
        settings = json.loads((self.root / "desktop.json").read_text(encoding="utf-8"))
        if not settings.get("allow_resume"):
            raise RuntimeError("Enable allow_resume when opening the desktop monitor")
        path = self.root / "resume-request.json"
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if path.exists():
                path.unlink()
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(interval if deadline is None else min(interval, max(0,deadline-time.monotonic())))

    def epoch_boundary(self, epoch, *, global_step):
        """Call after validation and scheduler updates; break loop if True."""
        self.poll_control()
        with self.lock:
            if self.target is None or epoch < self.target:
                pending = self.manual_path.with_name("manual-save.processing.json")
                try:
                    self.manual_path.replace(pending)
                except FileNotFoundError:
                    pass
                if pending.exists():
                    if (self.monitor.completed != epoch * self.monitor.batches or
                            "validation" not in self.monitor.epoch_records.get(epoch, {})):
                        raise RuntimeError("Manual save requires a completed epoch and validation")
                    manual = TorchCheckpoint(CheckpointStore(self.root, "manual", CheckpointPolicy()),
                        self.recovery.model, self.recovery.optimizer, monitor=self.monitor,
                        refresh=self.recovery.refresh, **self.recovery.components)
                    manual.save_epoch(epoch, global_step=global_step, metadata=self.metadata)
                    pending.unlink(missing_ok=True)
                    self._desktop_status(saved_epoch=epoch)
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
            self.manual_path.unlink(missing_ok=True)
            self.manual_path.with_name("manual-save.processing.json").unlink(missing_ok=True)
            self.target = None
            self._desktop_status(saved_epoch=epoch)
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
        self.monitor.started -= snap["elapsed_seconds"]
        self.monitor._append({"event": "resume", "epoch": result["epoch"]})
        self.monitor._publish(force=True)
        atomic_json(self.state_path, {"status": "running", **self.metadata})
        self._desktop_status()
        return result

    def recovery_panel(self):
        import ipywidgets as w
        from IPython.display import display
        saved = json.loads(self.state_path.read_text(encoding="utf-8"))
        epoch = saved.get("epoch", 0)
        path = self.recovery.store.latest()
        verified = path is not None and path.name == f"epoch-{epoch}.bin"
        label = w.Label(value=f"Saved epoch {epoch} / {self.monitor.epochs}; next {epoch+1}; verified: {verified}")
        resume = w.Button(description="Restore training state", button_style="success", disabled=saved["status"] != "paused" or not verified)
        def recover(_):
            try:
                result = self.restore()
                label.value = f"Restored. Start training at epoch {result['epoch']+1}"
                resume.disabled = True
            except Exception as exc:
                label.value = f"Recovery failed: {exc}"
        resume.on_click(recover)
        display(w.VBox([label,resume]))
        return resume,label

    def controls(self):
        """Optional ipywidgets controls; frontend must deliver callbacks."""
        import ipywidgets as w
        from IPython.display import display
        mode = w.Dropdown(options=[("After next epoch",True),("After current epoch",False)],value=True)
        stop = w.Button(description="Request safe stop", button_style="danger")
        cancel = w.Button(description="Cancel stop")
        label = w.Label(value="Running")
        def request(_):
            self.control.request(after_next=mode.value)
            label.value = "Stop request queued; awaiting training acknowledgement"
        def reset(_):
            self.control.cancel()
            label.value = "Cancellation queued"
        stop.on_click(request)
        cancel.on_click(reset)
        display(w.VBox([mode, w.HBox([stop, cancel]), label]))
        return stop, cancel, label
