"""Opt-in PyTorch checkpoint adapter for user-owned training loops."""
import io
import random

from .checkpoint import CheckpointStore


class TorchCheckpoint:
    def __init__(self, store: CheckpointStore, model, optimizer, *,
                 scheduler=None, scaler=None, sampler=None, monitor=None, refresh=None):
        self.monitor = monitor
        self.refresh = refresh
        self.store = store
        self.model = model
        self.optimizer = optimizer
        self.components = {"scheduler": scheduler, "scaler": scaler, "sampler": sampler}

    def save_epoch(self, epoch, *, global_step, metadata, best=False):
        from .checkpoint_reporting import report_save
        return report_save(self.store, epoch, best, lambda progress: self._save(epoch, global_step, metadata, best, progress), self.monitor, self.refresh, byte_progress=True)

    def _save(self, epoch, global_step, metadata, best, progress):
        import torch
        state = {
            "schema": 1, "epoch": epoch, "global_step": global_step,
            "metadata": metadata, "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "components": {k: v.state_dict() for k, v in self.components.items()
                           if v is not None},
        }
        try:
            import numpy as np
            name, keys, pos, has_gauss, cached = np.random.get_state()
            state["numpy_rng"] = (name, keys.tolist(), pos, has_gauss, cached)
        except ImportError:
            pass
        if torch.cuda.is_initialized():
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        stream = io.BytesIO()
        torch.save(state, stream)
        return self.store.save(epoch, stream.getvalue(), best=best, progress=progress)

    def restore_latest(self, *, expected_metadata):
        import torch
        path = self.store.latest()
        if path is None:
            raise FileNotFoundError("no verified checkpoint")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["schema"] != 1 or state["metadata"] != expected_metadata:
            raise ValueError("checkpoint metadata mismatch; review configuration before resume")
        if set(state["components"]) != {k for k, v in self.components.items() if v is not None}:
            raise ValueError("checkpoint component mismatch")
        if "cuda_rng" in state and len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count mismatch")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        for key, value in state["components"].items():
            self.components[key].load_state_dict(value)
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if "numpy_rng" in state:
            import numpy as np
            name, keys, pos, has_gauss, cached = state["numpy_rng"]
            np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
        if "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return {"epoch": state["epoch"], "global_step": state["global_step"]}
