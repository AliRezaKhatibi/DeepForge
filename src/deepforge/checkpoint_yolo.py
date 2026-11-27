"""Opt-in capture of native Ultralytics training checkpoints."""
from pathlib import Path
from .checkpoint import CheckpointStore


def attach_yolo_checkpoint(model, store: CheckpointStore):
    """Register an on_model_save callback; native resume remains Ultralytics-owned."""
    def capture(trainer):
        epoch = int(trainer.epoch) + 1
        if not store.policy.enabled:
            return
        last = Path(trainer.last)
        if not last.is_file():
            raise FileNotFoundError("Ultralytics did not publish last.pt")
        best = trainer.best_fitness == trainer.fitness
        store.save(epoch, last.read_bytes(), best=best)
    model.add_callback("on_model_save", capture)
    return capture
