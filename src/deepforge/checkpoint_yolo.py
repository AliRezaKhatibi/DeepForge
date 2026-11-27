"""Opt-in capture of native Ultralytics training checkpoints."""
from pathlib import Path
from .checkpoint import CheckpointStore


def attach_yolo_checkpoint(model, store: CheckpointStore, *, monitor=None, refresh=None):
    """Register an on_model_save callback; native resume remains Ultralytics-owned."""
    def capture(trainer):
        epoch = int(trainer.epoch) + 1
        from .checkpoint_reporting import report_save
        if not store.policy.enabled:
            return report_save(store, epoch, False, lambda: None, monitor, refresh)
        best = trainer.best_fitness == trainer.fitness
        def save():
            last = Path(trainer.last)
            if not last.is_file():
                raise FileNotFoundError("Ultralytics did not publish last.pt")
            return store.save(epoch, last.read_bytes(), best=best)
        report_save(store, epoch, best, save, monitor, refresh)
    model.add_callback("on_model_save", capture)
    return capture
