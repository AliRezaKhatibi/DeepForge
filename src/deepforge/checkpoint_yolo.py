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
        def save(progress):
            last = Path(trainer.last)
            if not last.is_file():
                raise FileNotFoundError("Ultralytics did not publish last.pt")
            return store.save(epoch, last.read_bytes(), best=best, progress=progress)
        report_save(store, epoch, best, save, monitor, refresh, byte_progress=True)
    model.add_callback("on_model_save", capture)
    return capture


def attach_yolo_safe_stop(model, store, control):
    """Opt-in native checkpoint capture before requesting trainer stop.

    Requires Ultralytics save=True and a version honoring trainer.stop.
    Resume uses the native checkpoint, not the finalized/stripped weights.
    """
    from .checkpoint import CheckpointPolicy, verify_payload, atomic_json
    import json
    recovery = CheckpointStore(store.root, "recovery", CheckpointPolicy())
    state = {"target":None}
    def poll(trainer):
        command = control.consume()
        if command:
            state["target"] = None if command["action"] == "cancel" else min(
                int(trainer.epochs), int(trainer.epoch)+1+int(command.get("after_next",True)))
    def capture(trainer):
        poll(trainer)
        epoch = int(trainer.epoch)+1
        if state["target"] is None or epoch < state["target"]:
            return
        path = recovery.save(epoch, Path(trainer.last).read_bytes())
        if not verify_payload(path,json.loads(path.with_suffix(".json").read_text())):
            raise OSError("Native recovery checkpoint verification failed")
        atomic_json(recovery.root/"session.json", {"status":"paused","epoch":epoch,
                    "epochs":int(trainer.epochs),"checkpoint":path.name})
        trainer.stop = True
        state["target"] = None
    model.add_callback("on_train_batch_end",poll)
    model.add_callback("on_model_save",capture)
    return state
