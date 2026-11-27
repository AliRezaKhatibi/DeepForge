"""Opt-in callback adapter for the Ultralytics trainer callback protocol."""
from .monitor import TrainingMonitor
from .resources import sample_resources


def attach_monitor(model, path, *, publish_interval=1.0, resources=False, archive=None, diagnostics=None):
    """Attach before model.train(). Fixed-length, fresh single-process runs only.

    Callback compatibility depends on Ultralytics version. This adapter observes
    completed batches, not optimizer steps; it never edits training settings.
    """
    state = {"monitor": None, "epoch": 0, "batch": 0}
    def start(trainer):
        if getattr(trainer, "start_epoch", 0) != 0:
            raise ValueError("Resumed training is not supported by this adapter")
        state["monitor"] = TrainingMonitor(path, epochs=int(trainer.epochs),
            batches_per_epoch=len(trainer.train_loader), publish_interval=publish_interval, archive=archive, diagnostics=diagnostics)
    def epoch_start(trainer):
        state["epoch"] = int(trainer.epoch) + 1
        state["batch"] = 0
    def batch_end(trainer):
        state["batch"] += 1
        metrics = {}
        loss = getattr(trainer, "loss_items", None)
        if loss is not None:
            values = loss.detach().flatten().tolist()
            names = getattr(trainer, "loss_names", ())
            metrics = {str(name): value for name,value in zip(names, values)}
            metrics["loss"] = sum(values)
        state["monitor"].batch(epoch=state["epoch"], batch=state["batch"], metrics=metrics)
    def epoch_end(trainer):
        metrics = {}
        for key, value in (getattr(trainer, "metrics", None) or {}).items():
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        monitor = state["monitor"]
        if monitor.epoch_history and monitor.epoch_history[-1]["epoch"] == state["epoch"]:
            return  # Final best-model evaluation can repeat the callback.
        for key, value in (getattr(trainer, "lr", None) or {}).items():
            metrics["lr/" + str(key)] = float(value)
        monitor.epoch_end(state["epoch"], metrics)
        if resources:
            state["monitor"].record_resources(sample_resources())

    def save(trainer):
        for key in ("last", "best"):
            path = getattr(trainer, key, None)
            if path is not None:
                state["monitor"].artifact(path)
    def end(trainer):
        state["monitor"].finish()
    for event, callback in (("on_train_start", start), ("on_train_epoch_start", epoch_start),
                            ("on_train_batch_end", batch_end), ("on_fit_epoch_end", epoch_end), ("on_model_save", save),
                            ("on_train_end", end)):
        model.add_callback(event, callback)
    return state
