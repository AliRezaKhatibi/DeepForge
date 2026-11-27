"""Optional notebook widgets; no training starts until the caller starts it."""
import math
from .checkpoint import atomic_json

def setup_panel(path):
    import ipywidgets as w
    from IPython.display import display
    from pathlib import Path
    path = Path(path)
    metrics = w.SelectMultiple(description="Epoch metrics", options=["loss","val_loss","accuracy","val_accuracy","precision","recall","f1","lr","gradient_norm","mAP50","mAP50-95"], value=("loss","val_loss","accuracy","val_accuracy"))
    overall = w.SelectMultiple(description="Overall", options=["Status","Epoch","Batch","Elapsed (s)","ETA (s)","Batches/s"], value=("Status","Epoch","Batch","Elapsed (s)","ETA (s)","Batches/s"))
    theme = w.Dropdown(description="Theme", options=["jade","dark","light"])
    cards = w.Dropdown(description="Epoch cards", options=[10,20,50,None], value=20)
    refresh = w.Dropdown(description="Refresh (s)", options=[0.5,1,2,5], value=1)
    retention = w.Dropdown(description="Keep latest", options=[2,5,10,None], value=2)
    periodic = w.Checkbox(description="Periodic checkpoints", value=True)
    audio = w.Checkbox(description="Critical audio alerts", value=False)
    diagnostics = w.Checkbox(description="Visual warnings", value=True)
    resources = w.Checkbox(description="Hardware monitoring", value=True)
    sensitivity = w.Dropdown(description="Sensitivity", options=[("Sensitive",2),("Balanced",3),("Conservative",5)], value=3)
    destination = w.Text(description="Storage path", value=str(path.parent))
    save = w.Button(description="Save configuration", button_style="success")
    result = w.Label()
    def commit(_):
        config = dict(epoch_metrics=list(metrics.value), overall_fields=list(overall.value),
                      theme=theme.value,max_epochs=cards.value,interval=refresh.value,
                      keep_last=retention.value,periodic=periodic.value,audio=audio.value,
                      show_diagnostics=diagnostics.value,show_resources=resources.value,
                      alert_patience=sensitivity.value,storage=destination.value)
        path.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(path,config)
        result.value="Configuration saved"
    save.on_click(commit)
    display(w.VBox([w.HTML("<h2 style='color:#24765c'>DeepForge Studio · Training setup</h2>"),overall,metrics,theme,cards,refresh,diagnostics,resources,sensitivity,audio,periodic,retention,destination,save,result]))
    return save

class AudioAlerts:
    def __init__(self, enabled=False):
        self.enabled=enabled
        self.last=None
    def notify(self, epoch, health):
        if not self.enabled or health not in ("Severe overfitting warning", "Nonfinite metric", "Gradient explosion") or self.last == (epoch, health):
            return False
        from IPython.display import Audio, display
        rate=8000
        frequency = {"Severe overfitting warning":660,"Nonfinite metric":880,"Gradient explosion":440}[health]
        samples=[0.25*math.sin(2*math.pi*frequency*i/rate) if (i//1600)%2==0 else 0 for i in range(8000)]
        display(Audio(samples,rate=rate,autoplay=True))
        self.last=(epoch, health)
        return True


def build_training_session(config_path, *, run_id, epochs, batches_per_epoch, model, optimizer,
                           experiment_metadata, scheduler=None, scaler=None, sampler=None):
    """Apply saved setup to a PyTorch session; caller owns training loop."""
    import json
    from pathlib import Path
    from .monitor import TrainingMonitor
    from .notebook import NotebookMonitor
    from .checkpoint import CheckpointPolicy, CheckpointStore
    from .checkpoint_torch import TorchCheckpoint
    from .session import TrainingSession
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    store = CheckpointStore(config["storage"], run_id, CheckpointPolicy(
        keep_last=config["keep_last"], enabled=config["periodic"]))
    monitor = TrainingMonitor(store.root/"live.json", epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        resource_interval=2 if config["show_resources"] else None)
    keys = ("epoch_metrics","overall_fields","theme","max_epochs","interval",
            "audio","show_diagnostics","show_resources","alert_patience")
    view = NotebookMonitor(monitor.path, **{k:config[k] for k in keys})
    checkpoint = TorchCheckpoint(store, model, optimizer, scheduler=scheduler,
        scaler=scaler, sampler=sampler, monitor=monitor,
        refresh=lambda: view.refresh(force=True))
    session = TrainingSession(monitor, checkpoint,
        {"setup":config,"experiment":experiment_metadata})
    return monitor, view, checkpoint, session
