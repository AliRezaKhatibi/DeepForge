import json
import pytest
from deepforge.monitor import TrainingMonitor
from deepforge.ultralytics import attach_monitor


def test_progress_and_artifacts(tmp_path):
    path = tmp_path / "live.json"
    m = TrainingMonitor(path, epochs=2, batches_per_epoch=2, publish_interval=0, history_limit=2)
    for e in (1,2):
        for b in (1,2):
            m.batch(epoch=e,batch=b,samples=3,metrics={"loss": 1/b})
    output = tmp_path / "model.pt"
    output.write_bytes(b"demo")
    m.artifact(output)
    m.finish()
    s = json.loads(path.read_text())
    assert s["status"] == "completed"
    assert s["progress"] == 1
    assert s["samples"] == 12
    assert len(s["history"]) == 2
    assert s["artifacts"] == [{"name":"model.pt","exists":True,"bytes":4}]
    assert s["eta_seconds"] == 0


def test_invalid_order(tmp_path):
    m = TrainingMonitor(tmp_path/"live.json", epochs=1,batches_per_epoch=2)
    with pytest.raises(ValueError):
        m.batch(epoch=1,batch=2)
    assert m.completed == 0


def test_nonfinite_and_stopped(tmp_path):
    m = TrainingMonitor(tmp_path/"live.json",epochs=2,batches_per_epoch=1,publish_interval=0)
    m.batch(epoch=1,batch=1,metrics={"loss":float("nan")})
    assert m.snapshot()["metrics"]["loss"] is None
    m.finish()
    assert m.snapshot()["status"] == "stopped"
    with pytest.raises(RuntimeError):
        m.batch(epoch=2,batch=1)


def test_adapter_protocol(tmp_path):
    class Model:
        def __init__(self): self.callbacks = {}
        def add_callback(self,event,callback): self.callbacks[event] = callback
    class Trainer:
        epochs=1
        epoch=0
        train_loader=[1,2]
    model, trainer = Model(), Trainer()
    state = attach_monitor(model,tmp_path/"live.json",publish_interval=0)
    for event in ("on_train_start","on_train_epoch_start","on_train_batch_end","on_train_batch_end","on_train_end"):
        model.callbacks[event](trainer)
    assert state["monitor"].snapshot()["status"] == "completed"
