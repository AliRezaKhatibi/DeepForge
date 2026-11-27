import json
import pytest
from deepforge.monitor import TrainingMonitor
from deepforge.notebook import snapshot_html
from deepforge.ultralytics import attach_monitor


def test_epoch_metrics_and_resources(tmp_path):
    path = tmp_path / "live.json"
    m = TrainingMonitor(path, epochs=2, batches_per_epoch=1)
    with pytest.raises(ValueError):
        m.epoch_end(1, {})
    m.batch(epoch=1, batch=1, metrics={"box_loss": .3})
    m.epoch_end(1, {"mAP": .7, "bad": float("nan")})
    m.record_resources({"ram_percent": 25})
    data = json.loads(path.read_text())
    assert data["epoch_history"] == [{"epoch":1,"metrics":{"mAP":.7,"bad":None}}]
    assert data["resources"]["ram_percent"] == 25
    with pytest.raises(ValueError):
        m.epoch_end(1, {})


def test_notebook_escapes_metrics(tmp_path):
    path = tmp_path / "live.json"
    m = TrainingMonitor(path, epochs=1, batches_per_epoch=1)
    m.batch(epoch=1,batch=1,metrics={"<script>":1})
    m.epoch_end(1, {"precision": .9})
    output = snapshot_html(path)
    assert "&lt;script&gt;" in output
    assert "<script>" not in output
    assert "Validation: precision" in output


def test_adapter_epoch_callback(tmp_path):
    class Model:
        callbacks = {}
        def add_callback(self,event,callback):
            self.callbacks[event] = callback
    class Trainer:
        epochs = 1
        epoch = 0
        train_loader = [1]
        metrics = {"metrics/mAP50(B)": .8}
    model, trainer = Model(), Trainer()
    state = attach_monitor(model,tmp_path/"live.json")
    for event in ("on_train_start","on_train_epoch_start","on_train_batch_end","on_fit_epoch_end","on_train_end"):
        model.callbacks[event](trainer)
    assert state["monitor"].snapshot()["epoch_history"][0]["metrics"]["metrics/mAP50(B)"] == .8
