import importlib
import json
import pytest
from deepforge.monitor import TrainingMonitor
from deepforge.notebook import snapshot_html
from deepforge.checkpoint import CheckpointStore, CheckpointPolicy
from deepforge.checkpoint_reporting import report_save

def test_entrypoint_import():
    importlib.import_module("deepforge.__main__")

def test_epoch_dashboard(tmp_path):
    path = tmp_path / "live.json"
    monitor = TrainingMonitor(path, epochs=3, batches_per_epoch=1)
    monitor.batch(epoch=1, batch=1, metrics={"loss": 0.123456, "accuracy": 90})
    monitor.record_stage(1, "validation", "running", completed=1, total=2)
    monitor.epoch_end(1, {"loss": 0.2, "accuracy": 89})
    monitor.record_stage(1, "validation", "completed", completed=2, total=2)
    store = CheckpointStore(tmp_path, "run")
    report_save(store, 1, False, lambda: store.save(1, b"payload"), monitor)
    html = snapshot_html(path)
    assert "Overall training progress" in html
    assert "Epoch 1 / 3" in html and "Epoch 2 / 3" in html
    assert "Checkpoint: completed" in html and "Verification: completed" in html
    assert "0.1235" in html and "Loss per epoch" in html
    assert "Accuracy per epoch" in html
    assert "Checkpoint: completed" not in snapshot_html(path, show_stages=False)
    assert "Loss per epoch" not in snapshot_html(path, show_charts=False)
    assert "#17212b" in snapshot_html(path, theme="dark")
    assert json.loads(path.read_text())["epoch_records"]["1"]["training"]["loss"] == 0.123456

def test_stage_skip_and_failure(tmp_path):
    monitor = TrainingMonitor(tmp_path / "live.json", epochs=1, batches_per_epoch=1)
    monitor.batch(epoch=1, batch=1)
    store = CheckpointStore(tmp_path, "disabled", CheckpointPolicy(enabled=False))
    assert report_save(store, 1, False, lambda: pytest.fail("unexpected save"), monitor) is None
    assert monitor.snapshot()["epoch_records"]["1"]["stages"]["checkpoint"]["status"] == "skipped"
    store = CheckpointStore(tmp_path, "enabled")
    def fail():
        raise OSError("disk unavailable")
    with pytest.raises(OSError):
        report_save(store, 1, False, fail, monitor)
    assert monitor.snapshot()["epoch_records"]["1"]["stages"]["checkpoint"]["status"] == "failed"
    with pytest.raises(ValueError):
        monitor.record_stage(1, "validation", "running", completed=3, total=2)


def test_long_run_card_window(tmp_path):
    path = tmp_path / "long.json"
    monitor = TrainingMonitor(path, epochs=510, batches_per_epoch=1, history_limit=10, publish_interval=3600)
    for epoch in range(1, 511):
        monitor.batch(epoch=epoch, batch=1, metrics={"loss": 1 / epoch})
    monitor.finish()
    html = snapshot_html(path, max_epochs=12)
    assert html.count("<article ") == 12
    assert "Epoch 510 / 510" in html
    assert "Epoch 1 / 510" not in html
    assert len(monitor.snapshot()["history"]) == 10
    assert len(monitor.snapshot()["epoch_records"]) == 510
    assert len(html) < 150000

def test_old_snapshot_compatibility(tmp_path):
    path = tmp_path / "legacy.json"
    monitor = TrainingMonitor(path, epochs=2, batches_per_epoch=2)
    data = monitor.snapshot()
    data.pop("epoch_records")
    path.write_text(json.dumps(data), encoding="utf-8")
    assert "Overall training progress" in snapshot_html(path)
