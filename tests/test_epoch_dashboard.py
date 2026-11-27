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
    assert "Epoch 1 / 3" in html
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
    assert "498 earlier epochs retained" in html
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


def test_resource_sampling_and_controls(tmp_path, monkeypatch):
    import deepforge.resources as resources
    monkeypatch.setattr(resources, "sample_resources", lambda path: {"cpu_percent": 25, "gpu_utilization_percent": 75})
    path = tmp_path / "live.json"
    monitor = TrainingMonitor(path, epochs=1, batches_per_epoch=1, resource_interval=1)
    monitor.batch(epoch=1, batch=1)
    assert "GPU compute: 75.0%" in snapshot_html(path)
    assert "GPU compute" not in snapshot_html(path, show_resources=False)
    assert "<article " not in snapshot_html(path, show_epochs=False)
    assert "Overall training progress" not in snapshot_html(path, show_overall=False)

def test_checkpoint_bytes(tmp_path):
    store = CheckpointStore(tmp_path, "bytes")
    counts = []
    payload = b"x" * (5 * 1024 * 1024)
    path = store.save(1, payload, progress=lambda done, total: counts.append((done,total)))
    assert counts == [(4 * 1024 * 1024, len(payload)), (len(payload), len(payload))]
    assert path.read_bytes() == payload

def test_long_run_python_allocation(tmp_path):
    import tracemalloc
    path = tmp_path / "memory.json"
    tracemalloc.start()
    try:
        monitor = TrainingMonitor(path, epochs=510, batches_per_epoch=1, history_limit=10)
        for epoch in range(1, 511):
            monitor.batch(epoch=epoch, batch=1, metrics={"loss": 1/epoch})
        snapshot_html(path)
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 32 * 1024 * 1024
    finally:
        tracemalloc.stop()


def test_fixed_header_all_epochs(tmp_path):
    path = tmp_path / "all.json"
    monitor = TrainingMonitor(path, epochs=25, batches_per_epoch=1)
    for epoch in range(1, 26):
        monitor.batch(epoch=epoch, batch=1)
    monitor.finish()
    html = snapshot_html(path, dashboard_height=320)
    assert html.count("<article ") == 20
    assert "Epoch 1 / 25" not in html
    assert "Epoch 6 / 25" in html
    assert "5 earlier epochs retained" in html
    events = [json.loads(line) for line in path.with_name(path.name + ".epochs.jsonl").read_text().splitlines()]
    assert [e["epoch"] for e in events if e["event"] == "training_epoch"] == list(range(1, 26))
    assert html.index("Overall training progress") < html.index("</header>") < html.index("overflow-y:auto")
    assert html.index("Epoch 6 / 25") < html.index("Epoch 25 / 25")
    assert "max-height:320px" in html
    assert "overflow-y:auto" not in snapshot_html(path, fixed_header=False)
    with pytest.raises(ValueError):
        snapshot_html(path, dashboard_height=0)


def test_epoch_health_warning(tmp_path):
    from deepforge.epoch_health import epoch_health
    path = tmp_path / "health.json"
    monitor = TrainingMonitor(path, epochs=4, batches_per_epoch=1)
    for epoch in range(1,5):
        monitor.batch(epoch=epoch,batch=1,metrics={"loss":1/epoch})
        monitor.epoch_end(epoch,{"loss":epoch/2})
    assert epoch_health(monitor.epoch_records,4)[0] == "Severe overfitting warning"
    assert "Severe overfitting warning" in snapshot_html(path)
    assert "Severe overfitting warning" not in snapshot_html(path,show_diagnostics=False)
