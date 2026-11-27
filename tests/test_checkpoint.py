import pytest

from deepforge.checkpoint import CheckpointPolicy, CheckpointStore


def test_retention_and_best(tmp_path):
    store = CheckpointStore(tmp_path, "run")
    store.save(1, b"best", best=True)
    for epoch in range(2, 5):
        store.save(epoch, str(epoch).encode())
    assert store.latest().read_bytes() == b"4"
    assert sorted(p.name for p in store.root.glob("*.bin")) == [
        "epoch-1.bin", "epoch-3.bin", "epoch-4.bin"
    ]


def test_corruption_fallback(tmp_path):
    store = CheckpointStore(tmp_path, "run")
    store.save(1, b"good")
    latest = store.save(2, b"new")
    latest.write_bytes(b"bad")
    assert store.latest().read_bytes() == b"good"


def test_quota_preserves_existing(tmp_path):
    store = CheckpointStore(tmp_path, "run", CheckpointPolicy(max_bytes=5))
    store.save(1, b"1234")
    with pytest.raises(OSError):
        store.save(2, b"56")
    assert store.latest().read_bytes() == b"1234"


def test_lock_blocks_second_writer(tmp_path):
    store = CheckpointStore(tmp_path, "run")
    (store.root / ".writer.lock").touch()
    with pytest.raises(FileExistsError):
        store.save(1, b"data")


def test_disabled_and_interval(tmp_path):
    store = CheckpointStore(tmp_path, "run", CheckpointPolicy(enabled=False))
    assert store.save(1, b"data") is None
    store = CheckpointStore(tmp_path, "other", CheckpointPolicy(save_every_epochs=2))
    assert store.save(1, b"data") is None
    assert store.save(2, b"data").exists()


def test_invalid_run_id(tmp_path):
    with pytest.raises(ValueError):
        CheckpointStore(tmp_path, "../escape")


def test_abandoned_lock_recovery(tmp_path):
    store = CheckpointStore(tmp_path, "run")
    store.save(1, b"safe")
    (store.root / ".writer.lock").touch()
    with pytest.raises(ValueError):
        store.recover_abandoned_lock()
    store.recover_abandoned_lock(confirmed_no_writer=True)
    assert store.save(2, b"resumed").exists()


def test_unpublished_payload_recovery(tmp_path):
    store = CheckpointStore(tmp_path, "run")
    (store.root / "epoch-1.bin").write_bytes(b"incomplete")
    assert store.latest() is None
    assert store.save(1, b"complete").read_bytes() == b"complete"
