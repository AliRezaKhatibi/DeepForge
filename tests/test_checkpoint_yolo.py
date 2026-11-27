from types import SimpleNamespace

from deepforge.checkpoint import CheckpointPolicy, CheckpointStore
from deepforge.checkpoint_yolo import attach_yolo_checkpoint


def test_yolo_native_capture(tmp_path):
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda name, fn: callbacks.update({name: fn}))
    store = CheckpointStore(tmp_path, "yolo")
    last = tmp_path / "last.pt"
    trainer = SimpleNamespace(epoch=0, last=last, best_fitness=1.0, fitness=1.0)
    attach_yolo_checkpoint(model, store)
    last.write_bytes(b"native-first")
    callbacks["on_model_save"](trainer)
    trainer.epoch = 1
    trainer.fitness = 0.5
    last.write_bytes(b"native-second")
    callbacks["on_model_save"](trainer)
    assert store.latest().read_bytes() == b"native-second"
    store.latest().write_bytes(b"corrupt")
    assert store.latest().read_bytes() == b"native-first"


def test_yolo_disabled(tmp_path):
    store = CheckpointStore(tmp_path, "disabled", CheckpointPolicy(enabled=False))
    model = SimpleNamespace(add_callback=lambda name, fn: None)
    capture = attach_yolo_checkpoint(model, store)
    capture(SimpleNamespace(epoch=0))
    assert store.latest() is None
