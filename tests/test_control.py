from types import SimpleNamespace
from deepforge.control import StopControl
from deepforge.checkpoint import CheckpointStore
from deepforge.checkpoint_yolo import attach_yolo_safe_stop

def test_control(tmp_path):
    c=StopControl(tmp_path/"control.json")
    assert c.consume() is None
    c.request(after_next=False)
    assert c.consume()=={"action":"stop","after_next":False}
    c.cancel()
    assert c.consume()=={"action":"cancel"}

def test_yolo_stop(tmp_path):
    callbacks={}
    model=SimpleNamespace(add_callback=lambda name,fn:callbacks.update({name:fn}))
    store=CheckpointStore(tmp_path,"run")
    control=StopControl(tmp_path/"control.json")
    last=tmp_path/"last.pt"
    last.write_bytes(b"native checkpoint")
    trainer=SimpleNamespace(epoch=0,epochs=5,last=last,stop=False)
    attach_yolo_safe_stop(model,store,control)
    control.request()
    callbacks["on_train_batch_end"](trainer)
    callbacks["on_model_save"](trainer)
    assert not trainer.stop
    trainer.epoch=1
    callbacks["on_model_save"](trainer)
    assert trainer.stop
    assert (store.root/"recovery"/"epoch-2.bin").read_bytes()==last.read_bytes()
