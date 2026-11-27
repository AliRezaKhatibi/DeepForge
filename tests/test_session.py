import pytest
from deepforge.monitor import TrainingMonitor
from deepforge.checkpoint import CheckpointStore, CheckpointPolicy
from deepforge.checkpoint_torch import TorchCheckpoint
from deepforge.session import TrainingSession

def test_safe_stop_resume(tmp_path):
    torch = pytest.importorskip("torch")
    def build():
        model = torch.nn.Linear(2,1)
        optimizer = torch.optim.Adam(model.parameters())
        monitor = TrainingMonitor(tmp_path/"live.json",epochs=4,batches_per_epoch=1)
        store = CheckpointStore(tmp_path,"run",CheckpointPolicy(enabled=False))
        return model,monitor,TrainingSession(monitor,TorchCheckpoint(store,model,optimizer),{"dataset":"test"})
    model,m,s = build()
    m.batch(epoch=1,batch=1)
    m.epoch_end(1,{"loss":1})
    assert s.request_stop() == 2
    assert not s.epoch_boundary(1,global_step=1)
    m.batch(epoch=2,batch=1)
    m.epoch_end(2,{"loss":0.5})
    assert s.epoch_boundary(2,global_step=2)
    assert m.status == "paused"
    expected = {k:v.clone() for k,v in model.state_dict().items()}
    model2,m2,s2 = build()
    assert s2.restore()["epoch"] == 2
    assert all(torch.equal(v,model2.state_dict()[k]) for k,v in expected.items())
    m2.batch(epoch=3,batch=1)
    assert m2.completed == 3
