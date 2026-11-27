import pytest
import random

torch = pytest.importorskip("torch")
from deepforge.checkpoint import CheckpointStore
from deepforge.checkpoint_torch import TorchCheckpoint


def test_resume_matches_uninterrupted(tmp_path):
    torch.manual_seed(42)
    random.seed(42)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    store = CheckpointStore(tmp_path, "training")
    adapter = TorchCheckpoint(store, model, optimizer, scheduler=scheduler)
    metadata = {"dataset": "synthetic-v1", "seed": 42}

    def step(m, opt, sched):
        opt.zero_grad()
        loss = m(torch.rand(4, 3)).square().mean()
        loss.backward()
        opt.step()
        sched.step()
        return loss.detach()

    step(model, optimizer, scheduler)
    adapter.save_epoch(1, global_step=1, metadata=metadata)
    expected_loss = step(model, optimizer, scheduler)
    expected_weights = [p.detach().clone() for p in model.parameters()]
    resumed = torch.nn.Linear(3, 1)
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1)
    recovery = TorchCheckpoint(store, resumed, resumed_optimizer, scheduler=resumed_scheduler)
    with pytest.raises(ValueError, match="metadata mismatch"):
        recovery.restore_latest(expected_metadata={"dataset": "different"})
    assert recovery.restore_latest(expected_metadata=metadata) == {"epoch": 1, "global_step": 1}
    actual_loss = step(resumed, resumed_optimizer, resumed_scheduler)
    assert torch.equal(expected_loss, actual_loss)
    for expected, actual in zip(expected_weights, resumed.parameters()):
        assert torch.equal(expected, actual)
