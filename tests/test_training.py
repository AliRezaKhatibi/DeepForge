import math
import pytest
from deepforge.automation import AuditFailed, run_checks
from deepforge.training import loss_check, model_check, gradient_check, gradient_report, TrainingGuard


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss(value):
    assert not run_checks([loss_check(value)])["ok"]


def test_finite_loss():
    assert run_checks([loss_check(-1.0)])["ok"]


def test_invalid_limit():
    with pytest.raises(ValueError):
        TrainingGuard(None, max_gradient_norm=float("nan"))


def test_guard_valid_and_no_mutation():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    before = [p.grad.clone() for p in model.parameters()]
    guard = TrainingGuard(model, max_gradient_norm=10)
    assert guard.before_step(loss)["ok"]
    assert all(torch.equal(a, p.grad) for a, p in zip(before, model.parameters()))
    assert run_checks([model_check(model, max_parameters=3)])["ok"]
    assert not run_checks([model_check(model, max_parameters=2)])["ok"]


def test_gradient_norm_and_missing():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    assert not run_checks([gradient_check(model)])["ok"]
    model.weight.grad = torch.tensor([[3., 4.]])
    assert gradient_report(model)["global_l2"] == pytest.approx(5.)
    assert run_checks([gradient_check(model)])["ok"]
    assert not run_checks([gradient_check(model, require_all=True)])["ok"]
    assert not run_checks([gradient_check(model, max_norm=4)])["ok"]


def test_nonfinite_gradient_stops_step():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    guard = TrainingGuard(model)
    with pytest.raises(AuditFailed) as exc:
        guard.before_step(1.)
    assert exc.value.report["checks"][1]["report"]["global_l2"] is None


def test_sparse_gradient():
    torch = pytest.importorskip("torch")
    model = torch.nn.Embedding(4, 2, sparse=True)
    model(torch.tensor([1, 1])).sum().backward()
    report = gradient_report(model)
    assert report["global_l2"] == pytest.approx(math.sqrt(8))
    assert report["finite"]
