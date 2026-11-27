import json
import pytest
from deepforge.profiler import profile_operators, profile_cuda_peak


def test_operator_matmul():
    torch = pytest.importorskip("torch")
    class Functional(torch.nn.Module):
        def forward(self, x, y):
            return x @ y
    model = Functional()
    result = profile_operators(model, torch.zeros(2, 3), torch.zeros(3, 4))
    assert result["estimated_flops"] == 48
    assert model.training
    assert result["operators_with_positive_estimate"] >= 1
    assert any(row["operator"] == "aten::mm" for row in result["operators"])
    json.dumps(result, allow_nan=False)


def test_operator_limit_and_cleanup():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 2)
    model.train()
    result = profile_operators(model, torch.ones(1, 3), max_rows=1)
    assert len(result["operators"]) == 1
    assert result["rows_truncated"]
    with pytest.raises(RuntimeError):
        profile_operators(model, torch.ones(1, 4))
    assert model.training


def test_cuda_requires_device():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="CUDA"):
        profile_cuda_peak(torch.nn.Linear(2, 2), torch.ones(1, 2))


def test_cuda_peak_on_available_hardware():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is not available")
    model = torch.nn.Linear(8, 4).cuda()
    result = profile_cuda_peak(model, torch.ones(16, 8, device="cuda"))
    row = result["devices"][0]
    assert row["peak_allocated_bytes"] >= row["baseline_allocated_bytes"]
    assert row["peak_above_baseline_bytes"] >= 0
    assert result["peak_counters_reset"] and model.training


def test_gui_batch_and_chart():
    pytest.importorskip("torch")
    from deepforge.gui_data import analyze, summary
    report, chart = analyze("Model", {"inputs":"4","hidden":"3","outputs":"2","batch":"5"})
    assert report["estimated_macs"] == 90
    assert sum(v for _,v in chart["values"]) == 90
    assert ("Partial MACs", "90") in summary("Model", report)
