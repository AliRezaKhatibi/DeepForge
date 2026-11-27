import json
import pytest
from deepforge.profiler import profile_model, profile_mlp


def test_linear_counts_and_shapes():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    result = profile_model(model, torch.zeros(5, 4), warmup=0, repeats=0)
    assert result["parameters"] == 23
    assert result["registered_modules"] == 4
    assert result["registered_leaf_modules"] == 3
    assert result["executed_leaf_calls"] == 3
    assert result["estimated_macs"] == 90
    assert result["estimated_flops_2x_macs"] == 180
    assert result["unestimated_leaf_types"] == {"ReLU": 1}
    assert result["calls"][0]["outputs"][0]["shape"] == [5, 3]
    assert result["output_tensors"][0]["shape"] == [5, 2]
    assert result["leaf_output_logical_bytes_sum"] == 160
    json.dumps(result, allow_nan=False)


def test_grouped_convolution():
    torch = pytest.importorskip("torch")
    model = torch.nn.Conv2d(4, 6, 3, groups=2, bias=False)
    result = profile_model(model, torch.zeros(2, 4, 8, 8), repeats=0)
    assert result["parameters"] == 108
    assert result["estimated_macs"] == 2*6*6*6*2*3*3
    assert result["calls"][0]["name"] == "<root>"


def test_reused_module_and_truncation():
    torch = pytest.importorskip("torch")
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
        def forward(self, x):
            return self.linear(self.linear(x))
    model = Model()
    report = profile_model(model, torch.ones(1, 2), repeats=0, max_rows=1)
    assert report["parameters"] == 6
    assert report["executed_leaf_calls"] == 2
    assert report["estimated_macs"] == 8
    assert report["calls_truncated"]
    assert len(report["calls"]) == 1


def test_mode_restoration_and_timing():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.train()
    model[0].eval()
    result = profile_model(model, torch.ones(1, 2), warmup=1, repeats=3)
    assert model.training and not model[0].training and model[1].training
    assert result["latency"]["repeats"] == 3
    assert 0 <= result["latency"]["min_ms"] <= result["latency"]["p95_ms"]
    assert not model[0]._forward_hooks


def test_cleanup_on_forward_failure():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 2)
    model.train()
    with pytest.raises(RuntimeError):
        profile_model(model, torch.ones(1, 4), repeats=0)
    assert model.training
    assert not model._forward_hooks


def test_nested_kwargs_and_output():
    torch = pytest.importorskip("torch")
    class Model(torch.nn.Module):
        def forward(self, *, data):
            return {"out": data["x"] + 1, "other": "private string"}
    report = profile_model(Model(), kwargs={"data": {"x":torch.ones(2,3)}}, repeats=0)
    assert report["input_tensors"][0]["shape"] == [2,3]
    assert report["output_tensors"][0]["shape"] == [2,3]
    assert "private string" not in json.dumps(report)
    assert report["estimated_macs"] == 0
    assert report["unestimated_leaf_types"] == {"Model":1}


def test_mlp_limits():
    with pytest.raises(ValueError):
        profile_mlp(4096, 4096, 1)
    with pytest.raises(ValueError):
        profile_mlp(2, 2, 2, batch=257)


def test_cli_and_gui(capsys):
    pytest.importorskip("torch")
    from deepforge.cli import main
    from deepforge.gui_data import analyze
    assert main(["profile-mlp", "--inputs", "4", "--hidden", "3",
                 "--outputs", "2", "--batch", "5", "--repeats", "0"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["estimated_macs"] == 90
    result, chart = analyze("Model", {"inputs":"4","hidden":"3","outputs":"2"})
    assert result["estimated_macs"] == 18
    assert chart["type"] == "bars"
