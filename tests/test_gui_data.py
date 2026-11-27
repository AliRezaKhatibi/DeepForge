from pathlib import Path
import pytest
from deepforge.gui_data import analyze, summary

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_gui_metrics():
    report, chart = analyze("Metrics", {"path": EXAMPLES / "predictions.csv", "true_column": "y_true", "pred_column": "y_pred"})
    assert chart["type"] == "matrix"
    assert chart["values"] == report["confusion_matrix"]
    assert ("Accuracy", "75.0%") in summary("Metrics", report)


def test_gui_training():
    report, chart = analyze("Training", {"path": EXAMPLES / "history.csv", "metric": "val_loss", "train_metric": "train_loss", "mode": "min", "epoch_column": "epoch"})
    assert report["best_epoch"] == 2
    assert chart["series"]["val_loss"][1] == .75


def test_gui_dataset(tmp_path):
    p = tmp_path / "train" / "cat"; p.mkdir(parents=True); (p / "a.png").write_bytes(b"sample")
    report, chart = analyze("Dataset", {"path": tmp_path})
    assert chart["values"] == [("train", 1)]
    assert ("Image files", "1") in summary("Dataset", report)


def test_gui_invalid_model():
    with pytest.raises(ValueError):
        analyze("Model", {"inputs": "0", "hidden": "32", "outputs": "3"})


def test_gui_unknown_tool():
    with pytest.raises(ValueError):
        analyze("missing", {})
