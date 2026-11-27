import json
from pathlib import Path
import pytest
from deepforge.cli import main
from deepforge.datasets import inspect_dataset
from deepforge.history import analyze_history
from deepforge.metrics import classification_metrics, evaluate_csv
from deepforge.models import inspect_model
from deepforge.environment import inspect_environment

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def make_file(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_dataset_counts(tmp_path):
    make_file(tmp_path, "train/cat/a.jpg", b"a")
    make_file(tmp_path, "train/dog/b.png", b"b")
    make_file(tmp_path, "test/cat/c.jpg", b"c")
    make_file(tmp_path, "test/cat/note.txt", b"ignore")
    r = inspect_dataset(tmp_path)
    assert r["total"] == 3
    assert r["splits"]["train"]["classes"] == {"cat": 1, "dog": 1}
    assert r["splits"]["train"]["class_fractions"] == {"cat": .5, "dog": .5}


def test_leakage_across_splits(tmp_path):
    make_file(tmp_path, "train/cat/a.jpg", b"same")
    make_file(tmp_path, "val/dog/b.png", b"same")
    make_file(tmp_path, "train/cat/c.jpg", b"only train")
    make_file(tmp_path, "train/dog/d.jpg", b"only train")
    r = inspect_dataset(tmp_path, hash_files=True)
    assert r["cross_split_duplicate_groups"] == 1
    assert r["duplicates"][0]["files"] == ["train/cat/a.jpg", "val/dog/b.png"]


def test_empty_split(tmp_path):
    (tmp_path / "train" / "cat").mkdir(parents=True)
    assert inspect_dataset(tmp_path)["splits"]["train"]["class_fractions"] == {"cat": 0.0}


def test_no_splits(tmp_path):
    with pytest.raises(ValueError):
        inspect_dataset(tmp_path)


def test_ambiguous_val(tmp_path):
    (tmp_path / "val").mkdir()
    (tmp_path / "validation").mkdir()
    with pytest.raises(ValueError):
        inspect_dataset(tmp_path)


def test_leakage_cli(tmp_path, capsys):
    make_file(tmp_path, "train/cat/a.jpg", b"same")
    make_file(tmp_path, "test/cat/b.jpg", b"same")
    assert main(["leakage", str(tmp_path), "--fail-on-leakage"]) == 1
    assert json.loads(capsys.readouterr().out)["cross_split_duplicate_groups"] == 1


def test_metrics_example():
    r = evaluate_csv(EXAMPLES / "predictions.csv")
    assert r["accuracy"] == .75
    assert r["confusion_matrix"] == [[1, 1], [0, 2]]
    assert r["per_class"]["cat"]["precision"] == 1
    assert r["per_class"]["dog"]["recall"] == 1
    assert r["macro"]["f1"] == pytest.approx((2/3 + .8)/2)


def test_unseen_label():
    r = classification_metrics(["a"], ["b"])
    assert r["labels"] == ["a", "b"]
    assert r["accuracy"] == r["macro"]["f1"] == 0


@pytest.mark.parametrize("actual,pred", [([], []), (["a"], []), ([""], ["a"])])
def test_invalid_labels(actual, pred):
    with pytest.raises(ValueError):
        classification_metrics(actual, pred)


def test_history_example():
    r = analyze_history(EXAMPLES / "history.csv", train_metric="train_loss")
    assert r["best_epoch"] == 2
    assert r["best_value"] == .75
    assert r["possible_overfitting_signal"] is True


def test_history_max(tmp_path):
    p = tmp_path / "history.csv"
    p.write_text("epoch,accuracy\n1,.5\n2,.9\n3,.9\n", encoding="utf-8")
    r = analyze_history(p, metric="accuracy", mode="max")
    assert r["best_epoch"] == 2
    assert not r["validation_worse_than_best"]


@pytest.mark.parametrize("rows", ["1,nan\n", "1,inf\n", "2,.2\n1,.3\n", "1,\n"])
def test_invalid_history(tmp_path, rows):
    p = tmp_path / "history.csv"
    p.write_text("epoch,val_loss\n" + rows, encoding="utf-8")
    with pytest.raises(ValueError):
        analyze_history(p)


def test_model_counts_without_torch():
    class Tensor:
        def __init__(self, n, trainable):
            self.n = n
            self.requires_grad = trainable
        def numel(self): return self.n
        def element_size(self): return 4
    a, b, c = Tensor(12, True), Tensor(3, False), Tensor(2, False)
    class Model:
        def parameters(self): return iter([a, b, a])
        def buffers(self): return iter([c])
    assert inspect_model(Model()) == {"parameters": 15, "trainable_parameters": 12,
                                      "frozen_parameters": 3, "buffer_elements": 2,
                                      "parameter_bytes": 60, "buffer_bytes": 8}


def test_model_real_torch():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(5, 3)
    model.bias.requires_grad_(False)
    r = inspect_model(model)
    assert r["parameters"] == 18
    assert r["trainable_parameters"] == 15
    assert r["frozen_parameters"] == 3


def test_environment():
    report = inspect_environment()
    assert report["python"]
    assert isinstance(report["pytorch"]["available"], bool)


def test_cli_metrics(capsys):
    assert main(["metrics", str(EXAMPLES / "predictions.csv")]) == 0
    assert json.loads(capsys.readouterr().out)["accuracy"] == .75


def test_cli_bad_path(tmp_path, capsys):
    assert main(["dataset", str(tmp_path / "missing")]) == 2
    assert "Not a dataset directory" in capsys.readouterr().err
