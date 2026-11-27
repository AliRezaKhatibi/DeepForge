import json
import pytest
from deepforge.automation import (
    AuditFailed, Check, classification_check, dataset_check, group_check,
    image_check, require_pass, run_checks, save_report, yolo_check,
)


def test_custom_checks_and_failure():
    report = run_checks([Check("a", lambda: {"x": {"y": 2}}, (("x.y", "ge", 2),)),
                         Check("b", lambda: {"x": 1}, (("x", "eq", 0),))])
    assert report["counts"] == {"passed": 1, "failed": 1, "error": 0}
    with pytest.raises(AuditFailed):
        require_pass(report)


def test_errors_fail_closed():
    report = run_checks([Check("missing", lambda: {}, (("x", "eq", 0),)),
                         Check("nonfinite", lambda: {"x": float("nan")})])
    assert report["counts"]["error"] == 2
    assert not report["ok"]


def test_empty_and_duplicate_checks():
    with pytest.raises(ValueError):
        run_checks([])
    with pytest.raises(ValueError):
        run_checks([Check("x", dict), Check("x", dict)])


def test_dataset_clean_and_leaking(tmp_path):
    for split in ("train", "val", "test"):
        folder = tmp_path / split / "class"
        folder.mkdir(parents=True)
        (folder / "sample.png").write_bytes(split.encode())
    assert run_checks([dataset_check(tmp_path)])["ok"]
    (tmp_path / "test/class/sample.png").write_bytes(b"train")
    assert not run_checks([dataset_check(tmp_path)])["ok"]


def test_missing_split(tmp_path):
    folder = tmp_path / "train/class"
    folder.mkdir(parents=True)
    (folder / "sample.png").write_bytes(b"sample")
    assert not run_checks([dataset_check(tmp_path)])["ok"]


def test_group_leakage(tmp_path):
    path = tmp_path / "groups.csv"
    path.write_text("group_id,split\na,train\na,test\n")
    assert not run_checks([group_check(path)])["ok"]


def test_classification_gate():
    check = classification_check(y_true=[0, 1], y_pred=[0, 0], min_accuracy=.8)
    assert not run_checks([check])["ok"]
    with pytest.raises(AuditFailed):
        run_checks([check], raise_on_failure=True)


def test_save_without_overwrite(tmp_path):
    report = run_checks([Check("x", dict)])
    path = tmp_path / "report.json"
    save_report(report, path)
    assert json.loads(path.read_text()) == report
    with pytest.raises(FileExistsError):
        save_report(report, path)


def test_yolo_missing_input(tmp_path):
    assert run_checks([yolo_check(tmp_path)])["counts"]["error"] == 1


def test_image_missing_input(tmp_path):
    assert run_checks([image_check(tmp_path)])["counts"]["error"] == 1
