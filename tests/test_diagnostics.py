import json
import pytest
from deepforge.diagnostics import diagnose, class_distribution
from deepforge.monitor import TrainingMonitor
from deepforge.notebook import snapshot_html


def test_trend_and_missing():
    rows = [{"metrics":{"train_loss": t,"val_loss":v}} for t,v in [(1,1),(.9,1.1),(.8,1.2),(.7,1.3)]]
    result = diagnose(rows)
    assert result["trend_available"]
    assert result["findings"][0]["code"] == "possible_overfitting"
    rows[2]["metrics"]["val_loss"] = None
    assert not diagnose(rows)["trend_available"]


def test_plateau_and_gradient():
    rows = [{"metrics":{"train_loss":1,"val_loss":2,"gradient_norm":.001}}]*4
    codes = {f["code"] for f in diagnose(rows,gradient_min=.01)["findings"]}
    assert codes == {"small_gradient_signal","loss_plateau"}


def test_labels():
    assert class_distribution([0,0,1])["counts"] == {"0":2,"1":1}
    with pytest.raises(ValueError):
        class_distribution([-1])


def test_archive_and_preview(tmp_path):
    from PIL import Image
    archive = tmp_path/"events.jsonl"
    m = TrainingMonitor(tmp_path/"live.json",epochs=1,batches_per_epoch=2,history_limit=1,archive=archive)
    for b in (1,2):
        m.batch(epoch=1,batch=b,labels=[0,1])
    m.epoch_end(1,{"train_loss":1,"val_loss":1})
    image = tmp_path/"demo.png"
    Image.new("RGB",(16,16),"green").save(image)
    m.record_preview(image)
    assert "data:image/png;base64,iVBORw0KGgo" in snapshot_html(m.path)
    m.finish()
    assert [json.loads(line)["event"] for line in archive.read_text().splitlines()] == ["start","batch","batch","epoch","finish"]
    assert len(m.snapshot()["history"]) == 1
    assert m.snapshot()["batch_distribution"]["total_labels"] == 2
    with pytest.raises(FileExistsError):
        TrainingMonitor(tmp_path/"other.json",epochs=1,batches_per_epoch=1,archive=archive)
