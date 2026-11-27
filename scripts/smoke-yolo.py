"""Real CPU integration test using synthetic data, no external dataset or weights."""
import json
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO, settings
from deepforge.ultralytics import attach_monitor

def main():
    settings.update({"sync": False})
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for split in ("train","val"):
            images, labels = root/"images"/split, root/"labels"/split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for i in range(8):
                im = Image.new("RGB",(96,96),(20,30,40))
                x = 20+i*3
                ImageDraw.Draw(im).rectangle((x,30,x+20,50),fill=(200,190,80))
                im.save(images/f"{i}.png")
                (labels/f"{i}.txt").write_text(f"0 {(x+10)/96} {40/96} {20/96} {20/96}\n")
        data = root/"data.yaml"
        data.write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames: [object]\n")
        model = YOLO("yolo11n.yaml")
        path = root/"live.json"
        attach_monitor(model,path,archive=root/"events.jsonl",publish_interval=0)
        model.train(data=str(data),epochs=1,imgsz=64,batch=4,device="cpu",workers=0,
                    pretrained=False,amp=False,plots=False,project=str(root/"runs"),
                    name="smoke",verbose=False,optimizer="SGD",mosaic=0)
        result = json.loads(path.read_text())
        assert result["status"] == "completed", result
        assert result["completed_batches"] == 2
        assert len(result["epoch_history"]) == 1
        assert any(a["name"] == "last.pt" and a["exists"] for a in result["artifacts"])
        assert len((root/"events.jsonl").read_text().splitlines()) == 5
        print("Actual YOLO CPU training monitor verified")

if __name__ == "__main__":
    main()
