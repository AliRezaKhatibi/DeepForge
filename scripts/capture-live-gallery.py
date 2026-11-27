"""Capture real Tk monitor windows using a deterministic synthetic training run.

Run on Linux under Xvfb with ImageMagick installed. No private data or network.
"""
import argparse
import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from deepforge.monitor import TrainingMonitor
from deepforge.notebook import snapshot_html
from deepforge.resources import sample_resources


def capture(output):
    import tkinter as tk
    from unittest.mock import patch
    from deepforge.live_view import main
    import math
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        monitor = TrainingMonitor(work / "live.json", epochs=4,
                                  batches_per_epoch=40, publish_interval=0)
        results = work / "results.csv"
        with results.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["epoch", "batch", "loss"])
            for step in range(1, 101):
                epoch, batch = (step - 1) // 40 + 1, (step - 1) % 40 + 1
                loss = 1.8 * math.exp(-step / 38) + .12 + .035 * math.sin(step)
                monitor.batch(epoch=epoch, batch=batch, samples=16,
                              metrics={"loss":loss,"box_loss":loss*.5,"cls_loss":loss*.3,"dfl_loss":loss*.2})
                writer.writerow([epoch,batch,loss])
                if batch == 40:
                    monitor.epoch_end(epoch, {"demo_precision":.4+epoch*.1,"demo_recall":.3+epoch*.1})
                time.sleep(.015)
        monitor.artifact(results)
        monitor.record_resources(sample_resources())
        # The data above are synthetic metrics, not a benchmark or trained model.
        (output / "live-notebook.html").write_text("<!doctype html><html><head><meta charset=\"utf-8\"><title>DeepForge notebook snapshot</title></head><body style=\"font-family: sans-serif;max-width: 900px;margin: 24px auto;\">" + snapshot_html(monitor.path) + "</body></html>", encoding="utf-8")
        real_mainloop = tk.Tk.mainloop

        def take(window, *args, **kwargs):
            def screenshot(name):
                window.update_idletasks()
                subprocess.run(["import", "-window", str(window.winfo_id()),
                                str(output / name)], check=True)
            def second():
                try:
                    screenshot("live-box-loss.png")
                finally:
                    window.destroy()
            def first():
                try:
                    screenshot("live-training.png")
                    selector = next(w for w in window.winfo_children()
                                    if w.winfo_class() == "TCombobox")
                    selector.set("box_loss")
                    window.after(1200, second)
                except Exception:
                    window.destroy()
                    raise
            window.after(1500, first)
            return real_mainloop(window, *args, **kwargs)

        with patch.object(tk.Tk, "mainloop", take):
            main([str(monitor.path)])
        if not all((output / name).is_file() for name in ("live-training.png", "live-box-loss.png")):
            raise RuntimeError("Screenshot capture failed")
        monitor.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/images"))
    capture(parser.parse_args().output.resolve())
