"""Capture the actual advanced Tk window with explicitly synthetic metrics under Xvfb."""
import argparse
import subprocess
import tempfile
from pathlib import Path

from deepforge.checkpoint import atomic_json
from deepforge.desktop_monitor import DesktopMonitor
from deepforge.monitor import TrainingMonitor
from deepforge.resources import sample_resources


def capture(output):
    import tkinter as tk
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"live.json"
        m=TrainingMonitor(path,epochs=5,batches_per_epoch=375,publish_interval=10)
        for epoch in (1,2):
            for batch in range(1,376): m.batch(epoch=epoch,batch=batch,metrics={"loss":.72/epoch,"accuracy":.75+.07*epoch,"gradient_norm":2.4})
            m.epoch_end(epoch,{"loss":.66/epoch,"accuracy":.79+.07*epoch,"precision":.8+.06*epoch,"recall":.78+.06*epoch,"f1":.79+.06*epoch})
            for stage in ("validation","checkpoint","verification"): m.record_stage(epoch,stage,"completed")
        for batch in range(1,219): m.batch(epoch=3,batch=batch,metrics={"loss":.22,"gradient_norm":2.41})
        m.record_resources(sample_resources())
        atomic_json(path.parent/"desktop.json",{"run_id":"synthetic-gallery","model":"Demo fixture","dataset":{"name":"Synthetic test observations"},"controls":True,"readiness":{"Configuration":True,"Checkpoint settings":True}})
        root=tk.Tk(); app=DesktopMonitor(root,path,interval=10); root.geometry("1440x880+0+0");root.title("DeepForge | SYNTHETIC TEST DATA")
        try:
            root.update();root.update_idletasks()
            for name in ("Training","Metrics","Hardware","Model","Dataset","Reports"):
                app.select(name); root.update();root.update_idletasks()
                subprocess.run(["import","-window",str(root.winfo_id()),str(output/f"desktop-{name.lower()}.png")],check=True)
        finally: app.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);capture(parser.parse_args().output.resolve())
