"""Capture the actual Model Inspector with a built-in synthetic MLP."""
import argparse
import subprocess
import time
from pathlib import Path
import tkinter as tk
from deepforge.gui import Studio, messagebox
from deepforge.reporting import export_report
from deepforge.gui_data import summary


def capture(output):
    output.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    errors = []
    messagebox.showerror = lambda *a, **kw: errors.append(str(a))
    try:
        app = Studio(root)
        root.geometry("1320x850+0+0")
        app.select("Model")
        for key, value in {"inputs":"64","hidden":"128","outputs":"10","batch":"8"}.items():
            app.fields[key].set(value)
        root.update()
        app.run()
        deadline = time.monotonic() + 60
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(.03)
        if errors or app.busy or app.report is None:
            raise RuntimeError(f"Model analysis failed: {errors}")
        for index, name in ((0, "architecture-profiler.png"), (1, "architecture-layers.png")):
            app.tabs.select(index)
            root.update()
            time.sleep(.2)
            root.update()
            subprocess.run(["import", "-window", str(root.winfo_id()), str(output/name)], check=True)
            if not (output/name).stat().st_size:
                raise RuntimeError("Empty screenshot")
        export_report(output/"architecture-report.html", "Architecture profile", app.report,
                      cards=summary("Model", app.report), chart=app.chart)
    finally:
        root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("gallery"))
    capture(parser.parse_args().output.resolve())
