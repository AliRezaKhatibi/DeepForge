"""Exercise the actual desktop window when a graphical display is available."""
import time
from pathlib import Path

import pytest


def test_studio_workspaces(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    from deepforge.gui import Studio, messagebox

    def fail_dialog(title, message, **kwargs):
        pytest.fail(f"{title}: {message}")

    monkeypatch.setattr(messagebox, "showerror", fail_dialog)
    monkeypatch.setattr(messagebox, "showwarning", fail_dialog)

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("A graphical display is required")
    try:
        app = Studio(root)
        examples = Path(__file__).resolve().parents[1] / "examples"
        for split in ("train", "val"):
            folder = tmp_path / split / "sample"
            folder.mkdir(parents=True)
            (folder / "sample.png").write_bytes(b"synthetic fixture")
        (tmp_path / 'yolo' / 'images' / 'train').mkdir(parents=True)
        (tmp_path / 'yolo' / 'labels' / 'train').mkdir(parents=True)
        (tmp_path / 'yolo' / 'images' / 'train' / 'a.jpg').write_bytes(b'fixture')
        (tmp_path / 'yolo' / 'labels' / 'train' / 'a.txt').write_text('0 .5 .5 .2 .2', encoding='utf-8')
        workspaces = [
            ("Dataset", {"path": tmp_path}),
            ("Leakage", {"path": tmp_path}),
            ("Training", {"path": examples / "history.csv"}),
            ("Metrics", {"path": examples / "predictions.csv"}),
            ("Environment", {}),
            ("Images", {"path": tmp_path}),
            ("Groups", {"path": examples / "groups.csv"}),
            ("YOLO", {"path": tmp_path / "yolo"}),
            ("Compare", {"path": examples / "runs"}),
            ("Scores", {"path": examples / "scores.csv"}),
            ("Manifest", {"path": examples / "history.csv"}),
        ]
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        else:
            workspaces.append(("Model", {}))
        for tool, options in workspaces:
            app.select(tool)
            for field, value in options.items():
                app.fields[field].set(str(value))
            root.update()
            app.run()
            deadline = time.monotonic() + 30
            while app.busy and time.monotonic() < deadline:
                root.update()
                time.sleep(0.02)
            assert not app.busy, tool
            assert app.report is not None, tool
            assert str(app.export_btn.cget("state")) == "normal", tool
            assert app.table.get_children(), tool
            if tool == "Scores":
                for mode in ("ROC", "Precision–recall", "Calibration"):
                    app.chart_mode.set(mode); app.draw(); root.update()
        app.select("Home"); root.update()
        assert len(app.recent) == len(workspaces)
        app.reopen(app.recent[-1]); root.update()
        assert app.report is not None
    finally:
        root.destroy()
