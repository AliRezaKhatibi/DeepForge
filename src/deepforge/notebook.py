"""Render a local training snapshot in Jupyter without a desktop display."""
import html
import json
from pathlib import Path


def snapshot_html(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["schema_version"] != 1:
        raise ValueError("Unsupported snapshot schema")
    esc = lambda value: html.escape(str(value), quote=True)
    progress = min(100, max(0, float(data["progress"]) * 100))
    rows = []
    for label, value in (
        ("Status", data["status"]), ("Epoch", f'{data["epoch"]}/{data["epochs"]}'),
        ("Batch", f'{data["batch"]}/{data["batches_per_epoch"]}'),
        ("Elapsed (s)", round(data["elapsed_seconds"], 1)),
        ("ETA (s)", None if data["eta_seconds"] is None else round(data["eta_seconds"], 1)),
        ("Batches/s", round(data["batches_per_second"], 2)),
    ):
        rows.append((label, value))
    rows.extend(data["metrics"].items())
    if data.get("epoch_history"):
        rows.extend(("Validation: " + k, v) for k,v in data["epoch_history"][-1]["metrics"].items())
    rows.extend(data.get("resources", {}).items())
    rows.extend(("Output: " + a["name"], a["bytes"]) for a in data["artifacts"])
    table = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k,v in rows)
    return ('<section style="background:#F3EFE5;color:#123B32;padding:16px;border-radius:12px">'
            '<h2>DeepForge · XVARNA</h2>'
            f'<progress max="100" value="{progress:.4f}"></progress> {progress:.1f}%'
            f'<table>{table}</table></section>')


def display_snapshot(path, *, display_id=None):
    """Display or update one notebook output. Caller controls refresh timing."""
    from IPython.display import HTML, display
    return display(HTML(snapshot_html(path)), display_id=display_id)
