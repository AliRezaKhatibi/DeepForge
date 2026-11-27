"""Render a local training snapshot in Jupyter without a desktop display."""
import html
import json
from pathlib import Path


def snapshot_html(path, *, max_epochs=12, show_stages=True):
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
    rows.extend(("Diagnostic", item["code"]) for item in data.get("diagnostics", {}).get("findings", []))
    rows.extend(("Batch class " + k, v) for k,v in data.get("batch_distribution", {}).get("counts", {}).items())
    encoded = data.get("preview_png")
    preview = ""
    if isinstance(encoded, str) and len(encoded) <= 800000 and encoded.startswith("iVBORw0KGgo"):
        preview = f'<img alt="Opt-in batch preview" style="max-width:100%" src="data:image/png;base64,{esc(encoded)}">'
    def fmt(value):
        return f"{value:.4f}" if isinstance(value, float) else esc(value)

    def bar(percent):
        return (f'<div role="progressbar" aria-valuenow="{percent:.1f}" aria-valuemin="0" aria-valuemax="100" '
                'style="background:#dddcd3;border-radius:8px;height:9px;overflow:hidden">'
                f'<div style="background:#24765c;width:{percent:.2f}%;height:9px"></div></div>')

    records = data.get("epoch_records", {})
    current = max(1, data["epoch"])
    first = max(1, current - max_epochs + 2)
    last = min(data["epochs"], current + 1)
    cards = []
    for epoch in range(first, last + 1):
        done = max(0, min(data["batches_per_epoch"], data["completed_batches"] - (epoch - 1) * data["batches_per_epoch"]))
        percent = 100 * done / data["batches_per_epoch"]
        record = records.get(str(epoch), {})
        state = "Training complete" if percent == 100 else ("Running" if done else "Pending")
        metrics = record.get("training", data["metrics"] if epoch == current else {})
        details = " · ".join(f"{esc(k)}: {fmt(v)}" for k, v in metrics.items())
        stages = []
        if show_stages:
            for stage in ("validation", "checkpoint", "verification"):
                item = record.get("stages", {}).get(stage, {})
                status = item.get("status", "Not reported")
                count, total = item.get("completed"), item.get("total")
                meter = bar(100 * count / total) if total else (bar(100) if status == "completed" else "")
                stages.append(f'<div style="margin-top:8px">{stage.title()}: {esc(status)}{meter}</div>')
        cards.append(f'<article style="background:#fff;color:#123b32;border:1px solid #dedbd0;border-radius:12px;padding:12px;margin-top:10px">'
                     f'<strong>Epoch {epoch} / {data["epochs"]}</strong> · {state}{bar(percent)}'
                     f'<div>{done}/{data["batches_per_epoch"]} batches · {percent:.1f}%</div>'
                     f'<div style="overflow-wrap:anywhere">{details}</div>{"".join(stages)}</article>')
    table = "".join(f"<tr><th>{esc(k)}</th><td>{fmt(v)}</td></tr>" for k,v in rows)
    return ('<section style="background:#F3EFE5;color:#123B32;padding:16px;border-radius:12px">'
            '<h2 style="color:#123B32;margin:0 0 12px">DeepForge · XVARNA</h2>'
            f'<strong>Overall training progress</strong> {progress:.1f}%{bar(progress)}'
            f'{"".join(cards)}<table style="color:#123B32;overflow-wrap:anywhere;width:100%">{table}</table>{preview}</section>')


def display_snapshot(path, *, display_id=None):
    """Display or update one notebook output. Caller controls refresh timing."""
    from IPython.display import HTML, display
    return display(HTML(snapshot_html(path)), display_id=display_id)


class NotebookMonitor:
    """Refresh a single inline output from an existing TrainingMonitor snapshot.

    Call refresh from the training thread after recording batches or epochs.
    No background thread, GUI, or browser server is required.
    """
    def __init__(self, path, *, interval=1.0, max_epochs=12, show_stages=True):
        import math
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("interval must be finite and nonnegative")
        if type(max_epochs) is not int or max_epochs < 2:
            raise ValueError("max_epochs must be an integer >= 2")
        self.max_epochs = max_epochs
        self.show_stages = show_stages
        self.path = Path(path)
        self.interval = interval
        self._last = None
        self._handle = None

    def refresh(self, *, force=False):
        import time
        from IPython.display import HTML, display
        now = time.monotonic()
        if not force and self._last is not None and now - self._last < self.interval:
            return False
        output = HTML(snapshot_html(self.path, max_epochs=self.max_epochs, show_stages=self.show_stages))
        if self._handle is None:
            self._handle = display(output, display_id=True)
        else:
            self._handle.update(output)
        self._last = now
        return True
