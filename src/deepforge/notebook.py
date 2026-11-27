"""Render a local training snapshot in Jupyter without a desktop display."""
import html
import json
import math
from pathlib import Path


def snapshot_html(path, *, max_epochs=20, show_stages=True, show_charts=True, show_resources=True, theme='jade', show_history=True, show_epochs=True, show_preview=True, show_diagnostics=True, show_artifacts=True, show_metrics=True, show_overall=True, fixed_header=True, dashboard_height=480, epoch_metrics=None, overall_fields=None, alert_patience=3, alert_threshold=0.25, gradient_threshold=100.0):
    if max_epochs is not None and (type(max_epochs) is not int or max_epochs < 2):
        raise ValueError("max_epochs must be None or an integer >= 2")
    if type(dashboard_height) is not int or dashboard_height < 120:
        raise ValueError('dashboard_height must be an integer >= 120')
    palettes = {"jade": ("#F3EFE5", "#123B32", "#ffffff"), "light": ("#ffffff", "#17212b", "#f5f5f5"), "dark": ("#17212b", "#f3efe5", "#253344")}
    if theme not in palettes:
        raise ValueError("Unknown theme")
    bg, fg, panel = palettes[theme]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["schema_version"] != 1:
        raise ValueError("Unsupported snapshot schema")
    esc = lambda value: html.escape(str(value), quote=True)
    progress = min(100, max(0, float(data["progress"]) * 100))
    from .epoch_health import epoch_health
    rows = []
    for label, value in (
        ("Status", data["status"]), ("Epoch", f'{data["epoch"]}/{data["epochs"]}'),
        ("Batch", f'{data["batch"]}/{data["batches_per_epoch"]}'),
        ("Elapsed (s)", round(data["elapsed_seconds"], 1)),
        ("ETA (s)", None if data["eta_seconds"] is None else round(data["eta_seconds"], 1)),
        ("Batches/s", round(data["batches_per_second"], 2)),
    ):
        if overall_fields is None or label in overall_fields:
            rows.append((label, value))
    if show_metrics:
        rows.extend(data["metrics"].items())
    if show_metrics and data.get("epoch_history"):
        rows.extend(("Validation: " + k, v) for k,v in data["epoch_history"][-1]["metrics"].items())
    if show_resources:
        rows.extend(data.get("resources", {}).items())
    if show_artifacts:
        rows.extend(("Output: " + a["name"], a["bytes"]) for a in data["artifacts"])
    if show_diagnostics:
        rows.extend(("Diagnostic", item["code"]) for item in data.get("diagnostics", {}).get("findings", []))
    if show_metrics:
        rows.extend(("Batch class " + k, v) for k,v in data.get("batch_distribution", {}).get("counts", {}).items())
    encoded = data.get("preview_png")
    preview = ""
    if show_preview and isinstance(encoded, str) and len(encoded) <= 800000 and encoded.startswith("iVBORw0KGgo"):
        preview = f'<img alt="Opt-in batch preview" style="max-width:100%" src="data:image/png;base64,{esc(encoded)}">'
    def fmt(value):
        return f"{value:.4f}" if isinstance(value, float) else esc(value)

    def bar(percent, color='#24765c'):
        return (f'<div role="progressbar" aria-valuenow="{percent:.1f}" aria-valuemin="0" aria-valuemax="100" '
                'style="background:#dddcd3;border-radius:8px;height:9px;overflow:hidden">'
                f'<div style="background:{color};width:{percent:.2f}%;height:9px"></div></div>')

    records = data.get("epoch_records", {})
    current = max(1, data["epoch"])
    last = current
    first = 1 if max_epochs is None else max(1, current - max_epochs + 1)
    cards = []
    for epoch in range(first, last + 1) if show_epochs else ():
        done = max(0, min(data["batches_per_epoch"], data["completed_batches"] - (epoch - 1) * data["batches_per_epoch"]))
        percent = 100 * done / data["batches_per_epoch"]
        record = records.get(str(epoch), {})
        state = "Training complete" if percent == 100 else ("Running" if done else "Pending")
        metrics = record.get("training", data["metrics"] if epoch == current else {})
        if record.get("validation"):
            metrics = {**metrics, **{"val_" + k: v for k, v in record["validation"].items()}}
        health, color = epoch_health(records, epoch, patience=alert_patience, threshold=alert_threshold, gradient_threshold=gradient_threshold) if percent == 100 and show_diagnostics else (state, '#305070' if done else '#808080')
        if epoch_metrics is not None:
            metrics = {k:v for k,v in metrics.items() if k in epoch_metrics}
        details = " · ".join(f"{esc(k)}: {fmt(v)}" for k, v in metrics.items())
        stages = []
        if show_stages:
            for stage in ("validation", "checkpoint", "verification"):
                item = record.get("stages", {}).get(stage, {})
                status = item.get("status", "Not reported")
                count, total = item.get("completed"), item.get("total")
                meter = bar(100 * count / total) if total else (bar(100) if status == "completed" else "")
                stages.append(f'<div style="margin-top:8px">{stage.title()}: {esc(status)}{meter}</div>')
        cards.append(f'<article style="background:{panel};color:{fg};border:1px solid {color};border-radius:12px;padding:12px;margin-top:10px">'
                     f'<strong>Epoch {epoch} / {data["epochs"]}</strong> · {health}{bar(percent, color)}'
                     f'<div>{done}/{data["batches_per_epoch"]} batches · {percent:.1f}%</div>'
                     f'<details{" open" if epoch == current else ""}><summary>Details</summary><div style="overflow-wrap:anywhere">{details}</div>{"".join(stages)}</details></article>')
    warning = ''
    health, color = epoch_health(records, current, patience=alert_patience, threshold=alert_threshold, gradient_threshold=gradient_threshold)
    if show_diagnostics and health in ('Severe overfitting warning','Possible overfitting','Watch','Nonfinite metric','Gradient explosion'):
        warning = f'<div role="alert" style="border:2px solid {color};padding:12px;margin:8px 0">{health}. Review recorded metrics before continuing.</div>'
    archive = ""
    if show_history and first > 1:
        archive = f'<p>{first-1} earlier epochs retained in the training log and snapshot.</p>'
    hardware = ""
    if show_resources:
        meters = []
        for key, label in (("cpu_percent","CPU"),("ram_percent","RAM"),("cuda_memory_percent","VRAM"),("gpu_utilization_percent","GPU compute"),("disk_percent","Disk")):
            value = data.get("resources", {}).get(key)
            if isinstance(value, (int,float)) and math.isfinite(value):
                meters.append(f"<div>{label}: {value:.1f}%{bar(max(0,min(100,value)))}</div>")
        hardware = "".join(meters)
    overall = f"<strong>Overall training progress</strong> {progress:.1f}%{bar(progress, '#E0C58F')}" if show_overall else ""
    charts = []
    if show_charts:
        for metric in ("loss", "accuracy"):
            series = []
            for kind, color in (("training", "#24765c"), ("validation", "#b77924")):
                points = []
                for key, record in sorted(records.items(), key=lambda item: int(item[0])):
                    value = record.get(kind, {}).get(metric)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        points.append((int(key), value))
                if points:
                    series.append((kind, color, points[-500:]))
            if not series:
                continue
            values = [v for _, _, points in series for _, v in points]
            low, high = min(values), max(values)
            span = high - low or 1
            start = min(e for _, _, points in series for e, _ in points)
            end = max(e for _, _, points in series for e, _ in points)
            lines = []
            for kind, color, points in series:
                coords = [(12 + 276 * (e-start)/max(1,end-start), 90 - 78*(v-low)/span) for e,v in points]
                xy = " ".join(f"{x:.2f},{y:.2f}" for x,y in coords)
                lines.append(f'<polyline points="{xy}" fill="none" stroke="{color}" stroke-width="2"/>')
                lines.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="{color}"/>' for x,y in coords)
            charts.append(f'<div><strong>{metric.title()} per epoch</strong><svg viewBox="0 0 300 100" role="img" aria-label="{metric} history" style="width:100%;max-width:600px">{"".join(lines)}</svg><div>Training (green) · Validation (gold) · Range {low:.4f}–{high:.4f}</div></div>')
    table = "".join(f"<tr><th>{esc(k)}</th><td>{fmt(v)}</td></tr>" for k,v in rows)
    header_bg = "#123B32" if theme == "jade" else fg
    header_fg = "#ffffff" if theme != "dark" else bg
    summary = " · ".join(f"{esc(k)}: {fmt(v)}" for k,v in rows if k in ("Status","Epoch","Batch","Elapsed (s)","ETA (s)","Batches/s"))
    scroll_style = f"max-height:{dashboard_height}px;overflow-y:auto;overscroll-behavior:contain" if fixed_header else ""
    return (f'<section style="background:{bg};color:{fg};padding:16px;border-radius:12px">'
            f'<header style="background:{header_bg};color:{header_fg};padding:12px;border-radius:12px"><h2 style="color:{header_fg};margin:0 0 12px">DeepForge · XVARNA</h2>'
            f'{overall}<div>{summary}</div></header><div style="{scroll_style}" role="region" aria-label="Training details" tabindex="0">'
            f'{warning}{archive}{"".join(cards)}{hardware}{"".join(charts)}<table style="color:{fg};overflow-wrap:anywhere;width:100%">{table}</table>{preview}</div></section>')


def display_snapshot(path, *, display_id=None):
    """Display or update one notebook output. Caller controls refresh timing."""
    from IPython.display import HTML, display
    return display(HTML(snapshot_html(path)), display_id=display_id)


class NotebookMonitor:
    """Refresh a single inline output from an existing TrainingMonitor snapshot.

    Call refresh from the training thread after recording batches or epochs.
    No background thread, GUI, or browser server is required.
    """
    def __init__(self, path, *, interval=1.0, max_epochs=20, show_stages=True, show_charts=True, show_resources=True, theme='jade', show_history=True, show_epochs=True, show_preview=True, show_diagnostics=True, show_artifacts=True, show_metrics=True, show_overall=True, fixed_header=True, dashboard_height=480, epoch_metrics=None, overall_fields=None, alert_patience=3, alert_threshold=0.25, gradient_threshold=100.0, audio=False, widget_scroll=False, advanced=False, metadata=None):
        from .notebook_setup import AudioAlerts
        self.audio = AudioAlerts(audio)
        self.widget_scroll = widget_scroll
        self._widget = None
        self.advanced = advanced
        self.metadata = dict(metadata or {})
        if not math.isfinite(alert_threshold) or alert_threshold <= 0 or type(alert_patience) is not int or alert_patience < 1:
            raise ValueError('Invalid alert settings')
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("interval must be finite and nonnegative")
        if max_epochs is not None and (type(max_epochs) is not int or max_epochs < 2):
            raise ValueError("max_epochs must be None or an integer >= 2")
        if type(dashboard_height) is not int or dashboard_height < 120:
            raise ValueError('dashboard_height must be an integer >= 120')
        self.max_epochs = max_epochs
        self.show_stages = show_stages
        self.show_charts = show_charts
        self.show_resources = show_resources
        self.theme = theme
        self.sections = dict(show_history=show_history, show_epochs=show_epochs, show_preview=show_preview, show_diagnostics=show_diagnostics, show_artifacts=show_artifacts, show_metrics=show_metrics, show_overall=show_overall, fixed_header=fixed_header, dashboard_height=dashboard_height, epoch_metrics=epoch_metrics, overall_fields=overall_fields, alert_patience=alert_patience, alert_threshold=alert_threshold, gradient_threshold=gradient_threshold)
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
        options = dict(self.sections)
        if self.widget_scroll:
            options['fixed_header'] = False
        render_options = dict(max_epochs=self.max_epochs, show_stages=self.show_stages, show_charts=self.show_charts, show_resources=self.show_resources, theme=self.theme, **options)
        if self.advanced:
            from .dashboard import dashboard_html, TABS
            if self.widget_scroll:
                import ipywidgets as w
                if self._widget is None:
                    self._tab_widgets = [w.HTML() for _ in TABS]
                    self._widget = w.Tab(children=self._tab_widgets)
                    for index, title in enumerate(TABS):
                        self._widget.set_title(index, title)
                    display(self._widget)
                for title, widget in zip(TABS, self._tab_widgets):
                    widget.value = dashboard_html(self.path, tab=title, metadata=self.metadata, **{**render_options,'fixed_header':self.sections['fixed_header']})
                markup = None
            else:
                markup = dashboard_html(self.path, metadata=self.metadata, **render_options)
        else:
            markup = snapshot_html(self.path, **render_options)
        output = HTML(markup) if markup is not None else None
        if self.advanced and self.widget_scroll:
            pass
        elif self.widget_scroll:
            import ipywidgets as w
            header, body = markup.split('</header>', 1)
            if self._widget is None:
                self._header_widget = w.HTML(value=header+'</header></section>')
                self._body_widget = w.HTML(value='<section>'+body)
                self._scroll_widget = w.VBox([self._body_widget], layout=w.Layout(max_height=f"{self.sections['dashboard_height']}px", overflow='auto'))
                self._widget = w.VBox([self._header_widget,self._scroll_widget])
                display(self._widget)
            else:
                self._header_widget.value = header+'</header></section>'
                self._body_widget.value = '<section>'+body
        elif self._handle is None:
            self._handle = display(output, display_id=True)
        else:
            self._handle.update(output)
        if self.audio.enabled:
            from .epoch_health import epoch_health
            data = json.loads(self.path.read_text(encoding='utf-8'))
            health, _ = epoch_health(data.get('epoch_records', {}), data['epoch'], patience=self.sections['alert_patience'], threshold=self.sections['alert_threshold'], gradient_threshold=self.sections['gradient_threshold'])
            self.audio.notify(data['epoch'], health)
        self._last = now
        return True
