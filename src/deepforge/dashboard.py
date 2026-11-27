"""Advanced local training dashboard; consumes recorded observations only."""
from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path

from .epoch_health import epoch_health

TABS = ("Training", "Metrics", "Hardware", "Model", "Dataset", "Reports")
ASSET = Path(__file__).with_name("assets") / "deepforge.svg"


def _esc(value):
    return html.escape(str(value), quote=True)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fmt(value):
    if value is None:
        return "Not reported"
    return f"{value:.4f}" if isinstance(value, float) else _esc(value)


def _bar(value, color="var(--df-accent)"):
    if not _number(value):
        return '<span class="df-muted">Not reported</span>'
    value = min(100, max(0, value))
    return (f'<div class="df-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" '
            f'aria-valuenow="{value:.2f}"><span style="width:{value:.2f}%;background:{color}"></span></div>')


def _card(title, body):
    return f'<section class="df-card"><h3>{_esc(title)}</h3>{body}</section>'


def _table(mapping):
    return '<dl class="df-list">' + ''.join(f'<dt>{_esc(k)}</dt><dd>{_fmt(v)}</dd>' for k, v in mapping.items()) + '</dl>' if mapping else '<p class="df-muted">Not reported</p>'


def _chart(records, metric):
    series = []
    for kind, color in (("training", "#38bdf8"), ("validation", "#f2b84b")):
        points = [(int(e), r.get(kind, {}).get(metric)) for e, r in records.items()]
        points = sorted((e, v) for e, v in points if _number(v))[-500:]
        if points:
            series.append((kind, color, points))
    if not series:
        return _card(metric.title() + " per epoch", '<p class="df-muted">Awaiting recorded data</p>')
    values = [v for _, _, pts in series for _, v in pts]
    epochs = [e for _, _, pts in series for e, _ in pts]
    low, high = min(values), max(values)
    first, last = min(epochs), max(epochs)
    span = high - low or 1
    lines = []
    for kind, color, pts in series:
        coords = [(48 + 310 * (e-first)/max(1,last-first), 128-100*(v-low)/span) for e,v in pts]
        lines.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in coords)}" stroke="{color}" stroke-width="2.5" fill="none"/>')
        lines.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"><title>Epoch {e}: {v:.4f}</title></circle>' for (x,y),(e,v) in zip(coords,pts))
    grid = ''.join(f'<path d="M48 {y}H358" stroke="currentColor" opacity=".12"/>' for y in (28,53,78,103,128))
    chart = (f'<svg viewBox="0 0 380 160" role="img" aria-label="{_esc(metric)} history" style="width:100%;height:auto">{grid}'
             f'<g fill="currentColor" font-size="10"><text x="2" y="32">{high:.3g}</text><text x="2" y="130">{low:.3g}</text><text x="48" y="150">E{first}</text><text x="334" y="150">E{last}</text></g>{"".join(lines)}</svg>')
    return _card(metric.title() + " per epoch", chart + ' <span style="color:#38bdf8">● Training</span> · <span style="color:#f2b84b">● Validation</span>')


def _hardware(resources):
    parts = []
    for key, name, color in (("gpu_utilization_percent","GPU compute","#26e6a4"),("cuda_memory_percent","VRAM","#26e6a4"),("cpu_percent","CPU","#37bdef"),("ram_percent","RAM","#a78bfa"),("disk_percent","Disk","#f2b84b")):
        val = resources.get(key)
        parts.append(f'<p>{name} <strong>{f"{val:.1f}%" if _number(val) else "Not reported"}</strong>{_bar(val,color)}</p>')
    for used, total, name in (("cuda_used_bytes","cuda_total_bytes","VRAM"),("ram_used_bytes","ram_total_bytes","RAM")):
        if _number(resources.get(used)) and _number(resources.get(total)):
            parts.append(f'<p class="df-muted">{name}: {resources[used]/2**30:.2f} / {resources[total]/2**30:.2f} GiB</p>')
    return _card("Hardware monitor", f'<p>{_fmt(resources.get("cuda_name"))}</p>' + ''.join(parts))


CSS = '''<style>
.df-pro{font:14px system-ui,sans-serif;line-height:1.5;color:var(--df-text);background:var(--df-bg);border:1px solid var(--df-border);border-radius:16px;padding:14px;box-sizing:border-box;overflow-wrap:anywhere}
.df-pro *{box-sizing:border-box}.df-pro h2,.df-pro h3,.df-pro p{color:inherit;margin:0 0 9px}.df-pro h3{font-size:15px}.df-pro h2{font-size:24px}.df-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px}.df-head img{width:38px;height:42px}.df-muted{color:var(--df-muted)}.df-card{background:var(--df-panel);border:1px solid var(--df-border);border-radius:12px;padding:12px;min-width:0;margin-bottom:10px}.df-grid{display:grid;grid-template-columns:minmax(145px,1fr) minmax(0,3.4fr) minmax(150px,1.2fr);gap:10px}.df-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:8px}.df-charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px}.df-track{height:8px;border-radius:8px;background:var(--df-border);overflow:hidden;margin:6px 0 10px}.df-track span{display:block;height:100%;border-radius:8px}.df-badge{display:inline-block;border:1px solid var(--df-border);border-radius:8px;padding:4px 8px;margin:2px}.df-list{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:6px;margin:0}.df-list dd{margin:0;text-align:right}.df-list dt{color:var(--df-muted)}.df-footer{border-top:1px solid var(--df-border);padding-top:10px}.df-history{overflow-y:auto;overscroll-behavior:contain;overflow-anchor:auto}.df-pro summary{cursor:pointer}.df-pro a{color:var(--df-accent)}.df-pro strong{color:inherit}.df-strip{display:flex;flex-wrap:wrap;gap:4px}.df-pro input.df-tab{position:absolute;opacity:0;width:1px}.df-pro label.df-nav{display:inline-block;cursor:pointer;border:1px solid var(--df-border);border-radius:8px;padding:7px 12px;margin:2px}.df-pro input:focus-visible+label{outline:2px solid var(--df-accent)}.df-pro input:checked+label{background:var(--df-accent);color:#041e20}.df-panel{display:none}
@media(max-width:920px){.df-grid{grid-template-columns:1fr}.df-charts{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}}
</style>'''


def dashboard_html(path, *, tab=None, metadata=None, max_epochs=20, theme="jade", dashboard_height=480,
                   show_stages=True, show_charts=True, show_resources=True, show_history=True,
                   show_epochs=True, show_diagnostics=True, show_metrics=True, show_overall=True,
                   show_preview=True, show_artifacts=True, fixed_header=True, epoch_metrics=None,
                   overall_fields=None, alert_patience=3, alert_threshold=.25, gradient_threshold=100.0):
    """Render a dashboard without JS, network access, or invented observations."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported snapshot schema")
    if max_epochs is not None and (type(max_epochs) is not int or max_epochs < 2):
        raise ValueError("max_epochs must be None or an integer >= 2")
    if tab is not None and tab not in TABS:
        raise ValueError("Unknown dashboard tab")
    palettes = {"jade":("#05161c","#0c242b","#20434a","#e9f8fa","#a0b9bf","#21e2ac"),"dark":("#121b24","#1b2936","#354759","#f1f5f9","#aabacb","#38bdf8"),"light":("#f3f8f6","#ffffff","#c4d8d0","#173c36","#46665e","#07966c")}
    if theme not in palettes:
        raise ValueError("Unknown theme")
    style = ';'.join(f'--df-{k}:{v}' for k,v in zip(('bg','panel','border','text','muted','accent'),palettes[theme]))
    meta = metadata or {}
    records = data.get("epoch_records", {})
    current = data['epoch']
    train = data.get('metrics', {})
    latest_validation = (data.get('epoch_history') or [{}])[-1]
    validation = latest_validation.get('metrics', {})
    resources = data.get('resources', {})
    health, color = epoch_health(records,current,patience=alert_patience,threshold=alert_threshold,gradient_threshold=gradient_threshold)
    readiness = meta.get('readiness', {})
    checks = ('Configuration','Dataset (Train / Val)','Model architecture','Optimizer & scheduler','Loss function','Metrics','Checkpoint settings','Hardware')
    checklist = ''.join(f'<p><span style="color:{"var(--df-accent)" if readiness.get(k) is True else "var(--df-muted)"}">{"✓ Verified" if readiness.get(k) is True else "○ Unassessed"}</span><br>{_esc(k)}</p>' for k in checks)
    status = _esc(data['status'].title())
    header = f'<div class="df-head"><img alt="DeepForge official symbol" src="data:image/svg+xml;base64,{base64.b64encode(ASSET.read_bytes()).decode("ascii")}"><div><h2>DeepForge</h2><span class="df-muted">Training Monitor · XVARNA &amp; Bushehr Machine Vision</span></div><span class="df-badge">{status}</span></div>'
    epoch_strip = ''.join(f'<span class="df-badge">E{e} {"✓" if records.get(str(e),{}).get("validation") else "·" if e==current else ""}</span>' for e in range(max(1,current-3),min(data['epochs'],max(1,current)+3)+1))
    summary = f'<h2>Epoch {current} / {data["epochs"]}</h2><div class="df-strip">{epoch_strip}</div><p>Batch {data["batch"]} / {data["batches_per_epoch"]}</p>{_bar(100*data["batch"]/data["batches_per_epoch"])}'
    if show_overall:
        summary += f'<p>Overall training progress · {100*data["progress"]:.1f}%</p>{_bar(100*data["progress"])}'
    overall = {'Elapsed (s)':data['elapsed_seconds'],'ETA (s)':data.get('eta_seconds'),'Batches/s':data['batches_per_second']}
    summary += _table({k:v for k,v in overall.items() if overall_fields is None or k in overall_fields})
    metric_values = {'Train loss':train.get('loss'),'Val loss':validation.get('loss'),'Val accuracy':validation.get('accuracy'),'Val precision':validation.get('precision'),'Val recall':validation.get('recall'),'Val F1 score':validation.get('f1')}
    cards = '<div class="df-metrics">'+''.join(_card(k,f'<strong style="font-size:19px">{_fmt(v)}</strong>') for k,v in metric_values.items())+'</div>' if show_metrics else ''
    if show_metrics:
        cards += f'<p class="df-muted">Validation: {"epoch " + str(latest_validation["epoch"]) if latest_validation else "not reported"} · Metric units are caller-defined.</p>'
    charts = '<div class="df-charts">'+_chart(records,'loss')+_chart(records,'accuracy')+'</div>' if show_charts else ''
    hardware = _hardware(resources) if show_resources else ''
    diagnostics = _card('Training health',f'<p style="color:{color}">{_esc(health)}</p><p>Gradient norm: {_fmt(train.get("gradient_norm"))}</p>') if show_diagnostics else ''
    epoch_cards=[]
    first = 1 if max_epochs is None else max(1,current-max_epochs+1)
    if show_epochs:
        for e in range(first,current+1):
            r=records.get(str(e),{}); done=max(0,min(data['batches_per_epoch'],data['completed_batches']-(e-1)*data['batches_per_epoch']))
            metrics={**r.get('training',train if e==current else {}),**{'val_'+k:v for k,v in r.get('validation',{}).items()}}
            if epoch_metrics is not None: metrics={k:v for k,v in metrics.items() if k in epoch_metrics}
            stages=''.join(f'<p>{s.title()}: {_esc(r.get("stages",{}).get(s,{}).get("status","Not reported"))}</p>' for s in ('validation','checkpoint','verification')) if show_stages else ''
            epoch_cards.append(_card(f'Epoch {e} / {data["epochs"]}',f'<p>{done}/{data["batches_per_epoch"]} batches</p>{_bar(100*done/data["batches_per_epoch"])}<details><summary>Details</summary>{_table(metrics)}{stages}</details>'))
    history = f'<div class="df-history" style="{"max-height:"+str(dashboard_height)+"px" if fixed_header else ""}" tabindex="0" aria-label="Epoch history">{"".join(epoch_cards)}</div>' if show_epochs else ''
    if first>1 and history and show_history: history=f'<p>{first-1} earlier epochs retained in the archive.</p>'+history
    preview=data.get('preview_png'); preview_html=f'<img alt="Opt-in batch preview" style="max-width:100%" src="data:image/png;base64,{_esc(preview)}">' if show_preview and isinstance(preview,str) and len(preview)<=800000 and preview.startswith('iVBORw0KGgo') else ''
    footer=_card('Run',_table({'Run ID':meta.get('run_id'),'Model':meta.get('model'),'Parameters':meta.get('parameters'),'Device':meta.get('device')}))
    panels={
        'Training':f'<div class="df-grid"><aside>{_card("Training checklist",checklist)}</aside><main>{_card("Training progress",summary)}{cards}{charts}</main><aside>{hardware}{diagnostics}</aside></div>{history}',
        'Metrics':cards+charts+(_card('Latest training metrics',_table(train))+_card('Latest validation metrics',_table(validation)) if show_metrics else ''),
        'Hardware':hardware or _card('Hardware','<p>Resource display disabled.</p>'),
        'Model':_card('Model',_table({k:meta.get(k) for k in ('model','parameters','trainable_parameters','device')})),
        'Dataset':_card('Dataset',_table(meta.get('dataset',{})))+_card('Batch distribution',_table(data.get('batch_distribution',{}).get('counts',{})))+preview_html,
        'Reports':(_card('Recorded artifacts',_table({a['name']:a.get('bytes') for a in data.get('artifacts',[])})) if show_artifacts else '')+_card('Snapshot',_table({'Snapshot':Path(path).name}))+(_card('Diagnostic findings',_table({str(i+1):x.get('code') for i,x in enumerate(data.get('diagnostics',{}).get('findings',[]))})) if show_diagnostics else ''),
    }
    if tab is None:
        import uuid
        uid='df-'+uuid.uuid4().hex
        nav=''; rules=''; body=''
        for i,name in enumerate(TABS):
            nav+=f'<input class="df-tab" type="radio" name="{uid}" id="{uid}-{i}"{" checked" if i==0 else ""}><label class="df-nav" for="{uid}-{i}">{name}</label>'
            rules+=f'#{uid}-{i}:checked~.df-panel-{i}{{display:block}}'
            body+=f'<div class="df-panel df-panel-{i}">{panels[name]}</div>'
        content=f'<style>{rules}</style>{nav}{body}'
    else:
        content=panels[tab]
    return CSS+f'<section class="df-pro" style="{style}">{header}{content}<footer class="df-footer">{footer}</footer></section>'
