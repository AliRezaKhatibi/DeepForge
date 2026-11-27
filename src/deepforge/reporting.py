"""Portable offline report export, with escaped data and explicit provenance."""
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from . import __version__


def _e(value):
    return html.escape(str(value), quote=True)


def table_rows(report):
    """Flatten primary values for a concise preview; raw JSON remains authoritative."""
    rows = []
    def walk(prefix, value, depth=0):
        if isinstance(value, dict) and depth < 3:
            for k, v in value.items():
                walk(f'{prefix} / {k}' if prefix else k, v, depth+1)
        elif isinstance(value, (list, dict)):
            rows.append((prefix, f'{len(value):,} entries (see JSON)'))
        elif isinstance(value, float):
            rows.append((prefix, f'{value:.7g}'))
        else:
            rows.append((prefix, str(value)))
    walk('', report)
    return rows


def finding_rows(report, limit=200):
    """Show concrete audit findings before summary fields, with a documented cap."""
    rows = []
    fields = ('duplicates', 'candidate_pairs', 'errors', 'overlaps', 'missing_labels',
              'orphan_labels', 'ambiguous_image_stems', 'issues', 'ranking', 'rejected', 'calls')
    for key in fields:
        value = report.get(key)
        if not isinstance(value, list):
            continue
        for i, item in enumerate(value, 1):
            if len(rows) == limit:
                rows.append(('Additional findings', 'Preview limited; inspect the complete JSON report.'))
                return rows
            if key == 'ranking':
                display = f"{item['run']} | best {item['best_value']:.6g} at epoch {item['best_epoch']:g}"
            elif isinstance(item, dict):
                display = json.dumps(item, ensure_ascii=False, allow_nan=False)
            else:
                display = str(item)
            rows.append((f"{key.replace('_', ' ').title()} / {i}", display))
    return rows


def chart_svg(chart):
    """Create a static, self-contained SVG chart for reports, without external assets."""
    if not chart:
        return ''
    w,h=820,350
    x0,y0,x1,y1=110,32,770,295
    colors=('#12664f','#8a651e','#476f9a','#a3353d')
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="Analysis chart" style="width:100%;height:auto;background:white;border-radius:8px">']
    def line(x,y,xx,yy,stroke='#d8e0e7',width=1,dash=''):
        parts.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{xx:.2f}" y2="{yy:.2f}" stroke="{stroke}" stroke-width="{width}"{dash}/>')
    def text(x,y,value,anchor='middle',size=12,color='#394c63'):
        parts.append(f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-family="system-ui,sans-serif" font-size="{size}" fill="{color}">{_e(value)}</text>')
    kind=chart.get('type')
    if kind=='bars':
        values=chart['values'][:12]; lo=min([0]+[v for _,v in values]); hi=max([0]+[v for _,v in values]); span=hi-lo or 1
        x0=180; zero=x0+(0-lo)/span*(x1-x0); step=(y1-y0)/max(1,len(values))
        for i in range(5):
            x=x0+(x1-x0)*i/4;line(x,y0,x,y1);text(x,y1+22,f'{lo+span*i/4:.3g}')
        for i,(label,value) in enumerate(values):
            yy=y0+(i+.5)*step;xx=x0+(value-lo)/span*(x1-x0);text(x0-8,yy+4,str(label)[:25],'end');parts.append(f'<rect x="{min(zero,xx):.2f}" y="{yy-min(12,step*.3):.2f}" width="{abs(xx-zero):.2f}" height="{min(24,step*.6):.2f}" rx="2" fill="{colors[i%4]}"/>');text(xx+5 if value>=0 else xx-5,yy+4,f'{value:.4g}','start' if value>=0 else 'end')
    elif kind in ('lines','roc','probability'):
        xs=chart['x']; series=chart['series'] if kind=='lines' else {chart.get('title','ROC'):chart['y']}; ys=[v for vs in series.values() for v in vs]
        lo,hi=(0,1) if kind in ('roc','probability') else (min(ys),max(ys)); xmin,xmax=(0,1) if kind in ('roc','probability') else (xs[0],xs[-1]); span=hi-lo or 1; dx=xmax-xmin or 1
        for i in range(5):
            y=y1-(y1-y0)*i/4;line(x0,y,x1,y);text(x0-9,y+4,f'{lo+span*i/4:.3g}','end');x=x0+(x1-x0)*i/4;text(x,y1+20,f'{xmin+dx*i/4:.3g}')
        if kind=='roc' or (kind=='probability' and chart.get('baseline') is None): line(x0,y1,x1,y0,'#98a6b6',1,' stroke-dasharray="5 5"')
        elif kind=='probability':
            yy=y1-chart['baseline']*(y1-y0);line(x0,yy,x1,yy,'#98a6b6',1,' stroke-dasharray="5 5"')
        for i,(label,vs) in enumerate(series.items()):
            points=' '.join(f'{x0+(x-xmin)/dx*(x1-x0):.2f},{y1-(v-lo)/span*(y1-y0):.2f}' for x,v in zip(xs,vs));parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[i%4]}" stroke-width="2.7"/>');text(x0+i*160,19,label,'start',color=colors[i%4])
        text((x0+x1)/2,335,'Epoch' if kind=='lines' else chart.get('xlabel','False positive rate'))
    elif kind=='matrix':
        labels=chart['labels'];matrix=chart['values'];n=len(labels)
        if n>20 or n==0:return ''
        cell=min(240/n,440/n);start=(w-n*cell)/2;top=60;maxv=max(max(r) for r in matrix) or 1
        text(w/2,16,'Confusion matrix · rows = true · columns = predicted')
        for i,label in enumerate(labels):
            text(start-8,top+(i+.5)*cell+4,str(label)[:15],'end',10);text(start+(i+.5)*cell,top-10,str(label)[:15],size=10)
            for j,value in enumerate(matrix[i]):
                t=value/maxv;x=start+j*cell;y=top+i*cell;fill=f'#{int(232-170*t):02x}{int(241-100*t):02x}{int(243-95*t):02x}'
                parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell:.2f}" height="{cell:.2f}" fill="{fill}" stroke="white"/>')
                if cell>=19:text(x+cell/2,y+cell/2+4,value,color='white' if t>.65 else '#183c33')
    else:return ''
    parts.append('</svg>')
    return ''.join(parts)


def export_report(path, tool, report, *, cards=(), fmt=None, chart=None):
    path = Path(path)
    fmt = fmt or path.suffix.lower().lstrip('.')
    if fmt == 'json':
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
        return
    if fmt != 'html':
        raise ValueError('Use JSON or HTML export')
    raw = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    top = ''.join(f'<article><small>{_e(k)}</small><strong>{_e(v)}</strong></article>' for k,v in cards)
    rows = ''.join(f'<tr><th scope="row">{_e(k)}</th><td>{_e(v)}</td></tr>' for k,v in finding_rows(report)+table_rows(report))
    doc = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DeepForge | {_e(tool)}</title>
<style>body{{font:16px system-ui,sans-serif;background:#f3efe5;color:#183c33;margin:0}}main{{max-width:1050px;margin:0 auto;padding:44px 24px}}header{{border-bottom:2px solid #8a651e;padding-bottom:20px}}small{{color:#536b60}}h1{{font-size:36px;margin:8px 0}}section{{display:flex;flex-wrap:wrap;gap:12px;margin:24px 0}}article{{background:white;border:1px solid #dddcd0;border-radius:10px;padding:18px;flex:1;min-width:160px}}article strong{{display:block;font-size:24px;margin-top:8px;color:#12664f}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{text-align:left;padding:10px 14px;border-bottom:1px solid #e2e7ea;overflow-wrap:anywhere}}th{{width:45%;font-weight:500}}pre{{background:#123b32;color:#f5f0e9;padding:20px;overflow:auto;border-radius:8px}}@media print{{body{{background:white}}main{{padding:0}}pre{{white-space:pre-wrap;color:#111;background:white}}}}</style>
<main><header><small>DEEPFORGE STUDIO · XVARNA · v{_e(__version__)} · LOCAL ANALYSIS</small><h1>{_e(tool)}</h1><small>Exported {_e(datetime.now(timezone.utc).isoformat())} · Review findings and sensitive paths before sharing.</small></header><section>{top}</section>{('<h2>Visualization</h2>'+chart_svg(chart)) if chart else ''}<h2>Results overview</h2><table>{rows}</table><h2>Complete report</h2><pre>{_e(raw)}</pre><footer><small>DeepForge Studio · Created by Alireza Khatibi · XVARNA</small></footer></main></html>'''
    path.write_text(doc, encoding='utf-8')
