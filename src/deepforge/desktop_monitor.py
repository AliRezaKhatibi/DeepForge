"""Optional independent native desktop monitor for local DeepForge training runs."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from .checkpoint import atomic_json
from .control import StopControl
from .epoch_health import epoch_health

TABS = ("Training", "Metrics", "Hardware", "Model", "Dataset", "Reports")
ASSETS = Path(__file__).with_name("assets")
PALETTES = {
    "jade": ("#04151b", "#0a2027", "#102b32", "#24444b", "#effbfc", "#96b5bc", "#20e6ae"),
    "dark": ("#101720", "#18232e", "#243443", "#3c5362", "#ecf3f8", "#a7b9c7", "#35c8ef"),
    "light": ("#eef5f2", "#ffffff", "#f0f7f4", "#c5d9d2", "#173c36", "#526e67", "#078561"),
}
CHECKS = ("Configuration", "Dataset (Train / Val)", "Model architecture", "Optimizer & scheduler",
          "Loss function", "Metrics", "Checkpoint settings", "Hardware (GPU / CPU)")


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def display_value(value):
    if value is None:
        return "Not reported"
    if isinstance(value, float):
        return f"{value:.4f}" if numeric(value) else "Nonfinite"
    return str(value)


def duration(value):
    if not numeric(value) or value < 0:
        return "Not reported"
    seconds = int(value)
    return f"{seconds//3600:02d}:{(seconds//60)%60:02d}:{seconds%60:02d}"


def read_json(path, default=None):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else default
    except (OSError, ValueError):
        return default


def launch_desktop(snapshot, *, theme="jade", interval=0.5):
    """Launch the native window independently; no server, browser, or notebook callback."""
    if theme not in PALETTES or not numeric(interval) or not 0.1 <= interval <= 10:
        raise ValueError("Invalid desktop theme or refresh interval")
    path = Path(snapshot).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return subprocess.Popen([sys.executable, "-m", "deepforge.desktop_monitor", str(path),
                             "--theme", theme, "--interval", str(interval)])


class DesktopChannel:
    """Local, explicitly enabled cooperative control; read-only for ordinary snapshots."""
    def __init__(self, snapshot):
        self.path = Path(snapshot)
        self.root = self.path.parent

    @property
    def metadata(self):
        return read_json(self.root / "desktop.json", {})

    @property
    def state(self):
        return read_json(self.root / "desktop-status.json", {})

    def command(self, action, *, after_next=True):
        if not self.metadata.get("controls"):
            raise RuntimeError("This is a read-only monitor. Open it from a TrainingSession to enable controls.")
        data = read_json(self.path, {})
        if action == "resume":
            if data.get("status") != "paused" or not self.metadata.get("allow_resume"):
                raise RuntimeError("Resume is not enabled for this paused training session.")
            if data.get("epoch",0) >= data.get("epochs",0):
                raise RuntimeError("No training epochs remain.")
            atomic_json(self.root / "resume-request.json", {"requested": True})
            return
        if data.get("status") != "running":
            raise RuntimeError("The training session is not running.")
        control = StopControl(self.root / "control.json")
        if action == "stop":
            control.request(after_next=after_next)
        elif action == "cancel":
            control.cancel()
        elif action == "save":
            atomic_json(self.root / "manual-save.json", {"requested": True})
        else:
            raise ValueError("Unknown desktop command")

    def feedback(self, data):
        if data.get("status") == "paused":
            if data.get("epoch",0) >= data.get("epochs",0):
                return "Paused at the final epoch · no training epochs remain."
            if (self.root / "resume-request.json").exists():
                return "Resume requested · awaiting the caller's restore and training loop."
            if self.metadata.get("allow_resume"):
                return "Paused · Resume requests are handled by the waiting training script."
            return "Paused · Restore this run in the training script or notebook to resume."
        if data.get("status") != "running":
            return "Training " + str(data.get("status", "unavailable"))
        pending = read_json(self.root / "control.json", {})
        if pending:
            return "Control queued · awaiting the next epoch boundary"
        if self.state.get("target_epoch") is not None:
            return f"Safe stop scheduled after epoch {self.state['target_epoch']}"
        if any((self.root / n).exists() for n in ("manual-save.json", "manual-save.processing.json")):
            return "Checkpoint requested · awaiting epoch boundary and verification"
        if self.state.get("saved_epoch") is not None:
            return f"Checkpoint saved at epoch {self.state['saved_epoch']}"
        if not self.metadata.get("controls"):
            return "Read-only snapshot · use TrainingSession.open_desktop() for controls"
        return "Live · training remains in your Python process"


def observations(data, metadata):
    """Build labels from recorded data; missing values never become zero or healthy."""
    records = data.get("epoch_records", {})
    latest = (data.get("epoch_history") or [{}])[-1]
    train, val = data.get("metrics", {}), latest.get("metrics", {})
    values = [("Train Loss", train.get("loss")), ("Val Loss", val.get("loss")),
              ("Accuracy", val.get("accuracy")), ("Precision", val.get("precision")),
              ("Recall", val.get("recall")), ("F1 Score", val.get("f1"))]
    checklist = dict(metadata.get("readiness", {}))
    if train or val:
        checklist["Metrics"] = True
    if data.get("resources"):
        checklist["Hardware (GPU / CPU)"] = True
    health, color = epoch_health(records, latest.get("epoch", 0),
        patience=metadata.get("config", {}).get("alert_patience", 3),
        gradient_threshold=metadata.get("config", {}).get("gradient_threshold", 100.0))
    gradient = train.get("gradient_norm")
    if any(v is None for v in train.values()):
        health, color = "Nonfinite metric", "#ef5350"
    elif numeric(gradient) and gradient > metadata.get("config", {}).get("gradient_threshold",100.0):
        health, color = "Gradient explosion", "#ef5350"
    return values, checklist, health, color, latest.get("epoch")


def series_for(data, metric):
    records = data.get("epoch_records", {})
    return [(name, color, sorted((int(e), r[side][metric]) for e, r in records.items()
                   if numeric(r.get(side, {}).get(metric)))[-500:])
            for name, color, side in (("Train", "#39b9f0", "training"), ("Validation", "#20e6ae" if metric=="accuracy" else "#f0a44b", "validation"))]


def model_metadata(model):
    """Inspect registered parameters/modules without running a forward pass."""
    from itertools import islice
    result={"model":type(model).__name__}
    try:
        params=list(model.parameters())
        result.update(parameters=sum(p.numel() for p in params),
                      trainable_parameters=sum(p.numel() for p in params if p.requires_grad),
                      device=str(params[0].device) if params else None)
        result["layers"]=[{"name":name or "(root)","type":type(layer).__name__,
                           "parameters":sum(p.numel() for p in layer.parameters(recurse=False))}
                          for name,layer in islice(model.named_modules(),100)]
    except (AttributeError,ValueError,RuntimeError):
        pass
    return result


def export_report(path, data, *, csv_format=False):
    """Export only the recorded snapshot or the recorded epoch metrics."""
    if not csv_format:
        atomic_json(Path(path),data);return
    import csv
    records=data.get("epoch_records",{}); fields=sorted({f"{side}_{key}" for record in records.values()
        for side in ("training","validation") for key in record.get(side,{})})
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=["epoch",*fields]);writer.writeheader()
        for epoch,record in sorted(records.items(),key=lambda item:int(item[0])):
            writer.writerow({"epoch":epoch,**{f"{side}_{key}":value for side in ("training","validation") for key,value in record.get(side,{}).items()}})


class DesktopMonitor:
    """Native Tk dashboard with a fixed toolbar/footer and independently scrolling body."""
    def __init__(self, root, snapshot, *, theme="jade", interval=.5):
        import tkinter as tk
        self.tk, self.root, self.channel = tk, root, DesktopChannel(snapshot)
        self.theme = theme
        self.interval = max(100, int(interval*1000))
        self.tab, self.data, self.metadata = "Training", {}, {}
        self.signature = None
        self._size_job = None
        self._timer = None
        self.history_limit = 20
        self.after_next = tk.BooleanVar(value=True)
        root.title("DeepForge | Training Monitor")
        root.geometry(f"{min(1440,max(640,root.winfo_screenwidth()-40))}x{min(880,max(480,root.winfo_screenheight()-80))}")
        root.minsize(640,480)
        self.logo = tk.PhotoImage(file=str(ASSETS / "deepforge.png"))
        root.iconphoto(True, self.logo)
        self.small_logo = self.logo.subsample(12,12)
        self._build()
        self.refresh()
        root.protocol("WM_DELETE_WINDOW", self.close)

    def close(self):
        if self._timer is not None:
            self.root.after_cancel(self._timer)
        if self._size_job is not None:
            self.root.after_cancel(self._size_job)
        self.root.destroy()

    def _palette(self):
        self.bg,self.panel,self.panel2,self.border,self.fg,self.muted,self.accent = PALETTES[self.theme]

    def _button(self, parent, label, command, *, color=None):
        return self.tk.Button(parent, text=label, command=command, bg=self.panel2, fg=color or self.fg,
            activebackground=self.accent, activeforeground=self.bg, relief="flat", bd=0,
            highlightthickness=1, highlightbackground=self.border, cursor="hand2", padx=10, pady=7,
            font=("TkDefaultFont", 10, "bold"))

    def _build(self):
        tk=self.tk; self._palette(); self.root.configure(bg=self.bg)
        self.header=tk.Frame(self.root,bg=self.bg); self.header.pack(fill="x",padx=14,pady=(12,4))
        tk.Label(self.header,image=self.small_logo,bg=self.bg).pack(side="left",padx=(0,8))
        brand=tk.Frame(self.header,bg=self.bg); brand.pack(side="left",padx=(0,16))
        tk.Label(brand,text="DeepForge",bg=self.bg,fg=self.fg,font=("TkDefaultFont",23,"bold")).pack(anchor="w")
        tk.Label(brand,text="TRAINING MONITOR",bg=self.bg,fg=self.accent,font=("TkDefaultFont",9,"bold")).pack(anchor="w")
        self.badge=tk.Label(self.header,text="WAITING",bg=self.panel2,fg=self.accent,padx=12,pady=7,font=("TkDefaultFont",10,"bold")); self.badge.pack(side="right")
        self._button(self.header,"Theme",self.cycle_theme).pack(side="right",padx=8)
        nav=tk.Frame(self.root,bg=self.bg); nav.pack(fill="x",padx=14,pady=(0,7))
        self.nav={}
        for i,name in enumerate(TABS):
            nav.columnconfigure(i,weight=1,uniform="nav")
            b=self._button(nav,name,lambda n=name:self.select(n)); b.grid(row=0,column=i,sticky="ew",padx=2); self.nav[name]=b
        self.footer=tk.Frame(self.root,bg=self.panel,highlightbackground=self.border,highlightthickness=1); self.footer.pack(side="bottom",fill="x",padx=14,pady=(5,12))
        self.info=tk.Label(self.footer,text="Waiting for snapshot",bg=self.panel,fg=self.fg,anchor="w",font=("TkDefaultFont",10,"bold")); self.info.pack(fill="x",padx=10,pady=(7,2))
        self.feedback=tk.Label(self.footer,text="",bg=self.panel,fg=self.muted,anchor="w",justify="left",wraplength=580); self.feedback.pack(fill="x",padx=10,pady=(0,3))
        buttons=tk.Frame(self.footer,bg=self.panel); buttons.pack(fill="x",padx=8,pady=(0,7))
        self.save=self._button(buttons,"Save Checkpoint",lambda:self.command("save"),color=self.accent); self.save.pack(side="right",padx=3)
        self.stop=self._button(buttons,"Stop Safely",lambda:self.command("stop"),color="#ff777b"); self.stop.pack(side="right",padx=3)
        self.cancel=self._button(buttons,"Cancel Stop",lambda:self.command("cancel")); self.cancel.pack(side="right",padx=3)
        self.resume=self._button(buttons,"Resume",lambda:self.command("resume")); self.resume.pack(side="right",padx=3)
        tk.Checkbutton(buttons,text="After next epoch",variable=self.after_next,bg=self.panel,fg=self.fg,selectcolor=self.panel2,activebackground=self.panel,activeforeground=self.fg).pack(side="right",padx=3)
        self.export_json=self._button(buttons,"Export JSON",lambda:self.export(False)); self.export_csv=self._button(buttons,"Export CSV",lambda:self.export(True))
        frame=tk.Frame(self.root,bg=self.bg); frame.pack(fill="both",expand=True,padx=14)
        self.canvas=tk.Canvas(frame,bg=self.bg,highlightthickness=0); self.canvas.pack(side="left",fill="both",expand=True)
        scrollbar=tk.Scrollbar(frame,orient="vertical",command=self.canvas.yview); scrollbar.pack(side="right",fill="y")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind("<Configure>",self._resized)
        self.canvas.bind("<MouseWheel>",lambda e:self.canvas.yview_scroll(-1 if e.delta>0 else 1,"units"))
        self.canvas.bind("<Button-4>",lambda e:self.canvas.yview_scroll(-1,"units")); self.canvas.bind("<Button-5>",lambda e:self.canvas.yview_scroll(1,"units"))
        self.select(self.tab)

    def cycle_theme(self):
        themes=list(PALETTES); self.theme=themes[(themes.index(self.theme)+1)%len(themes)]
        for widget in self.root.winfo_children(): widget.destroy()
        self._build(); self.update_footer(); self.draw()

    def _resized(self,event):
        if self._size_job is not None: self.root.after_cancel(self._size_job)
        self._size_job=self.root.after(70,self._redraw_after_resize)

    def _redraw_after_resize(self):
        self._size_job=None; self.draw()

    def select(self,tab):
        self.tab=tab
        for name,b in self.nav.items(): b.configure(bg=("#0b503e" if self.theme=="jade" else self.accent) if name==tab else self.panel2,fg=(self.accent if self.theme=="jade" else self.bg) if name==tab else self.fg,highlightbackground=self.accent if name==tab else self.border)
        for button in (self.export_json,self.export_csv):
            if tab=="Reports":button.pack(side="left",padx=3)
            else:button.pack_forget()
        self.draw(); self.canvas.yview_moveto(0)

    def export(self,csv_format):
        from tkinter import filedialog,messagebox
        if not self.data:return
        ext=".csv" if csv_format else ".json"
        path=filedialog.asksaveasfilename(parent=self.root,defaultextension=ext,initialfile="deepforge-report"+ext,filetypes=[("CSV" if csv_format else "JSON","*"+ext)])
        if not path:return
        try:export_report(path,self.data,csv_format=csv_format)
        except OSError as exc:messagebox.showerror("Export failed",str(exc),parent=self.root)
        else:messagebox.showinfo("Export complete",f"Saved {path}",parent=self.root)

    def command(self,action):
        from tkinter import messagebox
        if action=="stop" and not messagebox.askyesno("Safe stop","Finish the selected epoch, validate and save a verified recovery checkpoint before pausing?",parent=self.root): return
        try: self.channel.command(action,after_next=self.after_next.get())
        except (OSError,ValueError,RuntimeError) as exc: messagebox.showerror("DeepForge",str(exc),parent=self.root)
        self.update_footer()

    def refresh(self):
        data=read_json(self.channel.path,{})
        if data.get("schema_version")==1:
            metadata=self.channel.metadata
            old=(self.data,self.metadata)
            self.data,self.metadata=data,metadata
            if old!=(data,metadata): self.draw()
            self.update_footer()
        else:
            self.badge.configure(text="SNAPSHOT UNAVAILABLE",fg="#ef777b")
            for b in (self.stop,self.cancel,self.save,self.resume): b.configure(state="disabled")
        self._timer=self.root.after(self.interval,self.refresh)

    def update_footer(self):
        data=self.data; meta=self.metadata
        self.badge.configure(text=f"{str(data.get('status','waiting')).upper()}  ·  E{data.get('epoch',0)}/{data.get('epochs',0)}  ·  {data.get('progress',0):.1%}",fg=self.accent if data.get("status")=="running" else self.fg)
        self.info.configure(text=f"Run: {meta.get('run_id',self.channel.path.stem)}    |    Model: {display_value(meta.get('model'))}    |    Params: {display_value(meta.get('parameters'))}    |    Device: {display_value(meta.get('device'))}")
        self.feedback.configure(text=self.channel.feedback(data))
        state="normal" if meta.get("controls") and data.get("status")=="running" else "disabled"
        for b in (self.stop,self.cancel,self.save): b.configure(state=state)
        self.resume.configure(state="normal" if meta.get("allow_resume") and data.get("status")=="paused" and data.get("epoch",0)<data.get("epochs",0) else "disabled")

    def text(self,x,y,label,*,size=11,color=None,bold=False,anchor="nw",width=None):
        return self.canvas.create_text(x,y,text=str(label),fill=color or self.fg,font=("TkDefaultFont",size,"bold" if bold else "normal"),anchor=anchor,width=width or 0)

    def rounded(self,x,y,w,h,*,fill=None,outline=None,r=12):
        r=min(r,w/2,h/2); pts=[x+r,y,x+w-r,y,x+w,y,x+w,y+r,x+w,y+h-r,x+w,y+h,x+w-r,y+h,x+r,y+h,x,y+h,x,y+h-r,x,y+r,x,y]
        return self.canvas.create_polygon(pts,smooth=True,splinesteps=20,fill=fill or self.panel,outline=outline or self.border,width=1)

    def card(self,x,y,w,h,title):
        self.rounded(x,y,w,h); self.text(x+12,y+12,title,size=12,bold=True)

    def meter(self,x,y,w,value,color=None):
        self.rounded(x,y,w,9,fill=self.panel2,outline=self.panel2,r=4)
        if numeric(value) and value>0:
            self.rounded(x,y,max(1,w*min(100,max(0,value))/100),9,fill=color or self.accent,outline=color or self.accent,r=4)

    def chart(self,x,y,w,h,metric):
        self.card(x,y,w,h,metric.title()+" (Train vs Validation)")
        series=series_for(self.data,metric); points=[p for _,_,s in series for p in s]
        if not points:
            self.text(x+w/2,y+h/2,"Awaiting epoch data",color=self.muted,anchor="center"); return
        lo,hi=min(p[1] for p in points),max(p[1] for p in points)
        if lo==hi: lo-=max(abs(lo)*.1,.01); hi+=max(abs(hi)*.1,.01)
        a,b=min(p[0] for p in points),max(p[0] for p in points)
        l,t,r,bottom=x+47,y+62,x+w-14,y+h-32
        for i in range(5):
            gy=t+(bottom-t)*i/4
            self.canvas.create_line(l,gy,r,gy,fill=self.border)
            self.text(l-5,gy,f"{hi-(hi-lo)*i/4:.3g}",size=8,color=self.muted,anchor="e")
        for name,color,s in series:
            if not s: continue
            coords=[(l+(r-l)*(e-a)/max(1,b-a),bottom-(bottom-t)*(v-lo)/(hi-lo)) for e,v in s]
            if len(coords)>1:self.canvas.create_line(*[v for pt in coords for v in pt],fill=color,width=2.5)
            for cx,cy in coords:self.canvas.create_oval(cx-3,cy-3,cx+3,cy+3,fill=color,outline=color)
        self.text(l,bottom+8,f"E{a}",size=8,color=self.muted); self.text(r,bottom+8,f"E{b}",size=8,color=self.muted,anchor="ne")
        self.text(x+14,y+37,"● Train",size=9,color="#39b9f0"); self.text(x+100,y+37,"● Validation",size=9,color="#20e6ae" if metric=="accuracy" else "#f0a44b")

    def hardware(self,x,y,w):
        resources=self.data.get("resources",{}); self.card(x,y,w,302,"Hardware Monitor")
        self.text(x+12,y+38,resources.get("cuda_name","GPU not reported"),size=9,color=self.muted,width=w-24)
        yy=y+70
        for name,key,color in (("GPU compute","gpu_utilization_percent",self.accent),("VRAM","cuda_memory_percent",self.accent),("CPU","cpu_percent","#31bdf1"),("RAM","ram_percent","#af83ff")):
            v=resources.get(key)
            self.text(x+12,yy,name,size=10,bold=True); self.text(x+w-12,yy,f"{v:.1f}%" if numeric(v) else "Not reported",size=9,anchor="ne",color=self.muted)
            self.meter(x+12,yy+21,w-24,v,color)
            yy+=43
            prefix={"VRAM":"cuda","RAM":"ram"}.get(name)
            if prefix:
                used,total=resources.get(prefix+"_used_bytes"),resources.get(prefix+"_total_bytes")
                if numeric(used) and numeric(total):
                    self.text(x+12,yy-6,f"{used/2**30:.2f} / {total/2**30:.2f} GiB",size=8,color=self.muted); yy+=17
        return y+314

    def health(self,x,y,w,health,color):
        self.card(x,y,w,195,"Training Health")
        metrics=self.data.get("metrics",{});finite=bool(metrics) and all(numeric(v) for v in metrics.values());gradient=metrics.get("gradient_norm")
        self.text(x+12,y+40,"✓ Current metrics finite" if finite else "Current metrics: nonfinite" if metrics else "Current metrics: unassessed",size=10,color=self.accent if finite else self.muted if not metrics else "#ef777b",width=w-24)
        self.text(x+12,y+68,"Gradient norm: "+display_value(gradient),size=10,width=w-24)
        self.text(x+12,y+94,health,color=self.accent if health=="No trend warning" else ("#f3b74a" if health=="Watch" else color),bold=True,width=w-24)
        self.text(x+12,y+146,"Loss trend: last recorded validation. Warnings are not a diagnosis.",size=9,color=self.muted,width=w-24)
        return y+207

    def overview(self,x,y,w,values,val_epoch):
        data=self.data; self.card(x,y,w,185,"Epoch {} / {}".format(data.get("epoch",0),data.get("epochs",0)))
        epochs=data.get("epochs",1); current=data.get("epoch",0); records=data.get("epoch_records",{})
        first=max(1,current-2); shown=list(range(first,min(epochs,first+5)+1)); bw=min(75,(w-24)/max(1,len(shown))-6)
        for i,e in enumerate(shown):
            done="validation" in records.get(str(e),{}); active=e==current and data.get("status")=="running"
            fill="#0b503e" if done else "#1264a0" if active else self.panel2
            self.rounded(x+12+i*(bw+6),y+39,bw,26,fill=fill,outline=self.accent if done else "#35aefd" if active else self.border,r=7)
            self.text(x+12+i*(bw+6)+bw/2,y+52,f"E{e}"+(" ✓" if done else " ·" if active else ""),size=10,color="#ffffff" if done or active else self.fg,bold=True,anchor="center")
        batch,total=data.get("batch",0),data.get("batches_per_epoch",1)
        self.text(x+12,y+76,f"Batch {batch} / {total} ({batch/total:.1%})",size=11,bold=True)
        self.text(x+w-12,y+76,"ETA: "+duration(data.get("eta_seconds")),size=9,anchor="ne",color=self.muted)
        self.meter(x+12,y+100,w-24,100*batch/total)
        self.text(x+12,y+117,f"Overall training progress · {data.get('progress',0):.1%}",size=9,color=self.muted)
        self.meter(x+12,y+137,w-24,100*data.get("progress",0),"#39b9f0")
        self.text(x+12,y+157,f"Elapsed {duration(data.get('elapsed_seconds'))}   ·   {data.get('batches_per_second',0):.1f} batches/s",size=9,color=self.muted)
        yy=y+195; cols=6 if w>=735 else 3; mw=(w-(cols-1)*7)/cols
        for i,(name,value) in enumerate(values):
            xx=x+(i%cols)*(mw+7); cy=yy+(i//cols)*77
            self.card(xx,cy,mw,70,name); self.text(xx+mw/2,cy+46,display_value(value),size=15 if numeric(value) else 9,bold=True,anchor="center",color=self.accent if numeric(value) else self.muted)
        yy+=math.ceil(len(values)/cols)*77
        self.text(x+4,yy+1,"Validation epoch: "+str(val_epoch or "not reported")+" · Units supplied by training loop",size=9,color=self.muted)
        yy+=24
        if w>=490:
            cw=(w-8)/2; self.chart(x,yy,cw,205,"loss"); self.chart(x+cw+8,yy,cw,205,"accuracy"); return yy+215
        self.chart(x,yy,w,205,"loss"); self.chart(x,yy+213,w,205,"accuracy"); return yy+428

    def checklist(self,x,y,w,checks):
        self.card(x,y,w,477,"Training Checklist")
        for i,name in enumerate(CHECKS):
            yy=y+50+i*46; yes=checks.get(name) is True
            self.text(x+12,yy,"✓" if yes else "○",color=self.accent if yes else self.muted,bold=True,size=13)
            self.text(x+34,yy,name,size=10,width=w-45)
            self.text(x+34,yy+20,"Verified / reported" if yes else "Unassessed",size=8,color=self.muted)
        ready=all(checks.get(n) is True for n in CHECKS)
        self.rounded(x+12,y+420,w-24,45,outline=self.accent if ready else self.border,fill=self.panel2,r=8)
        self.text(x+w/2,y+442,"✓ Ready to Train" if ready else "Preflight not fully assessed",size=9,bold=True,color=self.accent if ready else self.muted,width=w-36,anchor="center")
        return y+487

    def list_card(self,x,y,w,title,mapping):
        rows=list(mapping.items()) if mapping else [("Status","Not reported")]
        self.card(x,y,w,48+len(rows)*38,title)
        for i,(k,v) in enumerate(rows):
            yy=y+43+i*38; self.text(x+14,yy,str(k),color=self.muted,size=10,width=w*.39)
            self.text(x+w*.42,yy,display_value(v),size=10,width=w*.55-12)
            self.canvas.create_line(x+14,yy+31,x+w-14,yy+31,fill=self.border)
        return y+58+len(rows)*38

    def history(self,x,y,w):
        self.text(x+8,y,"Epoch history · latest "+str(self.history_limit or "all"),size=13,bold=True)
        for label,value in (("10",10),("20",20),("50",50),("All",None)):
            xx=x+w-164+((10,20,50,None).index(value))*40
            tag="hist-"+label; item=self.rounded(xx,y-2,36,23,fill=self.panel2,outline=self.accent if self.history_limit==value else self.border,r=5)
            txt=self.text(xx+18,y+9,label,size=9,anchor="center")
            for it in (item,txt): self.canvas.addtag_withtag(tag,it)
            self.canvas.tag_bind(tag,"<Button-1>",lambda e,v=value:self.set_history(v))
        y+=32; rec=self.data.get("epoch_records",{}); current=self.data.get("epoch",0); start=1 if self.history_limit is None else max(1,current-self.history_limit+1)
        if start>1:self.text(x+8,y,f"{start-1} earlier epochs retained in the archive",color=self.muted,size=9); y+=23
        for epoch in range(start,current+1):
            row=rec.get(str(epoch),{}); done=max(0,min(self.data['batches_per_epoch'],self.data['completed_batches']-(epoch-1)*self.data['batches_per_epoch']))
            self.card(x,y,w,89,f"Epoch {epoch} / {self.data['epochs']}")
            self.text(x+w-12,y+13,f"{done} / {self.data['batches_per_epoch']} batches",anchor="ne",size=10,color=self.muted)
            self.meter(x+12,y+38,w-24,100*done/self.data['batches_per_epoch'])
            stages=row.get("stages",{})
            def stage_label(s):
                item=stages.get(s,{}); label=item.get("status","recorded" if s=="validation" and "validation" in row else "not reported")
                if item.get("total"):label+=f" ({item['completed']}/{item['total']})"
                return f"{s.title()}: {label}"
            stages_text="  ·  ".join(stage_label(s) for s in ("validation","checkpoint","verification"))
            self.text(x+12,y+58,stages_text,size=9,color=self.muted,width=w-24); y+=97
        return y

    def set_history(self,value): self.history_limit=value; self.draw()

    def draw(self):
        c=self.canvas
        if c.winfo_width()<300:return  # Wait for the native window manager to map/size the canvas.
        previous=c.yview(); old_top=previous[0] if previous else 0; near_end=bool(previous and previous[1]>=.98 and previous[0]>0)
        c.delete("all"); w=max(300,c.winfo_width()-8); y=5
        if not self.data: self.text(16,15,"Waiting for a DeepForge snapshot…"); return
        data=self.data; meta=self.metadata; values,checks,health,color,val_epoch=observations(data,meta)
        if self.tab=="Training":
            if w>=1050:
                left=max(210,int(w*.17)); right=max(235,int(w*.20)); center=w-left-right-20
                a=self.checklist(0,y,left,checks); b=self.overview(left+10,y,center,values,val_epoch); z=self.hardware(left+center+20,y,right); z=self.health(left+center+20,z,right,health,color); y=max(a,b,z)+8
            else:
                y=self.overview(0,y,w,values,val_epoch); y=self.checklist(0,y,w,checks); y=self.hardware(0,y,w); y=self.health(0,y,w,health,color)
            y=self.history(0,y,w)
        elif self.tab=="Metrics":
            y=self.overview(0,y,w,values,val_epoch); y=self.list_card(0,y,w,"Latest training metrics",data.get("metrics",{})); y=self.list_card(0,y,w,"Latest validation metrics",(data.get("epoch_history") or [{}])[-1].get("metrics",{})); y=self.history(0,y,w)
        elif self.tab=="Hardware":
            y=self.hardware(0,y,w); y=self.list_card(0,y,w,"Resource readings",data.get("resources",{}))
        elif self.tab=="Model":
            y=self.list_card(0,y,w,"Model information",{k:meta.get(k) for k in ("model","parameters","trainable_parameters","device")})
            layers=meta.get("layers",[])
            y=self.list_card(0,y,w,"Registered architecture · up to 100 modules",{layer['name']:f"{layer['type']}  |  {layer['parameters']:,} direct parameters" for layer in layers})
        elif self.tab=="Dataset":
            y=self.list_card(0,y,w,"Dataset metadata",meta.get("dataset",{})); y=self.list_card(0,y,w,"Latest batch label distribution",data.get("batch_distribution",{}).get("counts",{}))
        elif self.tab=="Reports":
            y=self.list_card(0,y,w,"Registered artifacts",{a['name']:a.get('bytes') for a in data.get("artifacts",[])}); y=self.list_card(0,y,w,"Diagnostic findings",{str(i+1):f.get("code") for i,f in enumerate(data.get("diagnostics",{}).get("findings",[]))})
            y=self.list_card(0,y,w,"Training snapshot",{"Snapshot":self.channel.path.name,"Status":data.get("status")})
        c.configure(scrollregion=(0,0,w,max(y+8,c.winfo_height())))
        c.yview_moveto(1 if near_end else old_top)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot"); parser.add_argument("--theme",choices=PALETTES,default="jade"); parser.add_argument("--interval",type=float,default=.5)
    args=parser.parse_args(argv)
    if not numeric(args.interval) or not .1<=args.interval<=10: parser.error("Interval must be between 0.1 and 10 seconds")
    import tkinter as tk
    try: root=tk.Tk()
    except tk.TclError as exc: parser.exit(2,f"DeepForge desktop requires a local graphical display: {exc}\n")
    DesktopMonitor(root,args.snapshot,theme=args.theme,interval=args.interval)
    root.mainloop(); return 0


if __name__=="__main__":
    raise SystemExit(main())
