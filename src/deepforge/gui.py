"""DeepForge Studio: local-first desktop research workspaces."""
import json
import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import __version__
from .gui_data import TOOLS, analyze, summary, preview_threshold
from .reporting import export_report, table_rows, finding_rows

BG, PANEL, SIDEBAR, INK, MUTED = '#F3EFE5', '#FFFCF5', '#123B32', '#183C33', '#536B60'
ACCENT, GOLD, LINE, ERROR = '#12664F', '#8A651E', '#DDDCD0', '#A3353D'
SIDE_INK, SIDE_MUTED, SIDE_GOLD, SIDE_ACTIVE = '#FAF4E6', '#B6CDBF', '#E8C77D', '#235A49'
COLORS = (ACCENT, GOLD, '#476F9A', ERROR)
ASSETS = Path(__file__).parent / 'assets'
ICONS = {'Home':'01','Dataset':'02','Leakage':'03','Images':'04','Groups':'05','YOLO':'06','Training':'07','Compare':'08','Metrics':'09','Scores':'10','Model':'11','Environment':'12','Manifest':'13'}
GROUPS = {
    'DATA QUALITY': ('Dataset', 'Leakage', 'Images', 'Groups', 'YOLO'),
    'EXPERIMENTS': ('Training', 'Compare', 'Metrics', 'Scores'),
    'SYSTEM & MODEL': ('Model', 'Environment', 'Manifest'),
}
NAMES = {'Dataset':'Dataset inspector','Leakage':'Exact duplicates','Images':'Image integrity', 'Groups':'Group leakage', 'YOLO':'YOLO labels', 'Training':'Learning curves','Compare':'Compare runs','Metrics':'Classification','Scores':'Binary scores','Model':'Model inspector','Environment':'Runtime','Manifest':'Reproducibility'}
DIR_TOOLS = {'Dataset','Leakage','Images','YOLO','Compare'}
DEMO_FILES = {'Training':'history.csv','Metrics':'predictions.csv','Scores':'scores.csv','Groups':'groups.csv'}
DEMO_DIRS = {'Dataset':'dataset','Leakage':'dataset','Images':'dataset','YOLO':'yolo','Compare':'runs'}
DEMOS = {**DEMO_FILES, **DEMO_DIRS}


class Studio:
    def __init__(self, root):
        self.root = root
        root.title(f'DeepForge Studio | {__version__}')
        root.geometry('1440x900'); root.minsize(1000, 720); root.configure(bg=BG)
        self.tool, self.busy, self.report, self.chart = 'Home', False, None, None
        self.results, self.fields, self.nav = queue.Queue(), {}, {}
        self.recent, self.current_opts = [], {}
        self.saved_fields, self.chart_points = {}, []
        self.sidebar_visible = True
        self.nav_query = tk.StringVar()
        self.find_query = tk.StringVar()
        self.nav_sections = []
        self.chart_mode = tk.StringVar(value='ROC')
        self.style = ttk.Style(root); self.style.theme_use('clam')
        self.font = 'Segoe UI' if root.tk.call('tk', 'windowingsystem') == 'win32' else 'DejaVu Sans'
        self.configure_style()
        self.side = side = tk.Frame(root, bg=SIDEBAR, width=248); side.pack(side='left', fill='y'); side.pack_propagate(False)
        self.logo = self.load_brand_image('xvarna_sidebar.png')
        self.app_icon = self.load_brand_image('icon.png')
        root.iconphoto(True, self.app_icon)
        brand = tk.Frame(side, bg=SIDEBAR); brand.pack(fill='x', padx=20, pady=(16,10))
        tk.Label(brand,image=self.logo,bg=SIDEBAR).pack(anchor='w')
        self.label(brand,'DEEPFORGE',17,SIDE_INK,SIDEBAR,True).pack(anchor='w',pady=(8,0))
        self.label(brand,'RESEARCH STUDIO',9,SIDE_GOLD,SIDEBAR,True).pack(anchor='w')
        foot = tk.Frame(side,bg=SIDEBAR); foot.pack(side='bottom',fill='x',padx=16,pady=10)
        tk.Frame(foot,bg=SIDE_ACTIVE,height=1).pack(fill='x',pady=(0,8))
        self.label(foot,'Alireza Khatibi',10,SIDE_INK,SIDEBAR,True).pack(anchor='w')
        self.label(foot,f'XVARNA  /  v{__version__}',8,SIDE_GOLD,SIDEBAR).pack(anchor='w',pady=(3,0))
        search = tk.Frame(side,bg=SIDEBAR); search.pack(fill='x',padx=16,pady=(0,8))
        self.label(search,'FIND A WORKSPACE',8,SIDE_MUTED,SIDEBAR,True).pack(anchor='w',pady=(0,4))
        self.search_entry = ttk.Entry(search,textvariable=self.nav_query); self.search_entry.pack(fill='x')
        self.search_entry.bind('<Return>',self.open_first_match)
        nav_area=tk.Frame(side,bg=SIDEBAR); nav_area.pack(fill='both',expand=True)
        nav_canvas=tk.Canvas(nav_area,bg=SIDEBAR,highlightthickness=0)
        nav_scroll=ttk.Scrollbar(nav_area,style='Side.Vertical.TScrollbar',command=nav_canvas.yview)
        nav_canvas.configure(yscrollcommand=lambda a,b:self.auto_scroll(nav_scroll,a,b)); nav_canvas.pack(side='left',fill='both',expand=True)
        nav_body=tk.Frame(nav_canvas,bg=SIDEBAR); nav_window=nav_canvas.create_window((0,0),window=nav_body,anchor='nw')
        nav_body.bind('<Configure>',lambda _:nav_canvas.configure(scrollregion=nav_canvas.bbox('all')))
        nav_canvas.bind('<Configure>',lambda e:nav_canvas.itemconfigure(nav_window,width=e.width))
        self.nav_canvas=nav_canvas
        self.nav_button(nav_body,'Home','Overview')
        for group, tools in GROUPS.items():
            heading=self.label(nav_body,group,8,SIDE_MUTED,SIDEBAR,True); heading.pack(anchor='w',padx=20,pady=(9,3))
            self.nav_sections.append((heading,tools))
            for tool in tools: self.nav_button(nav_body,tool,NAMES[tool])
        self.nav_query.trace_add('write',self.filter_navigation)
        self.main = tk.Frame(root,bg=BG); self.main.pack(side='left',fill='both',expand=True,padx=24,pady=18)
        top = tk.Frame(self.main,bg=BG); top.pack(fill='x')
        self.menu_btn=ttk.Button(top,text='Menu',width=6,command=self.toggle_sidebar); self.menu_btn.pack(side='left',padx=(0,10))
        self.breadcrumb = self.label(top,'WORKSPACE  /  OVERVIEW',9,GOLD,BG,True); self.breadcrumb.pack(side='left')
        ttk.Button(top,text='Checkpoint',command=self.checkpoint_settings).pack(side='right',padx=4)
        ttk.Button(top,text='About',width=7,command=self.help).pack(side='right')
        self.top_status=self.label(top,'LOCAL  /  READ-ONLY INPUTS',9,ACCENT,BG,True); self.top_status.pack(side='right',padx=12)
        top.bind('<Configure>',lambda e:self.top_status.configure(text='LOCAL' if e.width<850 else 'LOCAL  /  READ-ONLY INPUTS'))
        self.heading = self.label(self.main,'',26,INK,BG,True); self.heading.pack(anchor='w',pady=(8,0))
        self.subheading = self.label(self.main,'',10,MUTED,BG); self.subheading.pack(anchor='w',pady=(4,14),fill='x'); self.main.bind('<Configure>',lambda e:self.subheading.configure(wraplength=max(220,e.width-20)))
        self.home = tk.Frame(self.main,bg=BG)
        self.home_canvas = tk.Canvas(self.home,bg=BG,highlightthickness=0)
        home_scroll = ttk.Scrollbar(self.home,command=self.home_canvas.yview)
        self.home_canvas.configure(yscrollcommand=lambda a,b:self.auto_scroll(home_scroll,a,b))
        self.home_canvas.pack(side='left',fill='both',expand=True)
        self.home_body = tk.Frame(self.home_canvas,bg=BG)
        self.home_window = self.home_canvas.create_window((0,0),window=self.home_body,anchor='nw')
        self.home_body.bind('<Configure>',lambda _:self.home_canvas.configure(scrollregion=self.home_canvas.bbox('all')))
        self.home_canvas.bind('<Configure>',lambda e:self.home_canvas.itemconfigure(self.home_window,width=e.width))
        self.work = tk.Frame(self.main,bg=BG)
        self.form = tk.Frame(self.work,bg=PANEL,padx=16,pady=10); self.form.pack(fill='x')
        config_header=tk.Frame(self.form,bg=PANEL);config_header.pack(fill='x')
        self.label(config_header,'INPUT CONFIGURATION',9,GOLD,PANEL,True).pack(side='left')
        self.config_btn=ttk.Button(config_header,text='Hide inputs',width=12,command=self.toggle_config);self.config_btn.pack(side='right')
        self.inputs = tk.Frame(self.form,bg=PANEL); self.inputs.pack(fill='x');self.config_visible=True
        toolbar = tk.Frame(self.work,bg=BG); toolbar.pack(fill='x',pady=12)
        self.run_btn = ttk.Button(toolbar,text='Run analysis',width=13,style='Action.TButton',command=self.run); self.run_btn.pack(side='left')
        self.demo_btn = ttk.Button(toolbar,text='Load demo',width=10,command=self.load_demo); self.demo_btn.pack(side='left',padx=7)
        self.export_btn = ttk.Button(toolbar,text='Export JSON',width=11,command=self.export,state='disabled'); self.export_btn.pack(side='left')
        self.html_btn = ttk.Button(toolbar,text='Export HTML',width=11,command=lambda:self.export('html'),state='disabled'); self.html_btn.pack(side='left',padx=7)
        self.status_row=tk.Frame(self.work,bg=BG); self.status_row.pack(fill='x',pady=(0,5))
        self.status = self.label(self.status_row,'Ready',9,MUTED,BG); self.status.pack(side='left')
        self.progress = ttk.Progressbar(self.status_row,mode='indeterminate',length=90)
        self.cards = tk.Frame(self.work,bg=BG); self.cards.pack(fill='x',pady=(0,12))
        self.threshold_panel=tk.Frame(self.work,bg=PANEL,padx=12,pady=6)
        self.threshold_var=tk.DoubleVar(value=.5)
        self.threshold_info=self.label(self.threshold_panel,'',9,INK,PANEL,True)
        self.label(self.threshold_panel,'Threshold preview',9,GOLD,PANEL,True).pack(side='left')
        self.threshold_scale=ttk.Scale(self.threshold_panel,from_=0,to=1,variable=self.threshold_var,command=self.threshold_changed,length=100); self.threshold_scale.pack(side='left',padx=8,fill='x',expand=True)
        self.threshold_info.pack(side='left',padx=4)
        ttk.Button(self.threshold_panel,text='Apply',command=self.apply_threshold).pack(side='right',padx=(4,0))
        self.tabs = ttk.Notebook(self.work); self.tabs.pack(fill='both',expand=True)
        plot_frame = tk.Frame(self.tabs,bg=PANEL); self.tabs.add(plot_frame,text=' Visualization ')
        self.chart_header = tk.Frame(plot_frame,bg=PANEL); self.chart_header.pack(fill='x',padx=16,pady=(12,0))
        self.chart_title = self.label(self.chart_header,'ANALYSIS PREVIEW',9,GOLD,PANEL,True); self.chart_title.pack(side='left')
        self.chart_select = ttk.Combobox(self.chart_header,textvariable=self.chart_mode,values=('ROC','Precision–recall','Calibration'),state='readonly',width=18)
        self.chart_select.bind('<<ComboboxSelected>>',lambda _:self.draw())
        self.canvas = tk.Canvas(plot_frame,bg=PANEL,highlightthickness=0); self.canvas.pack(fill='both',expand=True,padx=8,pady=4)
        self.canvas.bind('<Configure>',lambda _:self.draw())
        self.canvas.bind('<Motion>',self.chart_hover); self.canvas.bind('<Leave>',lambda _:self.canvas.delete('hover'))
        table = tk.Frame(self.tabs,bg=PANEL); self.tabs.add(table,text=' Findings ')
        filterbar=tk.Frame(table,bg=PANEL,padx=12,pady=8); filterbar.pack(fill='x')
        self.label(filterbar,'Filter findings',9,MUTED,PANEL,True).pack(side='left',padx=(0,8))
        ttk.Entry(filterbar,textvariable=self.find_query,width=24).pack(side='left',fill='x',expand=True)
        ttk.Button(filterbar,text='Copy row',command=self.copy_row).pack(side='right',padx=(8,0))
        self.table = ttk.Treeview(table,columns=('key','value'),show='headings',selectmode='browse')
        self.find_query.trace_add('write',lambda *_:self.render_findings())
        self.table.heading('key',text='Field'); self.table.heading('value',text='Value')
        self.table.column('key',width=280,stretch=True); self.table.column('value',width=430,stretch=True)
        sy = ttk.Scrollbar(table,command=self.table.yview); sx = ttk.Scrollbar(table,orient='horizontal',command=self.table.xview)
        self.table.configure(yscrollcommand=sy.set,xscrollcommand=sx.set); sy.pack(side='right',fill='y'); sx.pack(side='bottom',fill='x'); self.table.pack(fill='both',expand=True)
        report_frame = tk.Frame(self.tabs,bg=PANEL); self.tabs.add(report_frame,text=' Full report · JSON ')
        self.raw = tk.Text(report_frame,bg=PANEL,fg=INK,insertbackground=INK,wrap='none',font=('Consolas' if self.font=='Segoe UI' else 'DejaVu Sans Mono',10),relief='flat',padx=12,pady=12)
        sy=ttk.Scrollbar(report_frame,command=self.raw.yview); sx=ttk.Scrollbar(report_frame,orient='horizontal',command=self.raw.xview)
        self.raw.configure(yscrollcommand=sy.set,xscrollcommand=sx.set,state='disabled'); sy.pack(side='right',fill='y'); sx.pack(side='bottom',fill='x'); self.raw.pack(fill='both',expand=True)
        root.bind('<Control-Return>', lambda _:self.run() if self.tool!='Home' else None)
        root.bind('<Control-s>',lambda _:self.export())
        root.bind('<Control-k>',self.focus_search)
        root.bind('<Control-b>',lambda _:self.toggle_sidebar())
        root.bind_all('<MouseWheel>',self.scroll_wheel)
        root.bind_all('<Button-4>',self.scroll_wheel); root.bind_all('<Button-5>',self.scroll_wheel)
        self.select('Home'); root.after(80,self.poll)


    def checkpoint_settings(self):
        from .checkpoint import CheckpointPolicy, atomic_json
        from dataclasses import asdict
        dialog = tk.Toplevel(self.root)
        dialog.title("Checkpoint configuration")
        values = {key: tk.StringVar(value=value) for key, value in {
            "root": "checkpoints", "run_id": "run_001", "keep_last": "2",
            "save_every_epochs": "1", "max_bytes": ""
        }.items()}
        enabled = tk.BooleanVar(value=True)
        for key, var in values.items():
            ttk.Label(dialog, text=key).pack(anchor="w", padx=12)
            ttk.Entry(dialog, textvariable=var, width=48).pack(padx=12, pady=3)
        ttk.Checkbutton(dialog, text="Enable checkpointing", variable=enabled).pack()
        def save():
            try:
                policy = CheckpointPolicy(enabled=enabled.get(),
                    keep_last=int(values["keep_last"].get()),
                    save_every_epochs=int(values["save_every_epochs"].get()),
                    max_bytes=int(values["max_bytes"].get()) if values["max_bytes"].get() else None)
                import re
                run_id = values["run_id"].get()
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
                    raise ValueError("invalid run identifier")
                if not values["root"].get().strip():
                    raise ValueError("storage root is required")
                path = filedialog.asksaveasfilename(parent=dialog, defaultextension=".json")
                if path:
                    atomic_json(Path(path), {"root": values["root"].get(),
                        "run_id": run_id, "policy": asdict(policy)})
                    dialog.destroy()
            except (ValueError, OSError) as exc:
                messagebox.showerror("Checkpoint", str(exc), parent=dialog)
        ttk.Button(dialog, text="Export configuration", command=save).pack(pady=12)

    def load_brand_image(self, filename):
        path = ASSETS / filename
        try:
            return tk.PhotoImage(master=self.root, file=str(path))
        except tk.TclError:
            from PIL import Image, ImageTk
            with Image.open(path) as image:
                return ImageTk.PhotoImage(image.convert("RGBA"), master=self.root)

    def configure_style(self):
        s=self.style
        s.configure('.',font=(self.font,10),background=BG,foreground=INK)
        s.configure('TFrame',background=BG)
        s.configure('TEntry',fieldbackground='#FFFFFF',foreground=INK,insertcolor=INK,padding=8,bordercolor=LINE,lightcolor=LINE,darkcolor=LINE)
        s.configure('TCombobox',fieldbackground='#FFFFFF',foreground=INK,padding=7,arrowcolor=INK)
        s.map('TCombobox',fieldbackground=[('readonly','#FFFFFF')],foreground=[('readonly',INK)])
        self.root.option_add('*TCombobox*Listbox.background',PANEL); self.root.option_add('*TCombobox*Listbox.foreground',INK)
        s.configure('TButton',background='#E5EBE1',foreground=INK,bordercolor=LINE,lightcolor=LINE,darkcolor=LINE,padding=(12,8),font=(self.font,9,'bold'))
        s.map('TButton',background=[('active','#D1E4D5'),('disabled','#F1EDE4')],foreground=[('disabled','#8B968D')])
        s.configure('Action.TButton',background=ACCENT,foreground='#FFFFFF',font=(self.font,10,'bold'),padding=(16,9))
        s.map('Action.TButton',background=[('active','#1B795B'),('disabled','#A5B9AA')],foreground=[('active','#FFFFFF'),('disabled',INK)])
        s.configure('Horizontal.TScale',background=PANEL,troughcolor='#D4E1D4',sliderlength=18,bordercolor=LINE,lightcolor=LINE,darkcolor=LINE)
        s.configure('TNotebook',background=BG,borderwidth=0,bordercolor=LINE,lightcolor=LINE,darkcolor=LINE)
        s.configure('TNotebook.Tab',background='#E8E3D8',foreground=MUTED,padding=(13,9),bordercolor=LINE,lightcolor=LINE,darkcolor=LINE)
        s.map('TNotebook.Tab',background=[('selected',PANEL)],foreground=[('selected',ACCENT)])
        s.configure('Treeview',background=PANEL,fieldbackground=PANEL,foreground=INK,rowheight=30,bordercolor=LINE)
        s.configure('Treeview.Heading',background='#E8EEE5',foreground=GOLD,font=(self.font,9,'bold'),padding=6)
        s.map('Treeview',background=[('selected','#CFE4D4')],foreground=[('selected',INK)])
        s.configure('Vertical.TScrollbar',background='#B9C5B7',troughcolor=BG,bordercolor=BG,lightcolor=LINE,darkcolor=LINE,arrowcolor=MUTED)
        s.configure('Horizontal.TScrollbar',background='#B9C5B7',troughcolor=BG,bordercolor=BG,lightcolor=LINE,darkcolor=LINE,arrowcolor=MUTED)
        s.configure('Side.Vertical.TScrollbar',background=SIDE_ACTIVE,troughcolor=SIDEBAR,bordercolor=SIDEBAR,lightcolor=SIDEBAR,darkcolor=SIDEBAR,arrowcolor=SIDE_MUTED)
        s.configure('Horizontal.TProgressbar',background=ACCENT,troughcolor=PANEL,bordercolor=BG,lightcolor=ACCENT,darkcolor=ACCENT)

    def label(self,parent,text,size,fg,bg,bold=False):
        return tk.Label(parent,text=text,fg=fg,bg=bg,font=(self.font,size,'bold' if bold else 'normal'),justify='left',anchor='w')

    def nav_button(self,parent,key,caption):
        b=tk.Button(parent,text=f'{ICONS[key]}   {caption}',anchor='w',font=(self.font,10),padx=12,pady=4,bg=SIDEBAR,fg=SIDE_INK,activebackground=SIDE_ACTIVE,activeforeground=SIDE_GOLD,relief='flat',borderwidth=0,highlightthickness=1,highlightcolor=SIDE_GOLD,highlightbackground=SIDEBAR,command=lambda:self.select(key),cursor='hand2')
        b.pack(fill='x',padx=10,pady=1); self.nav[key]=b
        b.bind('<Enter>',lambda _,k=key,w=b:w.configure(bg=SIDE_ACTIVE) if k!=self.tool else None)
        b.bind('<Leave>',lambda _,k=key,w=b:w.configure(bg=SIDEBAR) if k!=self.tool else None)

    def auto_scroll(self,scroll,first,last):
        scroll.set(first,last)
        if float(first)<=0 and float(last)>=1:
            if scroll.winfo_manager(): scroll.pack_forget()
        elif not scroll.winfo_manager():
            scroll.pack(side='right',fill='y',before=next(w for w in scroll.master.pack_slaves() if isinstance(w,tk.Canvas)))

    def threshold_changed(self,value):
        if self.tool!='Scores' or self.report is None: return
        t=round(float(value),3); m=preview_threshold(self.report,t)
        self.threshold_info.configure(text=f'{t:.3f}  |  P {m["precision"]:.2f}  R {m["recall"]:.2f}  F1 {m["f1"]:.2f}')

    def apply_threshold(self):
        if self.busy or self.tool!='Scores' or self.report is None: return
        # Recompute on the explicitly selected CSV and update the exported report.
        self.fields['threshold'].set(f'{self.threshold_var.get():.3f}'); self.run()

    def toggle_config(self,visible=None):
        visible=not self.config_visible if visible is None else bool(visible)
        if visible: self.inputs.pack(fill='x')
        else: self.inputs.pack_forget()
        self.config_visible=visible;self.config_btn.configure(text='Hide inputs' if visible else 'Show inputs')

    def toggle_sidebar(self):
        if self.sidebar_visible:
            self.side.pack_forget()
        else:
            self.side.pack(side='left',fill='y',before=self.main)
        self.sidebar_visible=not self.sidebar_visible

    def focus_search(self,event=None):
        if not self.sidebar_visible: self.toggle_sidebar()
        self.search_entry.focus_set(); self.search_entry.selection_range(0,'end')
        return 'break'

    def matches(self,key):
        q=self.nav_query.get().strip().casefold()
        return not q or q in ('overview home' if key=='Home' else ' '.join((key,NAMES[key],*TOOLS[key])).casefold())

    def filter_navigation(self,*_):
        for b in self.nav.values(): b.pack_forget()
        for heading,_ in self.nav_sections: heading.pack_forget()
        if self.matches('Home'): self.nav['Home'].pack(fill='x',padx=10,pady=1)
        for heading,keys in self.nav_sections:
            matches=[k for k in keys if self.matches(k)]
            if matches: heading.pack(anchor='w',padx=20,pady=(9,3))
            for k in matches: self.nav[k].pack(fill='x',padx=10,pady=1)
        self.nav_canvas.yview_moveto(0)

    def open_first_match(self,event=None):
        for key in self.nav:
            if self.matches(key): self.select(key); break
        return 'break'

    def scroll_wheel(self,event):
        widget=event.widget
        while widget is not None:
            if widget in (self.home_canvas,self.nav_canvas):
                units=-1 if getattr(event,'num',None)==4 or getattr(event,'delta',0)>0 else 1
                widget.yview_scroll(units*3,'units'); return 'break'
            widget=getattr(widget,'master',None)

    def render_findings(self):
        for item in self.table.get_children(): self.table.delete(item)
        query=self.find_query.get().casefold().strip()
        if self.report is not None:
            for i,(k,v) in enumerate(finding_rows(self.report)+table_rows(self.report)):
                if not query or query in f'{k} {v}'.casefold():
                    self.table.insert('','end',values=(k,v),tags=('even' if i%2==0 else 'odd',))
        self.table.tag_configure('odd',background='#F1F4EB')
        self.table.tag_configure('even',background=PANEL)

    def copy_row(self):
        selected=self.table.selection()
        if selected:
            values=self.table.item(selected[0],'values'); self.root.clipboard_clear(); self.root.clipboard_append('\t'.join(values))
            self.status.configure(text='Finding copied',fg=ACCENT)

    def chart_hover(self,event):
        c=self.canvas; c.delete('hover')
        if not self.chart_points: return
        x,y,label=min(self.chart_points,key=lambda p:(p[0]-event.x)**2+(p[1]-event.y)**2)
        if (x-event.x)**2+(y-event.y)**2>32**2: return
        c.create_oval(x-5,y-5,x+5,y+5,outline=GOLD,width=2,tags='hover')
        tx=min(max(x+10,8),max(8,c.winfo_width()-240)); ty=max(20,y-24)
        t=c.create_text(tx,ty,text=label,anchor='w',fill=SIDE_INK,font=(self.font,9),tags='hover')
        a,b,d,e=c.bbox(t); rect=c.create_rectangle(a-7,b-5,d+7,e+5,fill=SIDEBAR,outline=GOLD,tags='hover'); c.tag_lower(rect,t)

    def field(self,parent,name,caption,default='',choices=None,width=18):
        box=tk.Frame(parent,bg=PANEL); box.pack(side='left',fill='x',expand=True,padx=(0,9))
        self.label(box,caption,9,MUTED,PANEL).pack(anchor='w',pady=(0,4))
        var=tk.StringVar(value=default); self.fields[name]=var
        widget=ttk.Combobox(box,textvariable=var,values=choices,state='readonly',width=min(width,12)) if choices else ttk.Entry(box,textvariable=var,width=min(width,15))
        widget.pack(fill='x'); return box

    def render_home(self):
        for child in self.home_body.winfo_children(): child.destroy()
        hero=tk.Frame(self.home_body,bg=SIDEBAR,padx=26,pady=22); hero.pack(fill='x',pady=(0,14))
        hero_top=tk.Frame(hero,bg=SIDEBAR); hero_top.pack(fill='x')
        self.label(hero_top,'XVARNA  /  RESEARCH OPERATIONS',9,SIDE_GOLD,SIDEBAR,True).pack(side='left')
        self.label(hero_top,'PROFESSIONAL EDITION',8,SIDE_MUTED,SIDEBAR,True).pack(side='right')
        hero_heading=self.label(hero,'Build confidence in every experiment.',21,SIDE_INK,SIDEBAR,True); hero_heading.pack(anchor='w',pady=(12,6),fill='x')
        hero.bind('<Configure>',lambda e:hero_heading.configure(wraplength=max(220,e.width-52)))
        self.label(hero,'From dataset quality to model evaluation, one local-first workspace.',10,SIDE_MUTED,SIDEBAR).pack(anchor='w')
        actions=tk.Frame(hero,bg=SIDEBAR); actions.pack(fill='x',pady=(16,0))
        ttk.Button(actions,text='Explore synthetic demo',style='Action.TButton',command=lambda:self.open_demo('Training')).pack(side='left')
        ttk.Button(actions,text='Audit a dataset',command=lambda:self.select('Dataset')).pack(side='left',padx=8)
        tk.Frame(hero,bg=SIDE_GOLD,height=2).pack(fill='x',pady=(18,0))
        row=tk.Frame(self.home_body,bg=BG); row.pack(fill='x',pady=(0,14))
        for i,(label,value,detail) in enumerate([('Research tools',str(len(TOOLS)),'Independent workspaces'),('Session analyses',str(len(self.recent)),'Last 15 reports retained'),('Execution','Local','No cloud needed')]):
            row.grid_columnconfigure(i,weight=1,uniform='stats'); card=tk.Frame(row,bg=PANEL,padx=14,pady=12,highlightbackground=LINE,highlightthickness=1); card.grid(row=0,column=i,sticky='nsew',padx=(0,8))
            self.label(card,label,9,MUTED,PANEL).pack(anchor='w'); self.label(card,value,22,ACCENT,PANEL,True).pack(anchor='w'); self.label(card,detail,9,MUTED,PANEL).pack(anchor='w')
        self.label(self.home_body,'START A WORKFLOW',9,GOLD,BG,True).pack(anchor='w',pady=(0,6))
        row=tk.Frame(self.home_body,bg=BG); row.pack(fill='x',pady=(0,14))
        for i,(name,desc,tool) in enumerate([('01  /  Audit data','Inspect balance and leakage.','Dataset'),('02  /  Diagnose runs','Compare learning curves.','Training'),('03  /  Evaluate','Review prediction metrics.','Scores')]):
            row.grid_columnconfigure(i,weight=1,uniform='flows'); card=tk.Frame(row,bg=PANEL,padx=14,pady=12,highlightbackground=LINE,highlightthickness=1); card.grid(row=0,column=i,sticky='nsew',padx=(0,8))
            self.label(card,name,11,INK,PANEL,True).pack(anchor='w'); self.label(card,desc,9,MUTED,PANEL).pack(anchor='w',pady=6)
            ttk.Button(card,text='Open workspace',command=lambda t=tool:self.select(t)).pack(anchor='w')
        self.label(self.home_body,'RECENT SESSION ANALYSES',9,GOLD,BG,True).pack(anchor='w',pady=(0,6))
        if not self.recent:
            self.label(self.home_body,'No reports yet. Choose a workspace or run the bundled synthetic demo.',10,MUTED,BG).pack(anchor='w')
        else:
            for item in self.recent[:3]:
                row=tk.Frame(self.home_body,bg=PANEL,padx=12,pady=4); row.pack(fill='x',pady=2)
                self.label(row,item['time']+'  /  '+NAMES[item['tool']],10,INK,PANEL).pack(side='left')
                ttk.Button(row,text='View report',command=lambda r=item:self.reopen(r)).pack(side='right')

    def select(self,tool):
        if self.busy: return
        if self.tool!='Home' and self.fields: self.saved_fields[self.tool]={k:v.get() for k,v in self.fields.items()}
        self.tool,self.report,self.chart=tool,None,None; self.fields={}
        for t,b in self.nav.items(): b.configure(bg=SIDE_ACTIVE if t==tool else SIDEBAR,fg=SIDE_GOLD if t==tool else SIDE_INK)
        self.find_query.set('')
        self.breadcrumb.configure(text='WORKSPACE  /  '+('OVERVIEW' if tool=='Home' else NAMES[tool].upper()))
        self.work.pack_forget(); self.home.pack_forget()
        if tool=='Home':
            self.heading.configure(text='Research dashboard'); self.subheading.configure(text='Your local command center for reproducible deep learning.')
            self.render_home(); self.home.pack(fill='both',expand=True); return
        self.work.pack(fill='both',expand=True);self.toggle_config(True)
        self.heading.configure(text=TOOLS[tool][0]); self.subheading.configure(text=TOOLS[tool][1])
        for child in self.inputs.winfo_children(): child.destroy()
        if tool in DIR_TOOLS or tool in DEMO_FILES or tool=='Manifest':
            row=tk.Frame(self.inputs,bg=PANEL); row.pack(fill='x',pady=4)
            self.field(row,'path','Dataset / run directory' if tool in DIR_TOOLS else 'Optional input file' if tool=='Manifest' else 'Input CSV file',width=40)
            ttk.Button(row,text='Browse…',command=self.browse).pack(side='left',anchor='s')
        row=tk.Frame(self.inputs,bg=PANEL); row.pack(fill='x',pady=4)
        if tool in ('Training','Compare'):
            for k,cap,val in (('metric','Validation metric','val_loss'),('train_metric','Training metric (optional)','train_loss'),('epoch_column','Epoch column','epoch')):
                self.field(row,k,cap,val,width=14)
            self.field(row,'mode','Optimize','min',('min','max'),7)
        elif tool=='Metrics':
            self.field(row,'true_column','True label column','y_true'); self.field(row,'pred_column','Predicted label column','y_pred')
        elif tool=='Scores':
            for k,cap,val in (('true_column','True label column','y_true'),('score_column','Probability column','score'),('positive_label','Positive label','1'),('threshold','Decision threshold','0.5')):
                self.field(row,k,cap,val,width=12)
        elif tool=='Model':
            for k,cap,val in (('inputs','Input width','16'),('hidden','Hidden width','32'),('outputs','Output width','3'),('batch','Batch size','1')): self.field(row,k,cap,val)
        elif tool=='Images':
            self.field(row,'similarity','Perceptual comparison','yes',('yes','no')); self.field(row,'threshold','Max Hamming distance','4'); self.field(row,'max_images','Image scan limit','2000')
        elif tool=='Groups':
            self.field(row,'group_column','Group ID column','group_id'); self.field(row,'split_column','Split column','split')
        elif tool=='YOLO': self.field(row,'classes','Number of classes (optional)','')
        elif tool=='Leakage': self.label(row,'Exact byte duplicates only. A clean report does not establish patient independence.',9,GOLD,PANEL).pack(anchor='w')
        elif tool=='Environment': self.label(row,'Reports Python and optional PyTorch / CUDA availability without network access.',9,MUTED,PANEL).pack(anchor='w')
        for k,v in self.saved_fields.get(tool,{}).items():
            if k in self.fields: self.fields[k].set(v)
        self.demo_btn.configure(state='normal' if tool in DEMOS else 'disabled')
        self.export_btn.configure(state='disabled'); self.html_btn.configure(state='disabled'); self.status.configure(text='Ready',fg=MUTED)
        self.chart_select.pack_forget(); self.chart_mode.set('ROC'); self.tabs.select(0); self.render()

    def open_demo(self,tool):
        if self.busy: return
        self.select(tool); self.load_demo(); self.run()

    def load_demo(self):
        if self.tool not in DEMOS or self.busy: return
        self.fields['path'].set(str(Path(__file__).parent/'samples'/DEMOS[self.tool])); self.status.configure(text='Synthetic demo loaded',fg=GOLD)

    def browse(self):
        path=filedialog.askdirectory(parent=self.root) if self.tool in DIR_TOOLS else filedialog.askopenfilename(parent=self.root,filetypes=[('All files','*')] if self.tool=='Manifest' else [('CSV files','*.csv'),('All files','*')])
        if path: self.fields['path'].set(path)

    def run(self):
        if self.busy or self.tool=='Home': return
        opts={k:v.get().strip() for k,v in self.fields.items()}
        if 'path' in opts and not opts['path'] and self.tool!='Manifest':
            messagebox.showwarning('Input required','Select a file or directory first.',parent=self.root); return
        self.busy=True; self.report=self.chart=None; self.current_opts=opts; self.render()
        for b in (self.run_btn,self.demo_btn,self.export_btn,self.html_btn): b.configure(state='disabled')
        self.status.configure(text='Analyzing…',fg=GOLD); self.progress.pack(side='right',padx=5); self.progress.start(12)
        tool=self.tool
        def work():
            try: self.results.put((True,analyze(tool,opts)))
            except Exception as exc: self.results.put((False,f'{type(exc).__name__}: {exc}'))
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        try: ok,value=self.results.get_nowait()
        except queue.Empty: pass
        else:
            self.busy=False; self.progress.stop(); self.progress.pack_forget(); self.run_btn.configure(state='normal')
            self.demo_btn.configure(state='normal' if self.tool in DEMOS else 'disabled')
            if ok:
                self.report,self.chart=value
                self.recent.insert(0,{'time':datetime.now().strftime('%H:%M:%S'),'tool':self.tool,'options':self.current_opts.copy(),'report':self.report,'chart':self.chart}); del self.recent[15:]
                self.status.configure(text='Analysis complete',fg=ACCENT)
                if self.root.winfo_height()<800: self.toggle_config(False)
                self.export_btn.configure(state='normal'); self.html_btn.configure(state='normal'); self.render()
            else:
                self.status.configure(text='Analysis failed',fg=ERROR); messagebox.showerror('Analysis failed',value,parent=self.root)
        self.root.after(80,self.poll)

    def reopen(self,item):
        if self.busy: return
        self.select(item['tool'])
        for k,v in item['options'].items():
            if k in self.fields: self.fields[k].set(v)
        self.report,self.chart=item['report'],item['chart']; self.current_opts=item['options'].copy(); self.status.configure(text='Saved session result',fg=ACCENT)
        self.export_btn.configure(state='normal'); self.html_btn.configure(state='normal'); self.render()

    def export(self,fmt='json'):
        if self.report is None: return
        path=filedialog.asksaveasfilename(parent=self.root,defaultextension='.'+fmt,filetypes=[(fmt.upper()+' report','*.'+fmt)],initialfile=f'deepforge-{self.tool.lower()}.{fmt}')
        if path:
            try:
                source = self.current_opts.get('path')
                output = Path(path).resolve()
                if output.suffix.lower() != '.'+fmt:
                    raise ValueError('Choose an output filename ending in .'+fmt)
                if source:
                    source_path = Path(source).resolve()
                    if output == source_path or (source_path.is_dir() and output.is_relative_to(source_path) and output.exists()):
                        raise ValueError('Choose a different output path; an input file must not be overwritten.')
                export_report(path,TOOLS[self.tool][0],self.report,cards=summary(self.tool,self.report),fmt=fmt,chart=self.active_chart())
            except (OSError,ValueError) as exc: messagebox.showerror('Export failed',str(exc),parent=self.root)
            else: self.status.configure(text='Report exported',fg=ACCENT)

    def active_chart(self):
        if self.tool!='Scores' or self.report is None or self.chart_mode.get()=='ROC': return self.chart
        mode=self.chart_mode.get()
        if mode=='Calibration':
            rows=[r for r in self.report['calibration'] if r['count']]
            return {'type':'probability','title':'Calibration','x':[r['mean_probability'] for r in rows],'y':[r['positive_fraction'] for r in rows],'xlabel':'Mean probability','baseline':None}
        rows=self.report['precision_recall_curve']
        return {'type':'probability','title':'Precision–recall','x':[r['recall'] for r in rows],'y':[r['precision'] for r in rows],'xlabel':'Recall','baseline':self.report['positive_prevalence']}

    def help(self):
        dialog=tk.Toplevel(self.root); dialog.title('About DeepForge Studio'); dialog.configure(bg=PANEL); dialog.transient(self.root); dialog.resizable(False,False)
        hero=tk.Frame(dialog,bg=SIDEBAR,padx=28,pady=20); hero.pack(fill='x')
        logo=tk.PhotoImage(file=str(ASSETS/'xvarna_about.png')); image=tk.Label(hero,image=logo,bg=SIDEBAR); image.image=logo; image.pack()
        body=tk.Frame(dialog,bg=PANEL,padx=24,pady=18); body.pack(fill='both',expand=True)
        self.label(body,'DeepForge Studio',22,INK,PANEL,True).pack(anchor='w')
        self.label(body,f'Professional Edition  /  v{__version__}',10,GOLD,PANEL,True).pack(anchor='w',pady=(4,10))
        self.label(body,'Created by Alireza Khatibi  ·  XVARNA',11,INK,PANEL,True).pack(anchor='w')
        self.label(body,'Local research tools. MIT-licensed software.\nReview reports and sensitive paths before sharing.\nSimilarity and overfitting signals require human review.',10,MUTED,PANEL).pack(anchor='w',pady=12)
        self.label(body,'KEYBOARD SHORTCUTS',9,GOLD,PANEL,True).pack(anchor='w')
        self.label(body,'Ctrl+K  Find workspace     Ctrl+B  Toggle sidebar\nCtrl+Enter  Analyze         Ctrl+S  Export JSON',10,INK,PANEL).pack(anchor='w',pady=8)
        ttk.Button(body,text='Close',style='Action.TButton',command=dialog.destroy).pack(anchor='e',pady=(8,0))
        dialog.bind('<Escape>',lambda _:dialog.destroy())
        return dialog

    def render(self):
        for child in self.cards.winfo_children(): child.destroy()
        values=summary(self.tool,self.report) if self.report is not None else [('Status','Ready'),('Execution','Local'),('Report','Not generated')]
        for label,value in values:
            card=tk.Frame(self.cards,bg=PANEL,padx=16,pady=9); card.pack(side='left',fill='x',expand=True,padx=(0,8))
            self.label(card,label.upper(),8,MUTED,PANEL,True).pack(anchor='w'); self.label(card,value,21,ACCENT,PANEL,True).pack(anchor='w',pady=(4,0))
        self.raw.configure(state='normal'); self.raw.delete('1.0','end'); self.raw.insert('1.0',json.dumps(self.report,indent=2,ensure_ascii=False,allow_nan=False) if self.report is not None else 'Run an analysis to generate a report.'); self.raw.configure(state='disabled')
        self.render_findings()
        if self.tool=='Scores' and self.report is not None:
            self.chart_select.pack(side='right'); self.threshold_var.set(self.report['threshold']); self.threshold_changed(self.threshold_var.get())
            self.threshold_panel.pack(fill='x',pady=(0,8),before=self.tabs)
        else:
            self.chart_select.pack_forget(); self.threshold_panel.pack_forget()
        self.draw()

    def draw(self):
        c=self.canvas; c.delete('all'); self.chart_points=[]; w,h=c.winfo_width(),c.winfo_height()
        if w<100 or h<100: return
        def text(x,y,t,**kw): return c.create_text(x,y,text=t,fill=kw.pop('fill',MUTED),font=(self.font,kw.pop('size',9)),**kw)
        if not self.chart:
            self.chart_title.configure(text='ANALYSIS PREVIEW')
            text(w/2,h/2-18,'Ready to analyze' if self.report is None else 'Analysis report ready',size=18,fill=INK)
            text(w/2,h/2+12,'Configure inputs and run an analysis.' if self.report is None else 'Open Findings or Full report to inspect runtime details.',width=w-40)
            return
        d=self.chart; kind=d['type']; x0,y0,x1,y1=65,32,w-45,h-45
        if kind=='bars':
            self.chart_title.configure(text='ESTIMATED MACs BY LAYER' if self.tool=='Model' else 'RESULT DISTRIBUTION  /  FIRST 12 ITEMS')
            vals=d['values'][:12]; x0=170; x1=w-65; lo=min([0]+[v for _,v in vals]); hi=max([0]+[v for _,v in vals]); span=hi-lo or 1
            step=(y1-y0)/max(1,len(vals)); zero=x0+(0-lo)/span*(x1-x0)
            for i in range(5):
                xx=x0+(x1-x0)*i/4; c.create_line(xx,y0,xx,y1,fill=LINE); text(xx,y1+15,f'{lo+span*i/4:.3g}')
            for i,(name,v) in enumerate(vals):
                mid=y0+step*(i+.5); end=x0+(v-lo)/span*(x1-x0); bar=min(24,step*.55)
                text(x0-10,mid,str(name)[:23],anchor='e'); c.create_rectangle(min(zero,end),mid-bar/2,max(zero,end),mid+bar/2,fill=COLORS[i%4],outline='')
                text(end+6 if v>=0 else end-6,mid,f'{v:,.4g}',anchor='w' if v>=0 else 'e',fill=INK)
                self.chart_points.append((end,mid,f'{name}: {v:,.5g}'))
        elif kind=='lines':
            self.chart_title.configure(text='LEARNING CURVES  /  LOGGED EPOCHS')
            xs=d['x']; series=d['series']; ys=[v for vs in series.values() for v in vs]; lo,hi=min(ys),max(ys); spread=hi-lo or 1; dx=xs[-1]-xs[0] or 1
            y0+=18
            for i in range(5):
                yy=y1-(y1-y0)*i/4; c.create_line(x0,yy,x1,yy,fill=LINE); text(x0-9,yy,f'{lo+spread*i/4:.3g}',anchor='e')
            for j,(name,vs) in enumerate(series.items()):
                color=COLORS[j%4]; pts=[(x0+(x-xs[0])/dx*(x1-x0),y1-(v-lo)/spread*(y1-y0)) for x,v in zip(xs,vs)]
                if len(pts)>1: c.create_line(*[n for p in pts for n in p],fill=color,width=2.5)
                self.chart_points.extend((px,py,f'{name} | Epoch {x:g}: {v:.5g}') for (px,py),x,v in zip(pts,xs,vs))
                for x,y in pts if len(pts)<=100 else []: c.create_oval(x-3,y-3,x+3,y+3,fill=color,outline='')
                text(x0+j*max(130,(x1-x0)/len(series)),17,name,anchor='w',fill=color)
            if self.report and 'best_epoch' in self.report:
                xx=x0+(self.report['best_epoch']-xs[0])/dx*(x1-x0); c.create_line(xx,y0,xx,y1,fill=GOLD,dash=(4,5)); text(min(x1-55,xx+6),y0+10,'Best epoch',anchor='w',fill=GOLD)
            for i in range(min(len(xs),6)):
                idx=round(i*(len(xs)-1)/max(1,min(len(xs),6)-1)); xx=x0+(xs[idx]-xs[0])/dx*(x1-x0); text(xx,y1+15,f'{xs[idx]:g}')
            text((x0+x1)/2,y1+33,'Epoch')
        elif kind=='roc':
            mode=self.chart_mode.get(); self.chart_title.configure(text=mode.upper()+'  /  BINARY PROBABILITIES'); y0+=12
            if mode=='Precision–recall':
                rows=self.report['precision_recall_curve']; xs=[r['recall'] for r in rows]; ys=[r['precision'] for r in rows]; xl,yl='Recall','Precision'; baseline=self.report['positive_prevalence']
            elif mode=='Calibration':
                rows=[r for r in self.report['calibration'] if r['count']]; xs=[r['mean_probability'] for r in rows]; ys=[r['positive_fraction'] for r in rows]; xl,yl='Mean probability','Observed positive fraction'; baseline=None
            else: xs=d['x']; ys=d['y']; xl,yl='False positive rate','True positive rate'; baseline=None
            for i in range(6):
                f=i/5; xx=x0+f*(x1-x0); yy=y1-f*(y1-y0)
                c.create_line(x0,yy,x1,yy,fill=LINE); c.create_line(xx,y0,xx,y1,fill=LINE); text(x0-10,yy,f'{f:.1f}',anchor='e'); text(xx,y1+14,f'{f:.1f}')
            if baseline is None: c.create_line(x0,y1,x1,y0,fill=MUTED,dash=(4,5))
            else: c.create_line(x0,y1-baseline*(y1-y0),x1,y1-baseline*(y1-y0),fill=MUTED,dash=(4,5))
            pts=[(x0+x*(x1-x0),y1-y*(y1-y0)) for x,y in zip(xs,ys)]
            if len(pts)>1: c.create_line(*[n for p in pts for n in p],fill=ACCENT,width=3)
            self.chart_points.extend((px,py,f'{xl}: {x:.4g} | {yl}: {y:.4g}') for (px,py),x,y in zip(pts,xs,ys))
            for x,y in pts if len(pts)<=100 else []: c.create_oval(x-3,y-3,x+3,y+3,fill=ACCENT,outline='')
            text((x0+x1)/2,y1+33,xl); text(x0,y0-13,yl,anchor='w')
        elif kind=='matrix':
            self.chart_title.configure(text='CONFUSION MATRIX  /  COUNTS'); labels=d['labels']; matrix=d['values']; n=len(labels)
            if n>20: text(w/2,h/2,'More than 20 classes. See the Full report tab.',width=w-40); return
            x0=110; y0=50; cell=min((x1-x0)/n,(y1-y0)/n); x0=max(x0,(w-cell*n)/2); maxv=max(max(r) for r in matrix) or 1
            text(w/2,14,'Predicted labels (columns) · True labels (rows)')
            for i,label in enumerate(labels):
                text(x0-8,y0+(i+.5)*cell,label[:14],anchor='e'); text(x0+(i+.5)*cell,y0-15,label[:14])
                for j,v in enumerate(matrix[i]):
                    t=v/maxv; color=f'#{int(236-215*t):02x}{int(242-139*t):02x}{int(226-145*t):02x}'
                    x,y=x0+j*cell,y0+i*cell; c.create_rectangle(x,y,x+cell,y+cell,fill=color,outline=BG)
                    self.chart_points.append((x+cell/2,y+cell/2,f'True {labels[i]} / predicted {labels[j]}: {v}'))
                    if cell>=18: text(x+cell/2,y+cell/2,str(v),fill=INK if t<.48 else '#FFFFFF',size=min(17,max(9,int(cell/5))))


def main():
    root=tk.Tk(); Studio(root); root.mainloop()


if __name__=='__main__': main()
