"""Read-only Tk desktop viewer for a local training snapshot."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    args = parser.parse_args(argv)
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    root.title("DeepForge | Live Training · XVARNA")
    root.geometry("960x700")
    root.minsize(620, 480)
    jade, cream, gold = "#123B32", "#F3EFE5", "#E8C77D"
    root.configure(bg=cream)
    tk.Label(root, text="DEEPFORGE  /  LIVE TRAINING", bg=jade, fg=gold,
             font=("Arial", 20, "bold"), pady=18).pack(fill="x")
    status = tk.StringVar(value="Waiting for snapshot")
    tk.Label(root, textvariable=status, bg=cream, fg=jade,
             font=("Arial", 14, "bold"), pady=12).pack()
    progress = ttk.Progressbar(root, maximum=100)
    progress.pack(fill="x", padx=24, pady=8)
    details = tk.StringVar()
    tk.Label(root, textvariable=details, bg=cream, fg=jade, justify="left").pack(pady=8)
    metric_choice = tk.StringVar(value="loss")
    metric_select = ttk.Combobox(root, textvariable=metric_choice, state="readonly")
    metric_select.pack(fill="x", padx=24)
    chart = tk.Canvas(root, bg="#FFFCF5", highlightthickness=0, height=220)
    chart.pack(fill="both", expand=True, padx=24, pady=8)
    preview_label = tk.Label(root, bg=cream)
    preview_label.image = None
    preview_value = None
    artifacts = tk.StringVar()
    tk.Label(root, textvariable=artifacts, bg=cream, fg=jade, justify="left",
             wraplength=580).pack(pady=8)
    def duration(value):
        return "unknown" if value is None else f"{int(value)//60}m {int(value)%60:02d}s"
    def refresh():
        nonlocal preview_value
        try:
            data = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
            if data["schema_version"] != 1:
                raise ValueError("Unsupported snapshot schema")
            status.set(f'{data["status"].upper()}   |   Epoch {data["epoch"]}/{data["epochs"]}   |   Batch {data["batch"]}/{data["batches_per_epoch"]}')
            progress["value"] = 100 * data["progress"]
            details.set(f'{data["progress"]:.1%}  •  {data["completed_batches"]}/{data["total_batches"]} batches  •  {data["samples"]} samples\n'
                        f'Elapsed {duration(data["elapsed_seconds"])}  •  ETA {duration(data["eta_seconds"])}  •  {data["batches_per_second"]:.2f} batches/s  •  {data["samples_per_second"]:.1f} samples/s')
            chart.delete("all")
            w, h = max(chart.winfo_width(), 100), max(chart.winfo_height(), 100)
            history = data["history"]
            keys = sorted({k for row in history for k,v in row["metrics"].items() if v is not None})
            metric_select["values"] = keys
            key = metric_choice.get() if metric_choice.get() in keys else (keys[0] if keys else None)
            if key:
                metric_choice.set(key)
            chart.create_text(15, 15, anchor="nw", text=f"Recent {key or 'metric'} history", fill=jade, font=("Arial", 12, "bold"))
            pts = [(r["step"], r["metrics"][key]) for r in history if key and r["metrics"].get(key) is not None]
            if pts:
                lo, hi = min(v for _,v in pts), max(v for _,v in pts)
                x0, x1 = pts[0][0], pts[-1][0]
                coords = [(40 + (x-x0)/max(x1-x0,1)*(w-65), h-30-(v-lo)/max(hi-lo,1e-12)*(h-70)) for x,v in pts]
                if len(coords)>1:
                    chart.create_line(*[v for xy in coords for v in xy], fill=jade, width=2)
                for x,y in coords[-1:]:
                    chart.create_oval(x-4,y-4,x+4,y+4,fill=gold,outline=jade)
                chart.create_text(w-15, 15, anchor="ne", text=f"{pts[-1][1]:.5g}", fill=jade)
            validation = data.get("epoch_history", [])
            extra = "\nValidation: " + "  |  ".join(f"{k}: {v:.4g}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in validation[-1]["metrics"].items()) if validation else ""
            extra += "\nResources: " + "  |  ".join(f"{k}: {v:.4g}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in data["resources"].items()) if data.get("resources") else ""
            findings = data.get("diagnostics", {}).get("findings", [])
            if findings:
                extra += "\nDiagnostics: " + " | ".join(str(x["code"]) for x in findings)
            distribution = data.get("batch_distribution", {}).get("counts", {})
            if distribution:
                extra += "\nBatch labels: " + str(distribution)
            encoded = data.get("preview_png")
            if encoded != preview_value:
                if encoded is None:
                    preview_label.pack_forget()
                    preview_label.image = None
                elif isinstance(encoded, str) and len(encoded) <= 800000 and encoded.startswith("iVBORw0KGgo"):
                    image = tk.PhotoImage(data=encoded, format="png")
                    preview_label.image = image
                    preview_label.configure(image=image)
                    preview_label.pack(pady=4)
                preview_value = encoded
            artifacts.set("Output files: " + (" | ".join(f'{a["name"]} ({a["bytes"] if a["exists"] else "missing"} bytes)' for a in data["artifacts"][-5:]) or "None registered") + extra)
        except (OSError, ValueError, KeyError, TypeError, tk.TclError) as exc:
            status.set(f"Snapshot unavailable: {type(exc).__name__}")
        root.after(1000, refresh)
    root.after(100, refresh)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
