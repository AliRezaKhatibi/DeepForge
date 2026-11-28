import json
from pathlib import Path

import pytest

from deepforge.checkpoint import atomic_json
from deepforge.control import StopControl
from deepforge.desktop_monitor import DesktopChannel, observations, series_for, launch_desktop, export_report, model_metadata, TABS
from deepforge.monitor import TrainingMonitor


def test_desktop_is_optional_and_read_only(tmp_path, monkeypatch):
    path = tmp_path / "live.json"
    monitor = TrainingMonitor(path, epochs=3, batches_per_epoch=2)
    channel=DesktopChannel(path)
    with pytest.raises(RuntimeError, match="read-only"):
        channel.command("stop")
    atomic_json(tmp_path/"desktop.json", {"controls":True})
    channel.command("stop",after_next=False)
    assert StopControl(tmp_path/"control.json").consume()=={"action":"stop","after_next":False}
    channel.command("cancel")
    assert StopControl(tmp_path/"control.json").consume()=={"action":"cancel"}
    channel.command("save")
    assert json.loads((tmp_path/"manual-save.json").read_text())["requested"]
    monitor.finish()
    with pytest.raises(RuntimeError, match="not running"):
        channel.command("save")
    import subprocess, sys
    monkeypatch.setattr(subprocess,"Popen",lambda args:args)
    args=launch_desktop(path)
    assert args[:3]==[sys.executable,"-m","deepforge.desktop_monitor"]
    assert str(path.resolve()) in args


def test_desktop_observations_no_future_or_fake_values(tmp_path):
    m=TrainingMonitor(tmp_path/"live.json",epochs=4,batches_per_epoch=2)
    m.batch(epoch=1,batch=1,metrics={"loss":.6})
    values,checks,health,_,val_epoch=observations(m.snapshot(),{})
    assert dict(values)["Accuracy"] is None and val_epoch is None and health=="Unassessed"
    assert not checks.get("Hardware (GPU / CPU)")
    m.batch(epoch=1,batch=2,metrics={"loss":.5})
    m.epoch_end(1,{"loss":.4,"accuracy":.85})
    m.batch(epoch=2,batch=1,metrics={"loss":float("nan")})
    values,_,health,_,val_epoch=observations(m.snapshot(),{})
    assert health=="Nonfinite metric" and val_epoch==1 and dict(values)["Accuracy"]==.85
    assert series_for(m.snapshot(),"loss")[0][2]==[(1,.5)]
    export_report(tmp_path/"report.json",m.snapshot())
    assert json.loads((tmp_path/"report.json").read_text())["epoch"]==2
    export_report(tmp_path/"report.csv",m.snapshot(),csv_format=True)
    assert "validation_accuracy" in (tmp_path/"report.csv").read_text()


def test_desktop_window_and_tabs(tmp_path):
    import time
    tk=pytest.importorskip("tkinter")
    from deepforge.desktop_monitor import DesktopMonitor
    try: root=tk.Tk()
    except tk.TclError: pytest.skip("A graphical display is required")
    try:
        m=TrainingMonitor(tmp_path/"live.json",epochs=3,batches_per_epoch=2)
        atomic_json(tmp_path/"desktop.json",{"controls":True,"model":"Demo","dataset":{"name":"Fixture"}})
        m.batch(epoch=1,batch=1,metrics={"loss":.5})
        app=DesktopMonitor(root,m.path,interval=10)
        for size in ("1440x880","720x540"):
            root.geometry(size)
            deadline=time.monotonic()+3
            while time.monotonic()<deadline:
                root.update()
                if app.canvas.winfo_width()>=300 and root.winfo_viewable():break
                time.sleep(.02)
            assert app.canvas.winfo_width()>=300, "Native canvas was not laid out"
            for name in TABS:
                app.nav[name].invoke(); root.update()
                assert app.tab==name and app.canvas.find_all()
                assert app.canvas.bbox("all")[2] <= app.canvas.winfo_width()+4
        app.cycle_theme(); root.update_idletasks(); assert app.theme=="dark"
        m.batch(epoch=1,batch=2,metrics={"loss":.3}); m.epoch_end(1,{"loss":.4})
        root.after_cancel(app._timer); app.refresh()
        assert app.data["completed_batches"]==2
        app.select("Training"); app.set_history(10); assert app.history_limit==10
        app.save.invoke(); assert (tmp_path/"manual-save.json").is_file()
        app.cancel.invoke(); assert StopControl(tmp_path/"control.json").consume()=={"action":"cancel"}
        m.pause(); root.after_cancel(app._timer); app.refresh(); assert app.save.cget("state")=="disabled"
    finally:
        if root.winfo_exists(): root.destroy()


def test_desktop_session_real_checkpoint(tmp_path,monkeypatch):
    torch=pytest.importorskip("torch")
    from deepforge.checkpoint import CheckpointStore,CheckpointPolicy
    from deepforge.checkpoint_torch import TorchCheckpoint
    from deepforge.session import TrainingSession
    import deepforge.desktop_monitor as desktop
    monkeypatch.setattr(desktop,"launch_desktop",lambda *a,**k:object())
    store=CheckpointStore(tmp_path,"run",CheckpointPolicy(enabled=False))
    def make():
        model=torch.nn.Linear(2,1); opt=torch.optim.Adam(model.parameters())
        scheduler=torch.optim.lr_scheduler.StepLR(opt,1,.9)
        monitor=TrainingMonitor(store.root/"live.json",epochs=4,batches_per_epoch=1)
        session=TrainingSession(monitor,TorchCheckpoint(store,model,opt,scheduler=scheduler,monitor=monitor),{"dataset":"fixture"})
        return session,model,opt,scheduler
    s,model,opt,sch=make();s.open_desktop(allow_resume=True); c=DesktopChannel(s.monitor.path)
    assert c.metadata['parameters']==3 and c.metadata['layers'][0]['type']=='Linear'
    def step(epoch):
        opt.zero_grad();loss=model(torch.ones(1,2)).square().mean();loss.backward();opt.step();sch.step()
        s.monitor.batch(epoch=epoch,batch=1,metrics={"loss":loss.item()});s.monitor.epoch_end(epoch,{"loss":loss.item()})
    c.command("save");step(1);assert not s.epoch_boundary(1,global_step=1)
    assert CheckpointStore(store.root,"manual").latest() is not None
    assert c.state["saved_epoch"]==1
    c.command("stop",after_next=False);step(2);assert s.epoch_boundary(2,global_step=2)
    assert s.monitor.status=="paused";c.command("resume");assert s.wait_for_resume_request(timeout=0)
    expected={k:v.clone() for k,v in model.state_dict().items()}
    s2,m2,o2,sch2=make();assert s2.restore()["epoch"]==2
    assert all(torch.equal(v,m2.state_dict()[k]) for k,v in expected.items())
    assert o2.state and sch2.last_epoch==2
