"""Local VS Code / terminal example: real CPU training and optional desktop controls."""
import tempfile
import time
from pathlib import Path

import torch
from deepforge.checkpoint import CheckpointPolicy, CheckpointStore
from deepforge.checkpoint_torch import TorchCheckpoint
from deepforge.monitor import TrainingMonitor
from deepforge.session import TrainingSession


def main():
    torch.manual_seed(42)
    run_root=Path(tempfile.mkdtemp(prefix="deepforge-desktop-"))
    store=CheckpointStore(run_root,"example",CheckpointPolicy(keep_last=2))
    epochs,batches=8,40
    x=torch.randn(320,4); y=x.sum(dim=1,keepdim=True)
    xval=torch.randn(80,4); yval=xval.sum(dim=1,keepdim=True)
    def build():
        model=torch.nn.Linear(4,1); optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=.8)
        monitor=TrainingMonitor(store.root/"live.json",epochs=epochs,batches_per_epoch=batches,resource_interval=2)
        checkpoint=TorchCheckpoint(store,model,optimizer,scheduler=scheduler,monitor=monitor)
        session=TrainingSession(monitor,checkpoint,{"dataset":"synthetic regression","seed":42})
        session.desktop_metadata={"run_id":"example","dataset":{"name":"Synthetic regression","training samples":320,"validation samples":80}}
        return model,optimizer,scheduler,monitor,checkpoint,session
    model,optimizer,scheduler,monitor,checkpoint,session=build()
    desktop=session.open_desktop(allow_resume=True)
    print(f"Training snapshot: {monitor.path}")
    start=1; step=0
    while start<=epochs:
        for epoch in range(start,epochs+1):
            for batch in range(1,batches+1):
                xx=x[(batch-1)*8:batch*8]; yy=y[(batch-1)*8:batch*8]
                optimizer.zero_grad(); loss=(model(xx)-yy).square().mean();loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),100).item();optimizer.step();step+=1
                monitor.batch(epoch=epoch,batch=batch,samples=len(xx),metrics={"loss":loss.item(),"gradient_norm":grad,"lr":optimizer.param_groups[0]["lr"]})
                # Demonstration only: make the live UI easy to inspect.
                time.sleep(.045)
            monitor.record_stage(epoch,"validation","running",completed=0,total=1)
            with torch.no_grad(): val=(model(xval)-yval).square().mean().item()
            monitor.epoch_end(epoch,{"loss":val});monitor.record_stage(epoch,"validation","completed",completed=1,total=1)
            scheduler.step()
            if session.epoch_boundary(epoch,global_step=step): break
            checkpoint.save_epoch(epoch,global_step=step,metadata=session.metadata)
        if monitor.status!="paused": monitor.finish();break
        if epoch>=epochs: print("Paused at final epoch; no epochs remain.");break
        print("Paused. Choose Resume in the desktop window or close it to leave the checkpoint saved.")
        while desktop.poll() is None and not session.wait_for_resume_request(timeout=.5): pass
        if desktop.poll() is not None: break
        model,optimizer,scheduler,monitor,checkpoint,session=build()
        restored=session.restore(); start=restored["epoch"]+1;step=restored["global_step"]
    print(f"Final status: {monitor.status}; run folder: {store.root}")


if __name__=="__main__": main()
