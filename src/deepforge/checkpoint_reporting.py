"""Optional synchronous checkpoint reporting; no invented byte progress."""
import json
from .checkpoint import verify_payload

def report_save(store, epoch, best, save, monitor=None, refresh=None, *, byte_progress=False):
    def report(stage, status, **counts):
        if monitor is not None:
            monitor.record_stage(epoch, stage, status, **counts)
            if refresh is not None:
                refresh()
    if not store.policy.enabled or (epoch % store.policy.save_every_epochs and not best):
        report("checkpoint", "skipped")
        report("verification", "skipped")
        return None
    report("checkpoint", "running")
    try:
        import time
        last = -float('inf')
        def progress(done, total):
            nonlocal last
            now = time.monotonic()
            if done == total or now - last >= 0.5:
                report('checkpoint', 'running', completed=done, total=total)
                last = now
        path = save(progress if monitor is not None else None) if byte_progress else save()
    except Exception:
        report("checkpoint", "failed")
        raise
    report("checkpoint", "completed")
    if monitor is not None:
        report("verification", "running")
        try:
            valid = verify_payload(path, json.loads(path.with_suffix(".json").read_text(encoding="utf-8")))
            if not valid:
                raise OSError("Checkpoint integrity verification failed")
        except Exception:
            report("verification", "failed")
            raise
        report("verification", "completed")
        monitor.artifact(path)
    return path
