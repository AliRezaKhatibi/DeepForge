"""Synthetic live progress demo; run viewer in a separate terminal."""
import time
from deepforge.monitor import TrainingMonitor

if __name__ == "__main__":
    monitor = TrainingMonitor("live.json", epochs=3, batches_per_epoch=20)
    try:
        for epoch in range(1,4):
            for batch in range(1,21):
                time.sleep(.15)
                monitor.batch(epoch=epoch,batch=batch,samples=16,
                              metrics={"loss": 1/(epoch + batch/20)})
    except BaseException:
        monitor.finish(failed=True)
        raise
    else:
        monitor.finish()
