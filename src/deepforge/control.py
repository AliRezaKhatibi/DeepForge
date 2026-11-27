"""File-based control channel for a single training writer and trusted operator."""
import json
from pathlib import Path
from .checkpoint import atomic_json

class StopControl:
    def __init__(self, path):
        self.path = Path(path)

    def request(self, *, after_next=True):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, {"action":"stop", "after_next":bool(after_next)})

    def cancel(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, {"action":"cancel"})

    def consume(self):
        # Rename first: a later request must not be deleted by this reader.
        pending = self.path.with_name(self.path.name+".processing")
        try:
            self.path.replace(pending)
        except FileNotFoundError:
            if not pending.exists():
                return None
        data = json.loads(pending.read_text(encoding="utf-8"))
        pending.unlink()
        if data.get("action") not in ("stop","cancel"):
            raise ValueError("Unknown control action")
        return data
