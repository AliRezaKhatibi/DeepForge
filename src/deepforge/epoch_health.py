"""Conservative, configurable loss-trend warnings; not a diagnosis."""
import math

def epoch_health(records, epoch, *, patience=3, threshold=0.25):
    current = records.get(str(epoch), records.get(epoch, {}))
    for group in ("training", "validation"):
        metrics = current.get(group, {})
        if any(v is None or (isinstance(v,(int,float)) and not math.isfinite(v)) for v in metrics.values()):
            return "Nonfinite metric", "#b91c1c"
    pairs = []
    for i in range(max(1, epoch-patience), epoch+1):
        r = records.get(str(i), records.get(i, {}))
        t = r.get("training", {}).get("loss")
        v = r.get("validation", {}).get("loss")
        if not all(isinstance(x, (int,float)) and math.isfinite(x) for x in (t,v)):
            return "Unassessed", "#808080"
        pairs.append((t,v))
    if len(pairs) < 2:
        return "Unassessed", "#808080"
    streak = 0
    for a,b in zip(pairs,pairs[1:]):
        streak = streak+1 if b[0]<a[0] and b[1]>a[1] else 0
    if streak >= patience and pairs[-1][1]-pairs[0][1] >= threshold*max(abs(pairs[0][1]),1e-12):
        return "Severe overfitting warning", "#b91c1c"
    if streak >= 2:
        return "Possible overfitting", "#d97706"
    if streak:
        return "Watch", "#b77924"
    return "No trend warning", "#24765c"
