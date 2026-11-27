"""Conservative, configurable signals; not causal diagnoses."""
import math
from collections import Counter


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def diagnose(history, *, train_key="train_loss", val_key="val_loss",
             patience=3, relative_change=.05, gradient_min=None, gradient_max=None):
    """Compare the last patience+1 comparable epoch observations.

    Only loss-like quantities (lower is better) are supported. Missing/nonfinite
    observations prevent trend inference; no universal gradient thresholds.
    """
    if type(patience) is not int or patience < 1:
        raise ValueError("patience must be a positive integer")
    if not _number(relative_change) or relative_change <= 0:
        raise ValueError("relative_change must be finite and positive")
    for limit in (gradient_min, gradient_max):
        if limit is not None and (not _number(limit) or limit <= 0):
            raise ValueError("gradient limits must be positive and finite")
    if gradient_min is not None and gradient_max is not None and gradient_min >= gradient_max:
        raise ValueError("gradient_min must be below gradient_max")
    rows = list(history)
    findings = []
    def add(code, severity, evidence):
        findings.append({"code": code, "severity": severity, "evidence": evidence})
    if not rows:
        return {"findings": [], "trend_available": False}
    latest = rows[-1]["metrics"]
    invalid = [k for k, v in latest.items() if v is None or not _number(v)]
    if invalid:
        add("invalid_or_missing_metrics", "warning", {"metrics": invalid})
    norm = latest.get("gradient_norm")
    if _number(norm) and norm >= 0:
        if gradient_min is not None and norm < gradient_min:
            add("small_gradient_signal", "warning", {"norm": norm, "threshold": gradient_min})
        if gradient_max is not None and norm > gradient_max:
            add("large_gradient_signal", "warning", {"norm": norm, "threshold": gradient_max})
    window = [row["metrics"] for row in rows[-(patience+1):]]
    available = len(window) == patience+1 and all(
        _number(row.get(k)) for row in window for k in (train_key, val_key))
    if available:
        t0, t1 = window[0][train_key], window[-1][train_key]
        v0, v1 = window[0][val_key], window[-1][val_key]
        td = (t1-t0)/max(abs(t0), 1e-12)
        vd = (v1-v0)/max(abs(v0), 1e-12)
        evidence = {"train_relative_change": td, "val_relative_change": vd, "intervals": patience}
        if td <= -relative_change and vd >= relative_change:
            add("possible_overfitting", "warning", evidence)
        elif abs(td) < relative_change and abs(vd) < relative_change:
            add("loss_plateau", "review", evidence)
    return {"findings": findings, "trend_available": available}


def class_distribution(labels):
    """Count caller-provided class IDs, not inferred image labels."""
    counts = Counter()
    for label in labels:
        if type(label) is not int or label < 0:
            raise ValueError("Expected nonnegative integer class IDs")
        counts[str(label)] += 1
    total = sum(counts.values())
    return {"total_labels": total, "counts": dict(counts),
            "fractions": {k: n/total for k,n in counts.items()} if total else {}}
