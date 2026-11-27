"""Composable, read-only quality gates for deep learning pipelines."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import json
import math
import operator

from .datasets import inspect_dataset
from .audit import audit_groups, audit_images, audit_yolo
from .metrics import evaluate_csv, classification_metrics

_OPERATORS = {"eq": operator.eq, "ne": operator.ne, "ge": operator.ge,
              "gt": operator.gt, "le": operator.le, "lt": operator.lt}


@dataclass(frozen=True)
class Check:
    name: str
    run: Callable[[], dict]
    rules: tuple = field(default_factory=tuple)


class AuditFailed(RuntimeError):
    """One or more quality checks failed or could not run."""


def _resolve(report, key):
    value = report
    for part in key.split("."):
        value = value[part]
    return value


def run_checks(checks, *, raise_on_failure=False):
    """Execute every check, returning a JSON-compatible report.

    Rules are (field.path, operator, expected) tuples. Exceptions fail closed.
    A check without rules verifies execution only, not data quality.
    """
    checks = list(checks)
    if not checks:
        raise ValueError("At least one check is required")
    names = [c.name for c in checks]
    if len(set(names)) != len(names):
        raise ValueError("Check names must be unique")
    results = []
    for check in checks:
        entry = {"name": check.name, "status": "error", "rules": []}
        try:
            report = check.run()
            if not isinstance(report, dict):
                raise TypeError("Checks must return dictionaries")
            json.dumps(report, allow_nan=False)
            for key, op, expected in check.rules:
                actual = _resolve(report, key)
                passed = bool(_OPERATORS[op](actual, expected))
                entry["rules"].append({"field": key, "operator": op,
                                       "expected": expected, "actual": actual,
                                       "passed": passed})
            entry["report"] = report
            entry["status"] = "passed" if all(r["passed"] for r in entry["rules"]) else "failed"
            json.dumps(entry, allow_nan=False)
        except Exception as exc:
            entry = {"name": check.name, "status": "error",
                     "error": type(exc).__name__, "detail": str(exc), "rules": []}
        results.append(entry)
    counts = {s: sum(r["status"] == s for r in results)
              for s in ("passed", "failed", "error")}
    summary = {"schema_version": 1, "ok": counts["passed"] == len(results),
               "counts": counts, "checks": results}
    if raise_on_failure:
        require_pass(summary)
    return summary


def require_pass(report):
    """Stop the calling pipeline when any audit fails."""
    if not report["ok"]:
        names = ", ".join(c["name"] for c in report["checks"] if c["status"] != "passed")
        raise AuditFailed(f"Quality gate failed: {names}")


def save_report(report, path):
    """Write a report to an explicitly selected path without overwriting files."""
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload)


def dataset_check(root, *, min_samples=1, required_splits=("train", "val", "test")):
    """Check nonempty required classification splits and exact duplicate leakage."""
    if type(min_samples) is not int or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    splits = tuple(required_splits)
    if not splits or any(s not in {"train", "val", "validation", "test"} for s in splits):
        raise ValueError("Invalid required splits")
    def run():
        report = inspect_dataset(root, hash_files=True)
        found = report["splits"]
        report["required_splits_ok"] = all(
            found.get(s, found.get("validation" if s == "val" else "val" if s == "validation" else s, {})).get("total", 0) >= min_samples
            for s in splits)
        return report
    return Check("dataset", run, (("required_splits_ok", "eq", True),
                                  ("cross_split_duplicate_groups", "eq", 0)))


def group_check(csv, *, group_column="group_id", split_column="split"):
    return Check("groups", lambda: audit_groups(csv, group_column=group_column,
                 split_column=split_column), (("overlapping_groups", "eq", 0),))


def image_check(root, *, max_images=2000, similarity=False, threshold=4):
    rules = [("invalid_images", "eq", 0), ("unsupported_images", "eq", 0),
             ("scan_truncated", "eq", False), ("decoded", "gt", 0)]
    if similarity:
        rules.append(("candidate_pair_count", "eq", 0))
    return Check("images", lambda: audit_images(root, max_images=max_images,
                 similarity=similarity, threshold=threshold), tuple(rules))


def yolo_check(root, *, classes=None):
    return Check("yolo", lambda: audit_yolo(root, classes=classes),
                 (("images", "gt", 0), ("missing_labels", "eq", []),
                  ("orphan_labels", "eq", []), ("ambiguous_image_stems", "eq", []),
                  ("issues", "eq", []), ("invalid_rows", "eq", 0)))


def classification_check(*, csv=None, y_true=None, y_pred=None,
                         min_accuracy=0.0, true_column="y_true", pred_column="y_pred"):
    if not math.isfinite(min_accuracy) or not 0 <= min_accuracy <= 1:
        raise ValueError("min_accuracy must be in [0, 1]")
    if csv is not None:
        if y_true is not None or y_pred is not None:
            raise ValueError("Choose CSV or in-memory labels")
        run = lambda: evaluate_csv(csv, true_column, pred_column)
    else:
        if y_true is None or y_pred is None:
            raise ValueError("Both y_true and y_pred are required")
        truth, pred = list(y_true), list(y_pred)
        run = lambda: classification_metrics(truth, pred)
    return Check("classification", run, (("accuracy", "ge", min_accuracy),))
