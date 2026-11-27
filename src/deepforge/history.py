"""Summarize finite numeric training histories without inferring causality."""
import csv
import math


def _number(row, column, line):
    try:
        value = float(row[column])
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"Invalid {column!r} at CSV line {line}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {column!r} at CSV line {line}")
    return value


def analyze_history(path, *, metric="val_loss", mode="min", train_metric=None, epoch_column="epoch"):
    if mode not in {"min", "max"}:
        raise ValueError("mode must be min or max")
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {epoch_column, metric} | ({train_metric} if train_metric else set())
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing CSV columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Training history is empty")
    epochs = [_number(r, epoch_column, i) for i, r in enumerate(rows, 2)]
    if any(b <= a for a, b in zip(epochs, epochs[1:])):
        raise ValueError("Epochs must be strictly increasing")
    values = [_number(r, metric, i) for i, r in enumerate(rows, 2)]
    best = (min if mode == "min" else max)(range(len(values)), key=values.__getitem__)
    worse = values[-1] > values[best] if mode == "min" else values[-1] < values[best]
    result = {"metric": metric, "mode": mode, "epochs_logged": len(rows), "best_epoch": epochs[best],
              "best_value": values[best], "final_epoch": epochs[-1], "final_value": values[-1],
              "validation_worse_than_best": worse}
    if train_metric:
        train = [_number(r, train_metric, i) for i, r in enumerate(rows, 2)]
        improved = train[-1] < train[best] if mode == "min" else train[-1] > train[best]
        result["possible_overfitting_signal"] = bool(worse and improved)
        result["train_metric"] = train_metric
    return result
