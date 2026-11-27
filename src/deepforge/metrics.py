"""Single-label multiclass classification metrics with explicit zero division."""
import csv
from collections import Counter


def _ratio(n, d):
    return n / d if d else 0.0


def classification_metrics(y_true, y_pred):
    y_true, y_pred = list(y_true), list(y_pred)
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Labels must have equal, nonzero lengths")
    if any(not isinstance(x, str) or not x.strip() for x in y_true + y_pred):
        raise ValueError("Labels must be nonempty strings")
    labels = sorted(set(y_true) | set(y_pred))
    pairs = Counter(zip(y_true, y_pred))
    matrix = [[pairs[a, b] for b in labels] for a in labels]
    per_class = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision, recall = _ratio(tp, predicted), _ratio(tp, support)
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": _ratio(2 * precision * recall, precision + recall), "support": support}
    averages = {m: sum(v[m] for v in per_class.values()) / len(labels) for m in ("precision", "recall", "f1")}
    weighted = {m: sum(v[m] * v["support"] for v in per_class.values()) / len(y_true) for m in averages}
    return {"samples": len(y_true), "accuracy": sum(matrix[i][i] for i in range(len(labels))) / len(y_true),
            "labels": labels, "confusion_matrix": matrix, "per_class": per_class,
            "macro": averages, "weighted": weighted, "zero_division": 0}


def evaluate_csv(path, true_column="y_true", pred_column="y_pred"):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {true_column, pred_column}.issubset(reader.fieldnames):
            raise ValueError("CSV must contain the requested label columns")
        rows = list(reader)
    return classification_metrics([r[true_column] for r in rows], [r[pred_column] for r in rows])
