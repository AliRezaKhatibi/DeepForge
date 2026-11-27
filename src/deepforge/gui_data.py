"""UI-independent adapters for the desktop workspace."""
import csv
from .datasets import inspect_dataset
from .history import analyze_history
from .metrics import evaluate_csv
from .environment import inspect_environment
from .models import inspect_model
from .audit import audit_images, audit_groups, audit_yolo
from .advanced import read_binary_scores, compare_runs, reproducibility_manifest

TOOLS = {
    "Dataset": ("Dataset inspector", "Inspect image counts and class balance by split."),
    "Leakage": ("Leakage detector", "Find byte-identical images across dataset splits."),
    "Training": ("Training analyzer", "Identify the best epoch and inspect learning curves."),
    "Metrics": ("Classification evaluator", "Compare predictions and visualize the confusion matrix."),
    "Model": ("Model inspector", "Inspect a locally constructed PyTorch MLP; no checkpoint loading."),
    "Environment": ("Environment inspector", "Review Python, PyTorch and CUDA availability."),
    "Images": ("Image integrity & similarity", "Decode images and flag bounded cross-split perceptual matches."),
    "Groups": ("Group leakage audit", "Detect patient, subject or group IDs shared across splits."),
    "YOLO": ("YOLO label audit", "Validate normalized detection boxes, class IDs and label pairing."),
    "Compare": ("Experiment comparison", "Rank comparable validation histories and report rejected runs."),
    "Scores": ("Binary score evaluation", "Measure ROC-AUC, average precision, calibration and threshold metrics."),
    "Manifest": ("Reproducibility manifest", "Capture runtime, dependency versions and optional file SHA-256."),
}


def analyze(tool, options):
    """Run an explicitly selected analysis. Returns a report and optional chart data."""
    if tool in ("Dataset", "Leakage"):
        report = inspect_dataset(options["path"], hash_files=tool == "Leakage")
        chart = {"type": "bars", "values": [(s, v["total"]) for s, v in report["splits"].items()]}
    elif tool == "Training":
        args = {k: options[k] for k in ("metric", "mode", "epoch_column")}
        args["train_metric"] = options.get("train_metric") or None
        report = analyze_history(options["path"], **args)
        with open(options["path"], encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        chart = {"type": "lines", "x": [float(r[args["epoch_column"]]) for r in rows],
                 "series": {k: [float(r[k]) for r in rows] for k in dict.fromkeys(filter(None, [args["metric"], args["train_metric"]]))}}
    elif tool == "Metrics":
        report = evaluate_csv(options["path"], options["true_column"], options["pred_column"])
        chart = {"type": "matrix", "labels": report["labels"], "values": report["confusion_matrix"]}
    elif tool == "Model":
        sizes = [int(options[k]) for k in ("inputs", "hidden", "outputs")]
        if any(n < 1 or n > 4096 for n in sizes):
            raise ValueError("Layer widths must be integers between 1 and 4096.")
        if sizes[0]*sizes[1] + sizes[1]*sizes[2] > 10_000_000:
            raise ValueError("The preview model is limited to 10 million weights.")
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(sizes[0], sizes[1]), torch.nn.ReLU(), torch.nn.Linear(sizes[1], sizes[2]))
        report = inspect_model(model)
        report["architecture"] = sizes
        chart = {"type": "bars", "values": [("Trainable", report["trainable_parameters"]), ("Frozen", report["frozen_parameters"])]}
    elif tool == "Images":
        report = audit_images(options['path'], similarity=options.get('similarity', 'yes') == 'yes', threshold=int(options.get('threshold', '4')), max_images=int(options.get('max_images', '2000')))
        chart = {'type': 'bars', 'values': [('Decoded', report['decoded']), ('Invalid', report['invalid_images']), ('Unsupported', report['unsupported_images'])]}
    elif tool == "Groups":
        report = audit_groups(options['path'], group_column=options.get('group_column', 'group_id'), split_column=options.get('split_column', 'split'))
        chart = {'type': 'bars', 'values': list(report['split_counts'].items())}
    elif tool == "YOLO":
        report = audit_yolo(options['path'], classes=int(options['classes']) if options.get('classes') else None)
        chart = {'type': 'bars', 'values': list(report['class_counts'].items()) or [('Valid boxes', 0)]}
    elif tool == "Compare":
        report = compare_runs(options['path'], metric=options.get('metric', 'val_loss'), mode=options.get('mode', 'min'), train_metric=options.get('train_metric') or None, epoch_column=options.get('epoch_column', 'epoch'))
        chart = {'type': 'bars', 'values': [(r['run'], r['best_value']) for r in report['ranking'][:12]]}
    elif tool == "Scores":
        report = read_binary_scores(options['path'], true_column=options.get('true_column', 'y_true'), score_column=options.get('score_column', 'score'), positive_label=options.get('positive_label', '1'), threshold=float(options.get('threshold', '.5')))
        chart = {'type': 'roc', 'x': [r['fpr'] for r in report['roc_curve']], 'y': [r['tpr'] for r in report['roc_curve']]}
    elif tool == "Manifest":
        report, chart = reproducibility_manifest(options.get('path') or None), None
    elif tool == "Environment":
        report, chart = inspect_environment(), None
    else:
        raise ValueError(f"Unknown tool: {tool}")
    return report, chart


def summary(tool, report):
    if tool in ("Dataset", "Leakage"):
        values = [("Image files", f'{report["total"]:,}'), ("Splits", str(len(report["splits"])))]
        if tool == "Leakage":
            values.append(("Duplicate groups", str(report["cross_split_duplicate_groups"])))
        return values
    if tool == "Training":
        return [("Epochs", str(report["epochs_logged"])), ("Best epoch", f'{report["best_epoch"]:g}'), ("Best metric", f'{report["best_value"]:.5g}')]
    if tool == "Metrics":
        return [("Samples", str(report["samples"])), ("Accuracy", f'{report["accuracy"]:.1%}'), ("Macro F1", f'{report["macro"]["f1"]:.3f}')]
    if tool == "Model":
        return [("Parameters", f'{report["parameters"]:,}'), ("Trainable", f'{report["trainable_parameters"]:,}'), ("Parameter MiB", f'{report["parameter_bytes"]/1048576:.3f}')]
    if tool == "Images":
        return [('Decoded', str(report['decoded'])), ('Invalid', str(report['invalid_images'])), ('Similar pairs', str(report['candidate_pair_count']))]
    if tool == "Groups":
        return [('Samples', str(report['samples'])), ('Groups', str(report['groups'])), ('Overlaps', str(report['overlapping_groups']))]
    if tool == "YOLO":
        return [('Images', str(report['images'])), ('Valid boxes', str(report['valid_boxes'])), ('Label issues', str(len(report['issues'])+len(report['missing_labels'])+len(report['orphan_labels'])+len(report['ambiguous_image_stems'])))]
    if tool == "Compare":
        return [('Valid runs', str(report['valid_runs'])), ('Rejected', str(report['rejected_runs'])), ('Best value', f"{report['ranking'][0]['best_value']:.5g}")]
    if tool == "Scores":
        return [('ROC-AUC', f"{report['roc_auc']:.4f}"), ('Average precision', f"{report['average_precision']:.4f}"), ('Brier score', f"{report['brier_score']:.4f}")]
    if tool == "Manifest":
        return [('Python', report['python']), ('Packages tracked', str(len(report['packages']))), ('Fingerprint', 'SHA-256' if report['file'] else 'Not requested')]
    torch = report["pytorch"]
    return [("Python", report["python"]), ("Platform", report["system"]), ("CUDA", "Available" if torch.get("cuda_available") else "Unavailable / unknown")]


def preview_threshold(report, threshold):
    """Exact confusion counts from the accumulated ROC curve, including ties.

    This is a preview of the recorded probabilities, not a recommendation.
    """
    threshold = float(threshold)
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0, 1]')
    p = round(report['samples'] * report['positive_prevalence'])
    n = report['samples'] - p
    point = report['roc_curve'][0]
    for candidate in report['roc_curve'][1:]:
        if candidate['threshold'] < threshold:
            break
        point = candidate
    tp, fp = round(point['tpr'] * p), round(point['fpr'] * n)
    fn, tn = p-tp, n-fp
    return {'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn,
            'precision':tp/(tp+fp) if tp+fp else 0.0,
            'recall':tp/p, 'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}
