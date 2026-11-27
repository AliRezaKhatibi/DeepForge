"""Reproducible binary scoring, experiment comparison and runtime manifests."""
import csv
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__
from .datasets import file_digest
from .environment import inspect_environment
from .history import analyze_history
from .metrics import classification_metrics


def read_binary_scores(path, *, true_column='y_true', score_column='score', positive_label='1', threshold=.5, bins=10):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1 or not 2 <= bins <= 100:
        raise ValueError('Threshold must be in [0, 1]; calibration bins must be in [2, 100]')
    if not positive_label.strip():
        raise ValueError('Positive label must not be empty')
    pairs = []
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {true_column, score_column}.issubset(reader.fieldnames or []):
            raise ValueError('CSV must contain label and score columns')
        for n, row in enumerate(reader, 2):
            label = (row.get(true_column) or '').strip()
            try:
                score = float(row[score_column])
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Invalid score at line {n}') from exc
            if not label or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'Expected nonempty label and probability in [0, 1] at line {n}')
            pairs.append((label, score))
    labels = {label for label, _ in pairs}
    if len(labels) != 2 or positive_label not in labels:
        raise ValueError('Exactly two observed labels, including the positive label, are required')
    negative = next(iter(labels - {positive_label}))
    y = [label for label, _ in pairs]
    pred = [positive_label if score >= threshold else negative for _, score in pairs]
    metrics = classification_metrics(y, pred)
    ranked = sorted(((score, label == positive_label) for label, score in pairs), reverse=True)
    p = sum(int(pos) for _, pos in ranked); n = len(ranked) - p
    roc = [{'fpr': 0.0, 'tpr': 0.0, 'threshold': None}]
    pr = [{'recall': 0.0, 'precision': 1.0, 'threshold': None}]
    tp = fp = i = 0
    auc = ap = 0.0
    while i < len(ranked):
        score = ranked[i][0]
        while i < len(ranked) and ranked[i][0] == score:
            if ranked[i][1]: tp += 1
            else: fp += 1
            i += 1
        fpr, tpr, prec = fp/n, tp/p, tp/(tp+fp)
        auc += (fpr - roc[-1]['fpr'])*(tpr + roc[-1]['tpr'])/2
        ap += (tpr - pr[-1]['recall'])*prec
        roc.append({'fpr': fpr, 'tpr': tpr, 'threshold': score})
        pr.append({'recall': tpr, 'precision': prec, 'threshold': score})
    calibration = [{'count': 0, 'prob_sum': 0.0, 'positive': 0} for _ in range(bins)]
    for score, pos in ranked:
        b = calibration[min(bins-1, int(score*bins))]
        b['count'] += 1; b['prob_sum'] += score; b['positive'] += int(pos)
    cal = [{'lower': i/bins, 'upper': (i+1)/bins, 'count': b['count'],
            'mean_probability': b['prob_sum']/b['count'] if b['count'] else None,
            'positive_fraction': b['positive']/b['count'] if b['count'] else None} for i,b in enumerate(calibration)]
    brier = sum((score - int(pos))**2 for score, pos in ranked)/len(ranked)
    ece = sum(b['count']/len(ranked)*abs(b['mean_probability']-b['positive_fraction']) for b in cal if b['count'])
    return {'samples': len(ranked), 'positive_label': positive_label, 'negative_label': negative, 'positive_prevalence': p/len(ranked),
            'threshold': threshold, 'roc_auc': auc, 'average_precision': ap, 'brier_score': brier, 'ece': ece,
            'threshold_metrics': metrics, 'roc_curve': roc, 'precision_recall_curve': pr, 'calibration': cal,
            'limitations': 'Scores must be positive-class probabilities. Do not select thresholds on the final test set. ECE depends on binning.'}


def compare_runs(directory, *, metric='val_loss', mode='min', train_metric=None, epoch_column='epoch', max_runs=100):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError('Expected a directory containing run CSV files')
    if not 1 <= max_runs <= 1000:
        raise ValueError('Run limit must be 1–1000')
    files = sorted(p for p in directory.glob('*.csv') if p.is_file() and not p.is_symlink() and not p.name.startswith('.'))
    if not files:
        raise ValueError('No CSV files found in the directory')
    results, rejected = [], []
    for p in files[:max_runs]:
        try:
            report = analyze_history(p, metric=metric, mode=mode, train_metric=train_metric, epoch_column=epoch_column)
        except (ValueError, OSError, UnicodeError) as exc:
            rejected.append({'run': p.name, 'error': str(exc).replace(str(p), p.name)})
        else:
            results.append({'run': p.name, **report})
    if not results:
        raise ValueError('No valid runs: ' + '; '.join(r['run']+': '+r['error'] for r in rejected[:3]))
    results.sort(key=lambda r: ((r['best_value'] if mode == 'min' else -r['best_value']), r['run']))
    return {'metric': metric, 'mode': mode, 'files_found': len(files), 'scan_truncated': len(files)>max_runs,
            'valid_runs': len(results), 'rejected_runs': len(rejected), 'best_run': results[0]['run'], 'ranking': results,
            'rejected': rejected, 'limitations': 'Rankings compare reported best validation metrics only; comparable splits, metrics and budgets are required.'}


def reproducibility_manifest(path=None, *, include_runtime=True):
    """Fingerprint one explicit input file and report selected dependency versions."""
    packages = {}
    for name in ('numpy', 'pandas', 'torch', 'torchvision', 'pillow', 'scikit-learn', 'ultralytics'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report = {'schema_version': 1, 'deepforge_version': __version__, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'python': platform.python_version(), 'packages': packages, 'file': None}
    if path:
        p = Path(path)
        if not p.is_file() or p.is_symlink():
            raise ValueError('Expected a regular input file')
        report['file'] = {'name': p.name, 'size_bytes': p.stat().st_size, 'sha256': file_digest(p)}
    if include_runtime:
        report['runtime'] = inspect_environment()
    report['limitations'] = 'No environment variables, secrets, or full machine identifier collected. A manifest alone does not guarantee deterministic reproduction.'
    return report
