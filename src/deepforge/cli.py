"""Command line interface; JSON reports are emitted to standard output."""
import argparse
import json
import sys
from .datasets import inspect_dataset
from .environment import inspect_environment
from .history import analyze_history
from .metrics import evaluate_csv
from . import __version__
from .audit import audit_images, audit_groups, audit_yolo
from .advanced import read_binary_scores, compare_runs, reproducibility_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="deepforge")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("dataset", help="Inspect image-classification directory layout")
    data.add_argument("root")
    leak = commands.add_parser("leakage", help="Find byte-identical files across splits")
    leak.add_argument("root")
    leak.add_argument("--fail-on-leakage", action="store_true")
    metrics = commands.add_parser("metrics", help="Evaluate single-label predictions from CSV")
    metrics.add_argument("csv")
    metrics.add_argument("--true-column", default="y_true")
    metrics.add_argument("--pred-column", default="y_pred")
    history = commands.add_parser("history", help="Analyze a training-history CSV")
    history.add_argument("csv")
    history.add_argument("--metric", default="val_loss")
    history.add_argument("--mode", choices=["min", "max"], default="min")
    history.add_argument("--train-metric")
    history.add_argument("--epoch-column", default="epoch")
    commands.add_parser("env", help="Inspect Python and optional PyTorch runtime")
    images = commands.add_parser("images", help="Decode images and optionally search for cross-split similarities")
    images.add_argument("root")
    images.add_argument("--similarity", action="store_true")
    images.add_argument("--threshold", type=int, default=4)
    images.add_argument("--max-images", type=int, default=2000)
    groups = commands.add_parser("groups", help="Audit group IDs across splits")
    groups.add_argument("csv")
    groups.add_argument("--group-column", default="group_id")
    groups.add_argument("--split-column", default="split")
    groups.add_argument("--show-identifiers", action="store_true")
    groups.add_argument("--fail-on-leakage", action="store_true")
    yolo = commands.add_parser("yolo", help="Audit YOLO detection labels")
    yolo.add_argument("root")
    yolo.add_argument("--classes", type=int)
    scores = commands.add_parser("scores", help="Evaluate binary probability scores")
    scores.add_argument("csv")
    scores.add_argument("--true-column", default="y_true")
    scores.add_argument("--score-column", default="score")
    scores.add_argument("--positive-label", default="1")
    scores.add_argument("--threshold", type=float, default=.5)
    scores.add_argument("--bins", type=int, default=10)
    compare = commands.add_parser("compare", help="Rank comparable training histories")
    compare.add_argument("directory")
    compare.add_argument("--metric", default="val_loss")
    compare.add_argument("--mode", choices=["min", "max"], default="min")
    compare.add_argument("--train-metric")
    compare.add_argument("--epoch-column", default="epoch")
    compare.add_argument("--max-runs", type=int, default=100)
    manifest = commands.add_parser("manifest", help="Capture versions and optional file checksum")
    manifest.add_argument("file", nargs="?")
    profile = commands.add_parser("profile-mlp", help="Profile a bounded built-in CPU MLP")
    profile.add_argument("--inputs", type=int, default=16)
    profile.add_argument("--hidden", type=int, default=32)
    profile.add_argument("--outputs", type=int, default=3)
    profile.add_argument("--batch", type=int, default=1)
    profile.add_argument("--warmup", type=int, default=2)
    profile.add_argument("--repeats", type=int, default=5)
    checkpoint = commands.add_parser("checkpoint", help="Inspect verified checkpoints")
    checkpoint.add_argument("root")
    checkpoint.add_argument("run_id")
    args = parser.parse_args(argv)
    try:
        if args.command in {"dataset", "leakage"}:
            report = inspect_dataset(args.root, hash_files=args.command == "leakage")
        elif args.command == "metrics":
            report = evaluate_csv(args.csv, args.true_column, args.pred_column)
        elif args.command == "history":
            report = analyze_history(args.csv, metric=args.metric, mode=args.mode,
                                     train_metric=args.train_metric, epoch_column=args.epoch_column)
        elif args.command == "images":
            report = audit_images(args.root, similarity=args.similarity, threshold=args.threshold, max_images=args.max_images)
        elif args.command == "groups":
            report = audit_groups(args.csv, group_column=args.group_column, split_column=args.split_column, redact=not args.show_identifiers)
        elif args.command == "yolo":
            report = audit_yolo(args.root, classes=args.classes)
        elif args.command == "scores":
            report = read_binary_scores(args.csv, true_column=args.true_column, score_column=args.score_column,
                                        positive_label=args.positive_label, threshold=args.threshold, bins=args.bins)
        elif args.command == "compare":
            report = compare_runs(args.directory, metric=args.metric, mode=args.mode, train_metric=args.train_metric,
                                  epoch_column=args.epoch_column, max_runs=args.max_runs)
        elif args.command == "profile-mlp":
            from .profiler import profile_mlp
            report = profile_mlp(args.inputs, args.hidden, args.outputs, batch=args.batch,
                                 warmup=args.warmup, repeats=args.repeats)
        elif args.command == "checkpoint":
            from .checkpoint import CheckpointStore
            store = CheckpointStore(args.root, args.run_id)
            report = {"run_id": args.run_id, "verified": [item[3] for item in store._records()]}
        elif args.command == "manifest":
            report = reproducibility_manifest(args.file)
        else:
            report = inspect_environment()
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"deepforge: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.command == "leakage" and args.fail_on_leakage and report["cross_split_duplicate_groups"]:
        return 1
    if args.command == "groups" and args.fail_on_leakage and report["overlapping_groups"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
