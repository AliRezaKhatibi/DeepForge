"""Example: audit a dataset before training. No training is performed."""
import argparse
from deepforge.automation import dataset_check, group_check, run_checks, save_report, require_pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--groups")
    parser.add_argument("--report", default="quality-report.json")
    args = parser.parse_args()
    checks = [dataset_check(args.dataset)]
    if args.groups:
        checks.append(group_check(args.groups))
    result = run_checks(checks)
    save_report(result, args.report)
    require_pass(result)
    print("Configured quality gates passed.")
