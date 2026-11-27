"""Profile a caller-constructed model; no downloaded weights or datasets."""
import json
import torch
from deepforge.profiler import profile_model
from deepforge.reporting import export_report


def main():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        torch.nn.Linear(8, 3),
    )
    report = profile_model(model, torch.zeros(1, 3, 64, 64), warmup=2, repeats=5)
    print(json.dumps(report, indent=2))
    export_report("architecture.html", "Architecture profile", report,
                  cards=[("Parameters", report["parameters"]),
                         ("Estimated MACs", report["estimated_macs"]),
                         ("Forward median ms", report["latency"]["median_ms"])])


if __name__ == "__main__":
    main()
