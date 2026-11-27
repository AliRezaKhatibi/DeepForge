"""Synthetic example of explicit pre-update health checks."""
import torch
from deepforge.training import TrainingGuard, model_check
from deepforge.automation import run_checks

def main():
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    run_checks([model_check(model, max_parameters=100)], raise_on_failure=True)
    guard = TrainingGuard(model, max_gradient_norm=100)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.ones(2, 4)).square().mean()
        loss.backward()
        report = guard.before_step(loss)
        optimizer.step()
        print(step, report["ok"])

if __name__ == "__main__":
    main()
