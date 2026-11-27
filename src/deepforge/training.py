"""Opt-in PyTorch training health checks; never modify model or optimizer state."""
import math
from .automation import Check, AuditFailed, run_checks
from .models import inspect_model


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the optional torch dependency") from exc
    return torch


def _positive_limit(value, name):
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")


def model_check(model, *, max_parameters=None):
    """Check initialized model size using the existing model inspector."""
    if max_parameters is not None and (type(max_parameters) is not int or max_parameters < 1):
        raise ValueError("max_parameters must be a positive integer")
    rules = [("parameters", "gt", 0)]
    if max_parameters is not None:
        rules.append(("parameters", "le", max_parameters))
    return Check("model", lambda: inspect_model(model), tuple(rules))


def loss_check(loss):
    """Check a scalar loss without retaining its computation graph."""
    def run():
        value = float(loss.detach().item() if hasattr(loss, "detach") else loss)
        return {"finite": math.isfinite(value),
                "loss": value if math.isfinite(value) else None}
    return Check("loss", run, (("finite", "eq", True),))


def gradient_report(model):
    """Inspect registered trainable gradients after backward and AMP unscale.

    Sparse COO gradients are coalesced without modifying the original.
    Other sparse layouts are explicitly unsupported. Norms are global L2
    norms; missing gradients are reported, not assumed to be errors.
    """
    torch = _torch()
    missing, nonfinite, unsupported = [], [], []
    present = 0
    norm = 0.0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            grad = parameter.grad
            if grad is None:
                missing.append(name)
                continue
            present += 1
            grad = grad.detach()
            if grad.layout == torch.sparse_coo:
                grad = grad.coalesce().values()
            elif grad.layout != torch.strided:
                unsupported.append(name)
                continue
            if not bool(torch.isfinite(grad).all().item()):
                nonfinite.append(name)
                continue
            dtype = torch.complex128 if grad.is_complex() else torch.float64
            value = float(torch.linalg.vector_norm(grad.to(dtype=dtype)).item())
            norm = math.hypot(norm, value)
    valid = not nonfinite and not unsupported and math.isfinite(norm)
    return {"present": present, "missing": missing, "nonfinite": nonfinite,
            "unsupported": unsupported, "finite": valid,
            "global_l2": norm if valid else None}


def gradient_check(model, *, max_norm=None, require_all=False):
    _positive_limit(max_norm, "max_norm")
    rules = [("finite", "eq", True), ("present", "gt", 0)]
    if require_all:
        rules.append(("missing", "eq", []))
    if max_norm is not None:
        rules.append(("global_l2", "le", max_norm))
    def run():
        report = gradient_report(model)
        # Keep nonfinite reports serializable and allow rules to fail normally.
        report["within_norm_limit"] = (report["global_l2"] is not None and
                                      (max_norm is None or report["global_l2"] <= max_norm))
        return report
    if max_norm is not None:
        rules[-1] = ("within_norm_limit", "eq", True)
    return Check("gradients", run, tuple(rules))


class TrainingGuard:
    """Explicit pre-step guard. Call after backward, before optimizer.step.

    No hooks, background workers, optimizer changes, or automatic clipping.
    Checks synchronize devices and can slow training. With AMP call after
    scaler.unscale_(optimizer); with accumulation call at update boundaries.
    """
    def __init__(self, model, *, max_gradient_norm=None, require_all_gradients=False):
        _positive_limit(max_gradient_norm, "max_gradient_norm")
        self.model = model
        self.max_gradient_norm = max_gradient_norm
        self.require_all_gradients = require_all_gradients

    def before_step(self, loss, *, raise_on_failure=True):
        report = run_checks([
            loss_check(loss),
            gradient_check(self.model, max_norm=self.max_gradient_norm,
                           require_all=self.require_all_gradients),
        ])
        if raise_on_failure and not report["ok"]:
            error = AuditFailed("Training health check failed before optimizer step")
            error.report = report
            raise error
        return report
