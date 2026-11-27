"""Explicit forward-pass profiling of trusted, already constructed PyTorch models.

MAC estimates cover executed standard Linear and Conv1d/2d/3d leaves only.
No checkpoint loading, source evaluation, or external profiling dependency.
"""
import math
import statistics
import time
from collections import Counter
from .models import inspect_model


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ValueError("Install the optional torch dependency to profile a model") from exc
    return torch


def _tensors(value, torch):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item, torch)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item, torch)


def _spec(value, torch):
    """Describe tensor metadata only, without serializing input/output values."""
    return [{"shape": list(t.shape), "dtype": str(t.dtype),
             "device": str(t.device), "bytes": t.numel()*t.element_size()}
            for t in _tensors(value, torch)]


def _direct_parameters(module):
    return sum(p.numel() for p in module.parameters(recurse=False))


def _macs(module, output, torch):
    if not isinstance(output, torch.Tensor):
        return None
    if type(module) is torch.nn.Linear:
        return output.numel()*module.in_features
    if type(module) in (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d):
        return output.numel()*(module.in_channels//module.groups)*math.prod(module.kernel_size)
    return None


def profile_model(model, *args, kwargs=None, warmup=2, repeats=5, max_rows=10000):
    """Profile an explicit example forward, then time complete uninstrumented forwards.

    The model is temporarily put in evaluation/inference mode; registered hooks
    are removed and each original training flag restored even on failure.
    Arbitrary custom forwards may have side effects; use trusted code only.
    Measured latency is for the entire input batch. Only CPU and CUDA timing
    are supported; synchronization is included. Set repeats=0 to omit timing.
    """
    torch = _torch()
    if type(warmup) is not int or not 0 <= warmup <= 1000:
        raise ValueError("warmup must be an integer in [0, 1000]")
    if type(repeats) is not int or not 0 <= repeats <= 1000:
        raise ValueError("repeats must be an integer in [0, 1000]")
    if type(max_rows) is not int or not 1 <= max_rows <= 100000:
        raise ValueError("max_rows must be an integer in [1, 100000]")
    kwargs = dict(kwargs or {})
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    registered = list(model.named_modules())
    leaves = [(name, module) for name, module in registered if not list(module.children())]
    flags = [(module, module.training) for _, module in registered]
    inventory = inspect_model(model)
    modules = [{"name": name or "<root>", "type": type(module).__name__,
                "direct_parameters": _direct_parameters(module),
                "leaf": not list(module.children())}
               for name, module in registered]
    tensors = list(model.parameters()) + list(model.buffers()) + list(_tensors((args, kwargs), torch))
    devices = {t.device for t in tensors}
    cuda_devices = sorted((d for d in devices if d.type == "cuda"), key=str)
    if repeats and any(d.type not in ("cpu", "cuda") for d in devices):
        raise ValueError("Latency measurement supports CPU and CUDA only")

    def sync():
        for device in cuda_devices:
            torch.cuda.synchronize(device)

    def memory():
        return [{"device": str(d),
                 "allocated_bytes": torch.cuda.memory_allocated(d),
                 "reserved_bytes": torch.cuda.memory_reserved(d)}
                for d in cuda_devices]

    calls, unsupported = [], Counter()
    handles = []
    total_macs = total_bytes = count = counted = 0

    def hook(name):
        def observe(module, inputs, kw, output):
            nonlocal total_macs, total_bytes, count, counted
            count += 1
            out = _spec(output, torch)
            size = sum(t["bytes"] for t in out)
            macs = _macs(module, output, torch)
            total_bytes += size
            if macs is None:
                unsupported[type(module).__name__] += 1
            else:
                total_macs += macs
                counted += 1
            if len(calls) < max_rows:
                calls.append({"call": count, "name": name or "<root>",
                              "type": type(module).__name__,
                              "inputs": _spec((inputs, kw), torch),
                              "outputs": out, "output_logical_bytes": size,
                              "estimated_macs": macs,
                              "estimated_flops_2x_macs": None if macs is None else 2*macs})
        return observe

    before = memory()
    timings = []
    try:
        model.eval()
        with torch.inference_mode():
            for name, module in leaves:
                handles.append(module.register_forward_hook(hook(name), with_kwargs=True))
            try:
                output = model(*args, **kwargs)
                output_spec = _spec(output, torch)
                del output
            finally:
                for handle in handles:
                    handle.remove()
                handles.clear()
            if repeats:
                for _ in range(warmup):
                    model(*args, **kwargs)
                sync()
                for _ in range(repeats):
                    sync()
                    start = time.perf_counter()
                    model(*args, **kwargs)
                    sync()
                    timings.append((time.perf_counter()-start)*1000)
        after = memory()
    finally:
        for handle in handles:
            handle.remove()
        for module, flag in flags:
            module.training = flag

    timing = None
    if timings:
        ordered = sorted(timings)
        timing = {"repeats": repeats, "warmup": warmup,
                  "min_ms": ordered[0], "median_ms": statistics.median(ordered),
                  "mean_ms": statistics.mean(ordered),
                  "p95_ms": ordered[math.ceil(.95*len(ordered))-1]}
    return {**inventory, "registered_modules": len(registered),
            "registered_leaf_modules": len(leaves), "executed_leaf_calls": count,
            "estimated_leaf_calls": counted, "modules": modules, "calls": calls,
            "calls_truncated": count > len(calls),
            "estimated_macs": total_macs, "estimated_flops_2x_macs": 2*total_macs,
            "unestimated_leaf_types": dict(unsupported),
            "input_tensors": _spec((args, kwargs), torch), "output_tensors": output_spec,
            "leaf_output_logical_bytes_sum": total_bytes,
            "devices": sorted(str(d) for d in devices),
            "latency": timing, "cuda_allocator_before": before,
            "cuda_allocator_after": after,
            "limitations": [
                "MACs count standard Linear and Conv1d/2d/3d leaf calls only; FLOPs = 2*MACs, excluding bias, activation, normalization, pooling, attention and functional operations.",
                "Counts are partial, not total network FLOPs; one execution path and one input batch only. Registered modules and executed calls are different counts.",
                "Logical tensor bytes can count shared storage and views multiple times; not peak activation or training memory.",
                "CUDA allocator observations are process-wide, not model-specific or peak GPU memory; timing depends on device load and synchronization.",
                "Trusted forwards can mutate custom state or consume RNG even in eval/inference mode. Avoid concurrent training and profiling of the same model."
            ]}


def profile_mlp(inputs, hidden, outputs, *, batch=1, warmup=2, repeats=5):
    """Bounded built-in CPU example for CLI and desktop usage."""
    sizes = (inputs, hidden, outputs)
    if any(type(n) is not int or not 1 <= n <= 4096 for n in sizes):
        raise ValueError("Layer widths must be integers between 1 and 4096")
    if inputs*hidden + hidden*outputs > 10_000_000:
        raise ValueError("The example is limited to 10 million weights")
    if type(batch) is not int or not 1 <= batch <= 256:
        raise ValueError("batch must be an integer between 1 and 256")
    torch = _torch()
    model = torch.nn.Sequential(torch.nn.Linear(inputs, hidden),
                                torch.nn.ReLU(), torch.nn.Linear(hidden, outputs))
    report = profile_model(model, torch.zeros(batch, inputs), warmup=warmup, repeats=repeats)
    report["architecture"] = list(sizes)
    return report
