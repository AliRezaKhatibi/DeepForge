"""Optional resource sampling; no network calls or external processes."""


def sample_resources():
    result = {}
    try:
        import psutil
    except ImportError:
        pass
    else:
        result["cpu_percent"] = psutil.cpu_percent(interval=None)
        result["ram_percent"] = psutil.virtual_memory().percent
        result["process_rss_bytes"] = psutil.Process().memory_info().rss
    try:
        import torch
    except ImportError:
        pass
    else:
        if torch.cuda.is_initialized():
            device = torch.cuda.current_device()
            result["cuda_device"] = device
            result["cuda_allocated_bytes"] = torch.cuda.memory_allocated(device)
            result["cuda_reserved_bytes"] = torch.cuda.memory_reserved(device)
    return result
