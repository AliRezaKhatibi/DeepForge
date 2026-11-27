"""Optional resource sampling; no network calls or external processes."""


def sample_resources(path=None):
    result = {}
    if path is not None:
        import shutil
        disk = shutil.disk_usage(path)
        result['disk_percent'] = 100 * disk.used / disk.total
        result['disk_free_bytes'] = disk.free
    try:
        import psutil
    except ImportError:
        pass
    else:
        result["cpu_percent"] = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        result["ram_percent"] = memory.percent
        result["ram_total_bytes"] = memory.total
        result["ram_used_bytes"] = memory.used
        result["process_rss_bytes"] = psutil.Process().memory_info().rss
    try:
        import torch
    except ImportError:
        pass
    else:
        if torch.cuda.is_initialized():
            device = torch.cuda.current_device()
            result["cuda_device"] = device
            result["cuda_name"] = torch.cuda.get_device_name(device)
            free, total = torch.cuda.mem_get_info(device)
            result["cuda_total_bytes"] = total
            result["cuda_used_bytes"] = total - free
            result["cuda_memory_percent"] = 100 * (total - free) / total
            result["cuda_allocated_bytes"] = torch.cuda.memory_allocated(device)
            result["cuda_reserved_bytes"] = torch.cuda.memory_reserved(device)
            try:
                result["gpu_utilization_percent"] = float(torch.cuda.utilization(device))
            except Exception:
                pass  # Optional NVML support; unavailable is not zero.
    return result
