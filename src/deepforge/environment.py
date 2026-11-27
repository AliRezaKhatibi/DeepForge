"""Report runtime details, without collecting user names or machine identifiers."""
import platform
import sys


def inspect_environment():
    result = {"python": platform.python_version(), "system": platform.system(),
              "architecture": platform.machine(), "implementation": sys.implementation.name}
    try:
        import torch
    except ImportError:
        result["pytorch"] = {"available": False}
        return result
    except Exception as exc:
        result["pytorch"] = {"available": False, "error_type": type(exc).__name__}
        return result
    info = {"available": True, "version": str(torch.__version__), "cuda_build": torch.version.cuda}
    try:
        info["cuda_available"] = torch.cuda.is_available()
        info["devices"] = [{"name": torch.cuda.get_device_name(i),
                            "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                           for i in range(torch.cuda.device_count())]
    except Exception as exc:
        info["device_query_error"] = type(exc).__name__
    result["pytorch"] = info
    return result
