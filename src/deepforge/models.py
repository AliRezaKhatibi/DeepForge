"""Inspect a caller-provided PyTorch model; never deserialize checkpoints."""


def inspect_model(model):
    """Count unique registered parameters and buffers, not peak GPU memory or FLOPs."""
    params = list(model.parameters())
    buffers = list(model.buffers())
    def unique(items):
        return list({id(x): x for x in items}.values())
    params, buffers = unique(params), unique(buffers)
    def count(items):
        return sum(p.numel() for p in items)
    def size(items):
        return sum(p.numel() * p.element_size() for p in items)
    total = count(params)
    trainable = count([p for p in params if p.requires_grad])
    return {"parameters": total, "trainable_parameters": trainable,
            "frozen_parameters": total - trainable, "buffer_elements": count(buffers),
            "parameter_bytes": size(params), "buffer_bytes": size(buffers)}
