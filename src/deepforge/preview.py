"""Explicit opt-in local previews. Never scan or upload input images."""
import base64
import io
from pathlib import Path


def image_preview(path):
    """Return a bounded PNG preview of a caller-selected, already annotated image."""
    from PIL import Image, ImageOps
    with Image.open(Path(path)) as image:
        if image.width * image.height > 40_000_000:
            raise ValueError("Preview image exceeds 40 megapixels")
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((480, 320), Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        payload = stream.getvalue()
    if len(payload) > 600_000:
        raise ValueError("Preview exceeds size limit")
    return base64.b64encode(payload).decode("ascii")
