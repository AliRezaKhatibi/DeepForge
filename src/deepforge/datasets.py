"""Inspect image-classification directory layouts without modifying files."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

SPLITS = ("train", "val", "validation", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".dcm"}


def file_digest(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_dataset(root, *, hash_files=False):
    """Inspect root/<split>/<class>/...; optionally detect exact cross-split duplicates."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Not a dataset directory: {root}")
    splits = [s for s in SPLITS if (root / s).is_dir() and not (root / s).is_symlink()]
    if not splits:
        raise ValueError("Expected train, val, validation, or test directories")
    if "val" in splits and "validation" in splits:
        raise ValueError("Use either val or validation, not both")
    counts = {}
    hashes = defaultdict(list)
    for split in splits:
        classes = Counter()
        for folder in sorted((root / split).iterdir()):
            if not folder.is_dir() or folder.is_symlink() or folder.name.startswith("."):
                continue
            classes[folder.name] = 0
            for path in sorted(folder.rglob("*")):
                if path.is_symlink() or not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    continue
                relative = path.relative_to(root)
                if any(part.startswith(".") for part in relative.parts):
                    continue
                if any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents):
                    continue
                classes[folder.name] += 1
                if hash_files:
                    hashes[file_digest(path)].append((split, relative.as_posix()))
        total = sum(classes.values())
        counts[split] = {
            "total": total,
            "classes": dict(classes),
            "class_fractions": {k: v / total if total else 0.0 for k, v in classes.items()},
        }
    result = {"total": sum(v["total"] for v in counts.values()), "splits": counts}
    if hash_files:
        duplicates = []
        for digest, entries in sorted(hashes.items()):
            found_splits = sorted({split for split, _ in entries})
            if len(found_splits) > 1:
                duplicates.append({"sha256": digest, "splits": found_splits, "files": sorted(p for _, p in entries)})
        result.update({"cross_split_duplicate_groups": len(duplicates), "duplicates": duplicates})
    return result
