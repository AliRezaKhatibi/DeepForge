"""Read-only dataset integrity and group leakage audits."""
import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from .datasets import EXTENSIONS, SPLITS


def image_paths(root, *, classification=False):
    """Yield (split, path) from split folders, without following directory symlinks."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Not a dataset directory: {root}")
    splits = [s for s in SPLITS if (root / s).is_dir() and not (root / s).is_symlink()]
    if not splits or ("val" in splits and "validation" in splits):
        raise ValueError("Expected split directories; use either val or validation, not both")
    for split in splits:
        for folder, dirs, files in os.walk(root / split, followlinks=False):
            folder = Path(folder)
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and not (folder / d).is_symlink())
            if classification and folder == root / split:
                continue
            for name in sorted(files):
                p = folder / name
                if not name.startswith('.') and p.suffix.lower() in EXTENSIONS and not p.is_symlink() and p.is_file():
                    yield split, p


def audit_images(root, *, similarity=False, threshold=4, max_images=2000, max_pairs=2000):
    """Decode images with Pillow and optionally compare 64-bit difference hashes.

    The bounded exhaustive search compares only scanned, decodable cross-split images.
    A zero Hamming distance is not proof of identity; flat or similar images may collide.
    """
    if not 0 <= threshold <= 64 or not 1 <= max_images <= 5000 or not 1 <= max_pairs <= 10000:
        raise ValueError("Use a threshold of 0–64, image limit of 1–5000 and pair limit of 1–10000")
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ValueError("Image audits require Pillow: pip install '.[images]'") from exc
    root = Path(root)
    entries, errors, unsupported, modes, by_split = [], [], [], Counter(), Counter()
    widths, heights = [], []
    scanned = 0
    truncated = False
    for split, path in image_paths(root):
        if scanned >= max_images:
            truncated = True
            break
        scanned += 1
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == '.dcm':
            unsupported.append(relative)
            continue
        try:
            with Image.open(path) as im:
                if im.width * im.height > 40_000_000:
                    raise ValueError("Image exceeds the 40 megapixel safety limit")
                im.load()
                im = ImageOps.exif_transpose(im)
                w, h = im.size
                if similarity:
                    small = im.convert('L').resize((9, 8), Image.Resampling.LANCZOS)
                    sample = list(small.get_flattened_data() if hasattr(small, 'get_flattened_data') else small.getdata())
                    dh = 0
                    for row in range(8):
                        for col in range(8):
                            dh = (dh << 1) | int(sample[row * 9 + col] > sample[row * 9 + col + 1])
                    entries.append((split, relative, dh))
                widths.append(w); heights.append(h); modes[im.mode] += 1; by_split[split] += 1
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            errors.append({"file": relative, "error": type(exc).__name__, "detail": str(exc).replace(str(path), relative)})
    pairs, comparisons, matched = [], 0, 0
    if similarity:
        for i, (sa, pa, ha) in enumerate(entries):
            for j in range(i + 1, len(entries)):
                sb, pb, hb = entries[j]
                if sa == sb:
                    continue
                comparisons += 1
                distance = (ha ^ hb).bit_count()
                if distance <= threshold:
                    matched += 1
                    if len(pairs) < max_pairs:
                        pairs.append({"a": pa, "b": pb, "distance": distance})
        pairs.sort(key=lambda x: (x['distance'], x['a'], x['b']))
    def stats(values):
        return {"min": min(values), "mean": sum(values)/len(values), "max": max(values)} if values else None
    return {"scanned": scanned, "decoded": len(widths), "invalid_images": len(errors), "unsupported_images": len(unsupported),
            "scan_truncated": truncated, "image_limit": max_images, "dimensions": {"width": stats(widths), "height": stats(heights)},
            "modes": dict(modes), "decoded_by_split": dict(by_split), "errors": errors, "unsupported": unsupported,
            "similarity_enabled": similarity, "hash_algorithm": "dHash-64" if similarity else None, "hamming_threshold": threshold if similarity else None,
            "cross_split_comparisons": comparisons, "candidate_pair_count": matched, "candidate_pairs": pairs,
            "pair_list_truncated": matched > len(pairs), "limitations": "Candidate similarities require human review. Unscanned, unsupported and failed images are not compared; patient independence is not established."}


def audit_groups(path, *, group_column="group_id", split_column="split", redact=True):
    """Find group IDs assigned to multiple splits; redact raw identifiers by default."""
    groups = defaultdict(set)
    split_counts = Counter()
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {group_column, split_column}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain group and split columns")
        for line, row in enumerate(reader, 2):
            group, split = ((row.get(k) or '').strip() for k in (group_column, split_column))
            split = 'val' if split.lower() == 'validation' else split.lower()
            if not group or split not in {'train', 'val', 'test'}:
                raise ValueError(f"Missing group or invalid split at line {line}")
            groups[group].add(split); split_counts[split] += 1
    if not groups:
        raise ValueError("Group manifest is empty")
    overlaps = [{"group": f"group-{i:06d}" if redact else group, "splits": sorted(splits)}
                for i, (group, splits) in enumerate(sorted(groups.items()), 1) if len(splits) > 1]
    return {"samples": sum(split_counts.values()), "groups": len(groups), "split_counts": dict(split_counts),
            "overlapping_groups": len(overlaps), "overlaps": overlaps, "identifiers_redacted": redact,
            "limitations": "Checks only supplied IDs. Correct identifiers and complete manifests are required; other leakage can remain."}


def audit_yolo(root, *, classes=None, tolerance=1e-6):
    """Audit YOLO detection labels in images/<split> and labels/<split> layout."""
    root = Path(root)
    if classes is not None and classes < 1:
        raise ValueError("Class count must be positive")
    if not math.isfinite(tolerance) or tolerance < 0 or tolerance > .01:
        raise ValueError("Tolerance must be finite and between 0 and 0.01")
    image_root, label_root = root / 'images', root / 'labels'
    if image_root.is_symlink() or label_root.is_symlink() or not label_root.is_dir():
        raise ValueError("Expected real images/ and labels/ directories")
    expected, collisions, orphan_labels, missing, errors, counts, split_boxes = {}, [], [], [], [], Counter(), Counter()
    total_images = 0
    for split, path in image_paths(image_root):
        total_images += 1
        key = path.relative_to(image_root).with_suffix('.txt').as_posix()
        if key in expected:
            collisions.append(key)
        else:
            expected[key] = path
    existing = {}
    for folder, dirs, files in os.walk(label_root, followlinks=False):
        folder = Path(folder)
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and not (folder / d).is_symlink())
        for name in sorted(files):
            p = folder / name
            if not name.startswith('.') and p.suffix == '.txt' and not p.is_symlink() and p.is_file():
                key = p.relative_to(label_root).as_posix()
                if key.split('/')[0] in SPLITS:
                    existing[key] = p
    missing = sorted(set(expected) - set(existing))
    orphan_labels = sorted(set(existing) - set(expected))
    empty = valid_boxes = invalid_rows = 0
    for key in sorted(set(expected) & set(existing)):
        p = existing[key]
        try:
            with p.open(encoding='utf-8-sig') as stream:
                seen = set(); has_rows = False
                for n, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    has_rows = True
                    try:
                        tokens = line.split()
                        if len(tokens) != 5:
                            raise ValueError('Expected class x_center y_center width height')
                        cls = int(tokens[0]); x, y, w, h = map(float, tokens[1:])
                        if cls < 0 or (classes is not None and cls >= classes):
                            raise ValueError('Class ID outside configured range')
                        if not all(math.isfinite(v) for v in (x,y,w,h)):
                            raise ValueError('Nonfinite coordinates')
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            raise ValueError('Coordinates must be normalized and sizes positive')
                        if min(x-w/2, y-h/2) < -tolerance or max(x+w/2, y+h/2) > 1+tolerance:
                            raise ValueError('Box extends outside image')
                        box = (cls,x,y,w,h)
                        if box in seen:
                            raise ValueError('Duplicate box row')
                        seen.add(box)
                    except ValueError as exc:
                        invalid_rows += 1
                        errors.append({"file": key, "line": n, "error": str(exc)})
                    else:
                        valid_boxes += 1; counts[str(cls)] += 1; split_boxes[key.split('/')[0]] += 1
                empty += not has_rows
        except (UnicodeError, OSError) as exc:
            errors.append({"file": key, "line": None, "error": f"Unreadable label: {type(exc).__name__}"})
    return {"images": total_images, "matched_label_files": len(set(expected)&set(existing)), "missing_labels": missing,
            "orphan_labels": orphan_labels, "ambiguous_image_stems": collisions, "empty_label_files": empty,
            "valid_boxes": valid_boxes, "invalid_rows": invalid_rows, "issues": errors, "class_counts": dict(counts),
            "boxes_by_split": dict(split_boxes), "class_count_configured": classes,
            "limitations": "Detection boxes only; empty labels may represent background. Does not verify image content, semantics or segmentation polygons."}
