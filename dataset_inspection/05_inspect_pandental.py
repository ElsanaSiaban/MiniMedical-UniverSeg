from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANDENTAL_FOLDER = "Panoramic_Dental_Xrays"

RASTER_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif",
}
DICOM_EXTENSIONS = {".dcm", ".dicom"}

MASK_KEYWORDS = (
    "mask",
    "masks",
    "seg",
    "segment",
    "segmentation",
    "label",
    "labels",
    "groundtruth",
    "ground_truth",
    "ground truth",
    "annotation",
    "annotations",
    "mandible",
    "contour",
)

IMAGE_KEYWORDS = (
    "image",
    "images",
    "original",
    "originals",
    "xray",
    "x-ray",
    "radiograph",
    "panoramic",
)

REMOVABLE_STEM_TOKENS = (
    "mask",
    "masks",
    "seg",
    "segment",
    "segmentation",
    "label",
    "labels",
    "groundtruth",
    "ground_truth",
    "annotation",
    "annotations",
    "mandible",
    "contour",
    "image",
    "images",
    "original",
    "originals",
    "xray",
    "radiograph",
    "panoramic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a downloaded Panoramic Dental X-ray dataset before "
            "writing its UniverSeg preprocessing adapter. This script is "
            "read-only and does not modify the dataset."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Repository root. Defaults to this repository directory.",
    )
    parser.add_argument(
        "--pandental-folder",
        type=str,
        default=DEFAULT_PANDENTAL_FOLDER,
        help=(
            "Folder inside MiniMedical containing the dental dataset. "
            "Default: Panoramic_Dental_Xrays"
        ),
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=100,
        help="Number of relative paths to include. Default: 100",
    )
    parser.add_argument(
        "--max-inspect",
        type=int,
        default=40,
        help="Maximum number of image-like files to inspect. Default: 40",
    )
    parser.add_argument(
        "--max-pair-examples",
        type=int,
        default=50,
        help="Maximum candidate image-mask pairs to report. Default: 50",
    )
    return parser.parse_args()


def normalized_text(path: Path) -> str:
    return path.as_posix().lower()


def is_mask_candidate(path: Path) -> bool:
    text = normalized_text(path)
    return any(keyword in text for keyword in MASK_KEYWORDS)


def is_image_candidate(path: Path) -> bool:
    text = normalized_text(path)
    return any(keyword in text for keyword in IMAGE_KEYWORDS)


def compound_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def normalized_pair_key(path: Path) -> str:
    stem = path.stem.lower()

    for token in REMOVABLE_STEM_TOKENS:
        stem = re.sub(
            rf"(?i)(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)",
            "_",
            stem,
        )

    stem = re.sub(r"[^a-z0-9]+", "", stem)
    return stem


def extract_numeric_id(path: Path) -> str | None:
    candidates = re.findall(r"\d+", path.stem)
    if not candidates:
        return None
    return candidates[-1].lstrip("0") or "0"


def summarize_array(
    array: np.ndarray,
    likely_mask: bool,
) -> dict[str, Any]:
    array = np.asarray(array)

    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "ndim": int(array.ndim),
        "size": int(array.size),
    }

    if array.size == 0:
        result["error"] = "empty array"
        return result

    finite = np.isfinite(array)
    result["finite_fraction"] = float(finite.mean())

    if finite.any():
        values = array[finite]
        result["minimum"] = float(values.min())
        result["maximum"] = float(values.max())
        result["mean"] = float(values.mean())
        result["standard_deviation"] = float(values.std())

        if likely_mask:
            unique_values = np.unique(values)
            result["unique_count"] = int(unique_values.size)
            result["unique_values_first_50"] = [
                float(value) for value in unique_values[:50]
            ]
            result["binary_like"] = bool(
                unique_values.size <= 3
                and (
                    set(float(value) for value in unique_values)
                    <= {0.0, 1.0, 255.0}
                    or unique_values.size == 2
                )
            )

    return result


def inspect_raster(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        array = np.asarray(image)
        result = summarize_array(array, is_mask_candidate(path))
        result["pil_mode"] = image.mode
        result["width"] = int(image.width)
        result["height"] = int(image.height)
        return result


def inspect_dicom(path: Path) -> dict[str, Any]:
    try:
        import pydicom
    except ImportError:
        return {
            "skipped": True,
            "reason": (
                "pydicom is not installed. Install it with: "
                "python -m pip install pydicom"
            ),
        }

    dataset = pydicom.dcmread(str(path), stop_before_pixels=False)
    array = np.asarray(dataset.pixel_array)

    result = summarize_array(array, is_mask_candidate(path))
    result["patient_id"] = str(getattr(dataset, "PatientID", ""))
    result["study_instance_uid"] = str(
        getattr(dataset, "StudyInstanceUID", "")
    )
    result["series_instance_uid"] = str(
        getattr(dataset, "SeriesInstanceUID", "")
    )
    result["rows"] = int(getattr(dataset, "Rows", array.shape[-2]))
    result["columns"] = int(getattr(dataset, "Columns", array.shape[-1]))
    return result


def inspect_file(path: Path) -> dict[str, Any]:
    suffix = compound_suffix(path)

    try:
        if suffix in RASTER_EXTENSIONS:
            return inspect_raster(path)
        if suffix in DICOM_EXTENSIONS:
            return inspect_dicom(path)
        return {
            "skipped": True,
            "reason": f"unsupported extension: {suffix}",
        }
    except Exception as exc:
        return {
            "error": type(exc).__name__,
            "message": str(exc),
        }


def choose_representatives(
    files: list[Path],
    max_inspect: int,
) -> list[Path]:
    image_like = [
        path
        for path in files
        if compound_suffix(path) in (RASTER_EXTENSIONS | DICOM_EXTENSIONS)
    ]

    masks = [path for path in image_like if is_mask_candidate(path)]
    non_masks = [path for path in image_like if not is_mask_candidate(path)]

    selected: list[Path] = []

    mask_limit = min(len(masks), max_inspect // 2)
    selected.extend(masks[:mask_limit])

    remaining = max_inspect - len(selected)
    selected.extend(non_masks[:remaining])

    return selected


def discover_candidate_pairs(
    files: list[Path],
    max_examples: int,
) -> dict[str, Any]:
    supported = [
        path
        for path in files
        if compound_suffix(path) in (RASTER_EXTENSIONS | DICOM_EXTENSIONS)
    ]

    masks = [path for path in supported if is_mask_candidate(path)]
    images = [path for path in supported if path not in masks]

    masks_by_key: dict[str, list[Path]] = defaultdict(list)
    masks_by_numeric_id: dict[str, list[Path]] = defaultdict(list)

    for mask in masks:
        key = normalized_pair_key(mask)
        if key:
            masks_by_key[key].append(mask)

        numeric_id = extract_numeric_id(mask)
        if numeric_id is not None:
            masks_by_numeric_id[numeric_id].append(mask)

    pair_examples: list[dict[str, Any]] = []
    paired_images: set[Path] = set()
    paired_masks: set[Path] = set()

    for image in images:
        candidates: list[tuple[str, Path]] = []

        key = normalized_pair_key(image)
        for mask in masks_by_key.get(key, []):
            candidates.append(("normalized_stem", mask))

        if not candidates:
            numeric_id = extract_numeric_id(image)
            if numeric_id is not None:
                for mask in masks_by_numeric_id.get(numeric_id, []):
                    candidates.append(("numeric_id", mask))

        unique_candidates = []
        seen: set[Path] = set()
        for method, mask in candidates:
            if mask not in seen:
                seen.add(mask)
                unique_candidates.append((method, mask))

        if len(unique_candidates) == 1:
            method, mask = unique_candidates[0]
            paired_images.add(image)
            paired_masks.add(mask)

            if len(pair_examples) < max_examples:
                pair_examples.append(
                    {
                        "image": image.as_posix(),
                        "mask": mask.as_posix(),
                        "method": method,
                        "image_key": normalized_pair_key(image),
                        "mask_key": normalized_pair_key(mask),
                    }
                )

    return {
        "image_candidate_count": len(images),
        "mask_candidate_count": len(masks),
        "uniquely_paired_image_count": len(paired_images),
        "uniquely_paired_mask_count": len(paired_masks),
        "unpaired_image_count": len(images) - len(paired_images),
        "unpaired_mask_count": len(masks) - len(paired_masks),
        "pair_examples": pair_examples,
        "warning": (
            "These are heuristic candidate pairs only. Do not use them as "
            "the final adapter until folder and naming semantics are reviewed."
        ),
    }


def main() -> None:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    dental_root = (
        project_root
        / "data"
        / "raw"
        / args.pandental_folder
    )

    if not dental_root.is_dir():
        raise FileNotFoundError(
            f"PanDental directory not found: {dental_root}\n"
            "Pass the actual folder name with --pandental-folder."
        )

    files = sorted(
        path for path in dental_root.rglob("*") if path.is_file()
    )
    if not files:
        raise RuntimeError(f"No files found under: {dental_root}")

    relative_paths = [path.relative_to(dental_root) for path in files]

    extension_counts = Counter(
        compound_suffix(path) for path in files
    )
    top_folder_counts: Counter[str] = Counter()
    for path in relative_paths:
        top_folder = path.parts[0] if len(path.parts) > 1 else "."
        top_folder_counts[top_folder] += 1

    selected = choose_representatives(
        files,
        max_inspect=args.max_inspect,
    )

    inspected_files: list[dict[str, Any]] = []
    for index, path in enumerate(selected, start=1):
        relative_path = path.relative_to(dental_root)
        print(
            f"[{index:02d}/{len(selected)}] Inspecting {relative_path}",
            flush=True,
        )
        inspected_files.append(
            {
                "relative_path": relative_path.as_posix(),
                "extension": compound_suffix(path),
                "mask_candidate": is_mask_candidate(relative_path),
                "image_candidate": is_image_candidate(relative_path),
                "normalized_pair_key": normalized_pair_key(relative_path),
                "numeric_id": extract_numeric_id(relative_path),
                "inspection": inspect_file(path),
            }
        )

    pairing = discover_candidate_pairs(
        relative_paths,
        max_examples=args.max_pair_examples,
    )

    dimensions = Counter()
    mask_unique_patterns = Counter()

    for item in inspected_files:
        inspection = item["inspection"]
        if "width" in inspection and "height" in inspection:
            dimensions[
                f"{inspection['width']}x{inspection['height']}"
            ] += 1

        if item["mask_candidate"]:
            unique_values = inspection.get("unique_values_first_50")
            if unique_values is not None:
                mask_unique_patterns[str(unique_values)] += 1

    report = {
        "dental_root": str(dental_root),
        "total_files": len(files),
        "extension_counts": dict(sorted(extension_counts.items())),
        "top_level_folder_counts": dict(
            sorted(top_folder_counts.items())
        ),
        "first_relative_paths": [
            path.as_posix()
            for path in relative_paths[: args.max_paths]
        ],
        "inspected_file_count": len(inspected_files),
        "inspected_files": inspected_files,
        "observed_dimensions": dict(dimensions),
        "observed_mask_unique_value_patterns": dict(
            mask_unique_patterns
        ),
        "heuristic_pairing": pairing,
        "adapter_decisions_still_required": [
            "Identify the authoritative image folder.",
            "Identify the authoritative mandible-mask folder.",
            "Confirm whether one or two dentist annotations are present.",
            "Choose the annotation to use or define a combination rule.",
            "Confirm exact image-mask pairing.",
            "Confirm patient IDs and whether every patient has one image.",
            "Confirm mask polarity and binary threshold.",
            "Decide whether a valid image-region mask is needed.",
        ],
    }

    output_path = (
        project_root
        / "outputs"
        / "pandental_inspection.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== PANDENTAL INSPECTION SUMMARY ===")
    print(f"Root: {dental_root}")
    print(f"Total files: {len(files)}")
    print(f"Extensions: {dict(sorted(extension_counts.items()))}")
    print(
        "Top-level folders:",
        dict(sorted(top_folder_counts.items())),
    )
    print(
        "Heuristic candidates:",
        {
            key: value
            for key, value in pairing.items()
            if key.endswith("_count")
        },
    )
    print(f"Full JSON report: {output_path}")


if __name__ == "__main__":
    main()
