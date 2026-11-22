from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACDC_FOLDER = "ACDC"

RASTER_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif",
}
ARRAY_EXTENSIONS = {".npy", ".npz"}
NIFTI_EXTENSIONS = {".nii", ".nii.gz"}

MASK_KEYWORDS = (
    "mask",
    "label",
    "seg",
    "gt",
    "groundtruth",
    "ground_truth",
    "annotation",
    "manual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an unknown or preprocessed ACDC download before writing "
            "its dataset adapter. The script does not modify any files."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Repository root. Defaults to this repository directory.",
    )
    parser.add_argument(
        "--acdc-folder",
        type=str,
        default=DEFAULT_ACDC_FOLDER,
        help="Folder name inside MiniMedical. Default: ACDC",
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=50,
        help="Number of relative paths to print. Default: 50",
    )
    parser.add_argument(
        "--max-inspect",
        type=int,
        default=30,
        help="Maximum number of data files to inspect. Default: 30",
    )
    return parser.parse_args()


def compound_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def is_mask_candidate(path: Path) -> bool:
    normalized = path.as_posix().lower()
    return any(keyword in normalized for keyword in MASK_KEYWORDS)


def infer_patient_id(path: Path) -> str | None:
    text = path.as_posix()

    patterns = [
        r"(?i)(patient[_ -]?\d+)",
        r"(?i)(subject[_ -]?\d+)",
        r"(?i)(case[_ -]?\d+)",
        r"(?i)(^|[/\\])(\d{3})(?=[/\\_. -]|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(match.lastindex or 0)
            return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()

    return None


def summarize_array(array: np.ndarray, mask_candidate: bool) -> dict[str, Any]:
    array = np.asarray(array)
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
    }

    if array.size == 0:
        result["error"] = "empty array"
        return result

    finite = np.isfinite(array)
    result["finite_fraction"] = float(finite.mean())

    if finite.any():
        finite_values = array[finite]
        result["minimum"] = float(finite_values.min())
        result["maximum"] = float(finite_values.max())
        result["mean"] = float(finite_values.mean())

        if mask_candidate or np.issubdtype(array.dtype, np.integer):
            unique_values = np.unique(finite_values)
            result["unique_count"] = int(unique_values.size)
            result["unique_values_first_30"] = [
                float(value) for value in unique_values[:30]
            ]

    return result


def inspect_raster(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        array = np.asarray(image)
        result = summarize_array(array, is_mask_candidate(path))
        result["pil_mode"] = image.mode
        result["pil_size"] = list(image.size)
        return result


def inspect_numpy(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return summarize_array(array, is_mask_candidate(path))

    archive = np.load(path, allow_pickle=False)
    result: dict[str, Any] = {"keys": list(archive.files)}
    result["arrays"] = {
        key: summarize_array(
            archive[key],
            is_mask_candidate(path) or is_mask_candidate(Path(key)),
        )
        for key in archive.files[:10]
    }
    return result


def inspect_nifti(path: Path) -> dict[str, Any]:
    try:
        import nibabel as nib
    except ImportError:
        return {
            "skipped": True,
            "reason": (
                "nibabel is not installed. Install it with: "
                "python -m pip install nibabel"
            ),
        }

    image = nib.load(str(path))
    proxy = image.dataobj
    shape = tuple(int(value) for value in proxy.shape)

    result: dict[str, Any] = {
        "shape": list(shape),
        "dtype": str(image.get_data_dtype()),
        "affine": np.asarray(image.affine).round(6).tolist(),
        "zooms": [float(value) for value in image.header.get_zooms()],
    }

    # ACDC files are small enough for limited inspection, but avoid reading
    # arbitrary huge volumes into memory.
    voxel_count = int(np.prod(shape))
    if voxel_count <= 50_000_000:
        array = np.asarray(proxy)
        result.update(summarize_array(array, is_mask_candidate(path)))
    else:
        result["array_statistics_skipped"] = (
            f"Volume contains {voxel_count} voxels."
        )

    return result


def inspect_data_file(path: Path) -> dict[str, Any]:
    suffix = compound_suffix(path)

    try:
        if suffix in RASTER_EXTENSIONS:
            return inspect_raster(path)
        if suffix in ARRAY_EXTENSIONS:
            return inspect_numpy(path)
        if suffix in NIFTI_EXTENSIONS:
            return inspect_nifti(path)
        return {"skipped": True, "reason": f"unsupported extension: {suffix}"}
    except Exception as exc:
        return {
            "error": type(exc).__name__,
            "message": str(exc),
        }


def select_files_for_inspection(
    files: list[Path],
    max_inspect: int,
) -> list[Path]:
    supported = [
        path
        for path in files
        if compound_suffix(path)
        in (RASTER_EXTENSIONS | ARRAY_EXTENSIONS | NIFTI_EXTENSIONS)
    ]

    mask_files = [path for path in supported if is_mask_candidate(path)]
    image_files = [path for path in supported if not is_mask_candidate(path)]

    selected: list[Path] = []

    # Inspect masks first because label values determine the adapter.
    selected.extend(mask_files[: max_inspect // 2])
    remaining = max_inspect - len(selected)
    selected.extend(image_files[:remaining])

    return selected


def main() -> None:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    acdc_root = project_root / "data" / "raw" / args.acdc_folder

    if not acdc_root.is_dir():
        raise FileNotFoundError(
            f"ACDC directory not found: {acdc_root}\n"
            "Pass the correct folder using --acdc-folder or --project-root."
        )

    files = sorted(
        path for path in acdc_root.rglob("*") if path.is_file()
    )

    if not files:
        raise RuntimeError(f"No files found under: {acdc_root}")

    relative_paths = [path.relative_to(acdc_root) for path in files]
    extension_counts = Counter(compound_suffix(path) for path in files)
    patient_ids = [
        patient_id
        for path in relative_paths
        if (patient_id := infer_patient_id(path)) is not None
    ]

    selected = select_files_for_inspection(files, args.max_inspect)

    inspected_files: list[dict[str, Any]] = []
    for path in selected:
        relative_path = path.relative_to(acdc_root)
        print(f"Inspecting: {relative_path}", flush=True)
        inspected_files.append(
            {
                "relative_path": relative_path.as_posix(),
                "extension": compound_suffix(path),
                "mask_candidate": is_mask_candidate(relative_path),
                "patient_id_guess": infer_patient_id(relative_path),
                "inspection": inspect_data_file(path),
            }
        )

    report = {
        "acdc_root": str(acdc_root),
        "total_files": len(files),
        "extension_counts": dict(sorted(extension_counts.items())),
        "detected_patient_id_count": len(set(patient_ids)),
        "detected_patient_ids_first_30": sorted(set(patient_ids))[:30],
        "first_relative_paths": [
            path.as_posix() for path in relative_paths[: args.max_paths]
        ],
        "mask_candidate_count": sum(
            is_mask_candidate(path) for path in relative_paths
        ),
        "inspected_files": inspected_files,
        "adapter_decisions_still_required": [
            "Confirm the exact image-mask naming relationship.",
            "Confirm whether data are 2D, 3D, or 4D.",
            "Confirm the label mapping from mask unique values.",
            "Confirm patient and cardiac-phase identifiers.",
            "Confirm which patients contain public ground truth.",
            "Choose all positive slices or representative positive slices.",
        ],
    }

    output_path = project_root / "outputs" / "acdc_inspection.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== ACDC INSPECTION SUMMARY ===")
    print(f"Root: {acdc_root}")
    print(f"Total files: {len(files)}")
    print(f"Extensions: {dict(sorted(extension_counts.items()))}")
    print(f"Detected patient IDs: {len(set(patient_ids))}")
    print(
        "Mask candidates:",
        report["mask_candidate_count"],
    )

    print("\nFirst relative paths:")
    for relative_path in report["first_relative_paths"]:
        print(f"  {relative_path}")

    print(f"\nFull JSON report: {output_path}")


if __name__ == "__main__":
    main()
