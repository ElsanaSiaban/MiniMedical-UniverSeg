from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACDC_FOLDER = "ACDC"

LABEL_KEYWORDS = (
    "label",
    "mask",
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
            "Inspect the internal structure of a preprocessed ACDC HDF5 "
            "download before writing its preprocessing adapter. "
            "This script is read-only."
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
        help="Folder inside MiniMedical. Default: ACDC",
    )
    parser.add_argument(
        "--files-per-folder",
        type=int,
        default=3,
        help="Representative HDF5 files inspected per top-level folder.",
    )
    parser.add_argument(
        "--max-unique",
        type=int,
        default=30,
        help="Maximum unique values recorded for likely label datasets.",
    )
    return parser.parse_args()


def infer_patient_and_frame(path: Path) -> dict[str, str | None]:
    name = path.stem

    patient_match = re.search(r"(?i)(patient\d+)", name)
    frame_match = re.search(r"(?i)(frame\d+)", name)
    slice_match = re.search(r"(?i)(slice\d+|slice[_-]?\d+)", name)

    return {
        "patient_id": (
            patient_match.group(1).lower() if patient_match else None
        ),
        "frame_id": (
            frame_match.group(1).lower() if frame_match else None
        ),
        "slice_id": (
            slice_match.group(1).lower() if slice_match else None
        ),
    }


def likely_label_dataset(dataset_path: str, array: np.ndarray) -> bool:
    normalized = dataset_path.lower()

    if any(keyword in normalized for keyword in LABEL_KEYWORDS):
        return True

    if np.issubdtype(array.dtype, np.integer):
        # Small discrete integer ranges are often segmentation labels.
        if array.size > 0:
            minimum = int(np.min(array))
            maximum = int(np.max(array))
            return minimum >= 0 and maximum <= 20

    return False


def summarize_array(
    array: np.ndarray,
    dataset_path: str,
    max_unique: int,
) -> dict[str, Any]:
    array = np.asarray(array)

    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
        "ndim": int(array.ndim),
    }

    if array.size == 0:
        summary["error"] = "empty dataset"
        return summary

    finite = np.isfinite(array)
    summary["finite_fraction"] = float(finite.mean())

    if finite.any():
        values = array[finite]
        summary["minimum"] = float(values.min())
        summary["maximum"] = float(values.max())
        summary["mean"] = float(values.mean())
        summary["standard_deviation"] = float(values.std())

        if likely_label_dataset(dataset_path, array):
            unique_values = np.unique(values)
            summary["likely_label"] = True
            summary["unique_count"] = int(unique_values.size)
            summary["unique_values"] = [
                float(value)
                for value in unique_values[:max_unique]
            ]
            summary["unique_values_truncated"] = (
                unique_values.size > max_unique
            )
        else:
            summary["likely_label"] = False

    return summary


def inspect_h5_file(
    path: Path,
    max_unique: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "file_size_bytes": path.stat().st_size,
        "file_attributes": {},
        "groups": [],
        "datasets": {},
    }

    with h5py.File(path, "r") as handle:
        report["file_attributes"] = {
            str(key): to_jsonable(value)
            for key, value in handle.attrs.items()
        }

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Group):
                report["groups"].append(name)
                return

            if not isinstance(obj, h5py.Dataset):
                return

            dataset_report: dict[str, Any] = {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "compression": obj.compression,
                "chunks": list(obj.chunks) if obj.chunks else None,
                "attributes": {
                    str(key): to_jsonable(value)
                    for key, value in obj.attrs.items()
                },
            }

            # Read the complete dataset only when it is reasonably sized.
            element_count = int(np.prod(obj.shape)) if obj.shape else 1
            if element_count <= 20_000_000:
                array = obj[()]
                dataset_report.update(
                    summarize_array(
                        np.asarray(array),
                        dataset_path=name,
                        max_unique=max_unique,
                    )
                )
            else:
                dataset_report["statistics_skipped"] = (
                    f"Dataset contains {element_count} elements."
                )

            report["datasets"][name] = dataset_report

        handle.visititems(visitor)

    return report


def to_jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def choose_representative_files(
    acdc_root: Path,
    files_per_folder: int,
) -> tuple[dict[str, list[Path]], list[Path]]:
    h5_files = sorted(
        path
        for path in acdc_root.rglob("*.h5")
        if path.is_file()
    )

    if not h5_files:
        raise RuntimeError(f"No .h5 files found under: {acdc_root}")

    by_top_folder: dict[str, list[Path]] = defaultdict(list)
    for path in h5_files:
        relative = path.relative_to(acdc_root)
        top_folder = relative.parts[0] if len(relative.parts) > 1 else "."
        by_top_folder[top_folder].append(path)

    selected: list[Path] = []
    for top_folder in sorted(by_top_folder):
        folder_files = by_top_folder[top_folder]

        # Select the beginning, middle, and end where possible so inspection
        # is not limited to a single patient or frame.
        indexes = [0]
        if files_per_folder >= 2 and len(folder_files) > 1:
            indexes.append(len(folder_files) // 2)
        if files_per_folder >= 3 and len(folder_files) > 2:
            indexes.append(len(folder_files) - 1)

        next_index = 1
        while len(indexes) < min(files_per_folder, len(folder_files)):
            if next_index not in indexes:
                indexes.append(next_index)
            next_index += 1

        for index in sorted(set(indexes))[:files_per_folder]:
            selected.append(folder_files[index])

    return dict(by_top_folder), selected


def compare_dataset_schemas(
    inspected_files: list[dict[str, Any]],
) -> dict[str, Any]:
    schema_counter: Counter[str] = Counter()

    for item in inspected_files:
        datasets = item["inspection"].get("datasets", {})
        schema_parts = []
        for name, metadata in sorted(datasets.items()):
            schema_parts.append(
                f"{name}:{metadata.get('shape')}:{metadata.get('dtype')}"
            )
        schema_counter[" | ".join(schema_parts)] += 1

    return {
        "unique_schema_count": len(schema_counter),
        "schemas": [
            {"schema": schema, "file_count": count}
            for schema, count in schema_counter.most_common()
        ],
    }


def main() -> None:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    acdc_root = project_root / "data" / "raw" / args.acdc_folder

    if not acdc_root.is_dir():
        raise FileNotFoundError(
            f"ACDC folder not found: {acdc_root}"
        )

    by_top_folder, selected_files = choose_representative_files(
        acdc_root,
        files_per_folder=args.files_per_folder,
    )

    inspected_files: list[dict[str, Any]] = []

    for index, path in enumerate(selected_files, start=1):
        relative_path = path.relative_to(acdc_root)
        print(
            f"[{index}/{len(selected_files)}] Inspecting {relative_path}",
            flush=True,
        )

        item = {
            "relative_path": relative_path.as_posix(),
            "top_level_folder": (
                relative_path.parts[0]
                if len(relative_path.parts) > 1
                else "."
            ),
            **infer_patient_and_frame(relative_path),
        }

        try:
            item["inspection"] = inspect_h5_file(
                path,
                max_unique=args.max_unique,
            )
        except Exception as exc:
            item["inspection"] = {
                "error": type(exc).__name__,
                "message": str(exc),
            }

        inspected_files.append(item)

    folder_summary = {
        folder: {
            "file_count": len(paths),
            "first_file": paths[0].relative_to(acdc_root).as_posix(),
            "last_file": paths[-1].relative_to(acdc_root).as_posix(),
        }
        for folder, paths in sorted(by_top_folder.items())
    }

    report = {
        "acdc_root": str(acdc_root),
        "total_h5_files": sum(len(paths) for paths in by_top_folder.values()),
        "top_level_folders": folder_summary,
        "selected_file_count": len(selected_files),
        "inspected_files": inspected_files,
        "schema_comparison": compare_dataset_schemas(inspected_files),
        "decisions_enabled_by_this_report": [
            "Identify the HDF5 image key.",
            "Identify the HDF5 segmentation-label key.",
            "Determine 2D slice versus 3D volume storage.",
            "Verify label values such as 0, 1, 2, and 3.",
            "Determine whether testing volumes contain ground truth.",
            "Determine patient, frame, and slice naming rules.",
        ],
    }

    output_path = (
        project_root
        / "outputs"
        / "acdc_h5_inspection.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== ACDC HDF5 INSPECTION SUMMARY ===")
    print(f"Root: {acdc_root}")
    print(f"Total HDF5 files: {report['total_h5_files']}")
    print("Top-level folders:")
    for folder, metadata in folder_summary.items():
        print(f"  {folder}: {metadata['file_count']} files")

    print("\nRepresentative file contents:")
    for item in inspected_files:
        print(f"\n  {item['relative_path']}")
        inspection = item["inspection"]
        if "error" in inspection:
            print(
                f"    ERROR: {inspection['error']}: "
                f"{inspection['message']}"
            )
            continue

        datasets = inspection.get("datasets", {})
        if not datasets:
            print("    No datasets found.")
            continue

        for name, metadata in datasets.items():
            print(
                f"    {name}: "
                f"shape={metadata.get('shape')}, "
                f"dtype={metadata.get('dtype')}, "
                f"min={metadata.get('minimum')}, "
                f"max={metadata.get('maximum')}, "
                f"unique={metadata.get('unique_values')}"
            )

    print(f"\nFull JSON report: {output_path}")


if __name__ == "__main__":
    main()
