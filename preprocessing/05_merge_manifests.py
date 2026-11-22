from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_SIZE = 128

SOURCE_MANIFESTS = (
    "manifest_busi.csv",
    "manifest_wbc.csv",
    "manifest_acdc.csv",
    "manifest_pandental.csv",
)

REQUIRED_COLUMNS = (
    "sample_id",
    "task_id",
    "task_role",
    "dataset",
    "modality",
    "anatomy",
    "target_label",
    "image_path",
    "mask_path",
    "valid_mask_path",
    "subject_id",
    "split",
    "has_foreground",
    "source_image_path",
    "source_mask_paths",
)

EXPECTED_TASK_ROLE = {
    "busi_all_lesions": "train",
    "busi_benign_lesion": "train",
    "busi_malignant_lesion": "train",
    "wbc1_whole_cell": "train",
    "wbc1_nucleus": "train",
    "acdc_right_ventricle": "train",
    "acdc_myocardium": "train",
    "acdc_left_ventricle": "train",
    "wbc2_whole_cell": "heldout",
    "wbc2_nucleus": "heldout",
    "pandental_mandible": "heldout",
}

EXPECTED_DATASET_ROLE = {
    "busi": "train",
    "wbc_dataset1": "train",
    "acdc": "train",
    "wbc_dataset2": "heldout",
    "pandental": "heldout",
}

EXPECTED_ROW_COUNTS = {
    "manifest_busi.csv": 1693,
    "manifest_wbc.csv": 800,
    "manifest_acdc.csv": 900,
    "manifest_pandental.csv": 115,
}

ALLOWED_ROLES = {"train", "heldout"}
ALLOWED_SPLITS = {"support", "dev", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge BUSI, WBC, ACDC, and PanDental manifests into one "
            "validated quantitative dataset index. The script performs "
            "global duplicate, leakage, path, shape, and binary-mask checks."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Repository root. Defaults to this repository directory.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Expected processed square size. Default: 128",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace previously generated merged outputs.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"{path.name} is missing columns: {missing_columns}"
            )

        rows = list(reader)

    if not rows:
        raise RuntimeError(f"Source manifest is empty: {path}")

    return fieldnames, rows


def resolve_processed_path(processed_root: Path, value: str) -> Path:
    relative = Path(value)

    if relative.is_absolute():
        raise ValueError(
            "Processed manifest paths must be relative to processed/: "
            f"{value}"
        )

    resolved = (processed_root / relative).resolve()
    processed_resolved = processed_root.resolve()

    try:
        resolved.relative_to(processed_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Processed path escapes processed root: {value}"
        ) from exc

    return resolved


def read_gray(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"))
    except Exception as exc:
        raise RuntimeError(f"Could not read processed file: {path}") from exc


def validate_processed_image(path: Path, image_size: int) -> None:
    array = read_gray(path)

    if array.shape != (image_size, image_size):
        raise ValueError(
            f"Processed image has shape {array.shape}, expected "
            f"{(image_size, image_size)}: {path}"
        )

    if not np.isfinite(array).all():
        raise ValueError(f"Processed image contains non-finite values: {path}")


def validate_binary_mask(
    path: Path,
    image_size: int,
    require_all_foreground: bool = False,
) -> bool:
    array = read_gray(path)

    if array.shape != (image_size, image_size):
        raise ValueError(
            f"Processed mask has shape {array.shape}, expected "
            f"{(image_size, image_size)}: {path}"
        )

    unique_values = set(int(value) for value in np.unique(array))
    if not unique_values.issubset({0, 255}):
        raise ValueError(
            f"Processed mask is not binary 0/255: {path}; "
            f"values={sorted(unique_values)}"
        )

    foreground = array > 0

    if require_all_foreground and not foreground.all():
        raise ValueError(
            f"Valid-region mask is not fully foreground: {path}"
        )

    return bool(foreground.any())


def write_csv(
    path: Path,
    rows: Iterable[dict[str, object]],
    columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare_outputs(processed_root: Path, overwrite: bool) -> dict[str, Path]:
    outputs = {
        "manifest": processed_root / "manifest.csv",
        "train_manifest": processed_root / "manifest_train.csv",
        "heldout_manifest": processed_root / "manifest_heldout.csv",
        "task_summary": processed_root / "task_summary.csv",
        "dataset_summary": processed_root / "dataset_summary.csv",
        "metadata": processed_root / "metadata.json",
    }

    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Merged outputs already exist:\n"
            f"{formatted}\n"
            "Review them or rerun with --overwrite."
        )

    if overwrite:
        for path in outputs.values():
            if path.exists():
                path.unlink()

    return outputs


def main() -> None:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    processed_root = project_root / "data" / "processed"

    if not processed_root.is_dir():
        raise FileNotFoundError(
            f"Processed root does not exist: {processed_root}"
        )

    outputs = prepare_outputs(processed_root, args.overwrite)

    all_rows: list[dict[str, str]] = []
    source_manifest_metadata: dict[str, dict[str, object]] = {}

    for manifest_name in SOURCE_MANIFESTS:
        manifest_path = processed_root / manifest_name
        _, rows = read_csv_rows(manifest_path)

        expected_count = EXPECTED_ROW_COUNTS[manifest_name]
        if len(rows) != expected_count:
            raise RuntimeError(
                f"{manifest_name} has {len(rows)} rows; "
                f"expected {expected_count}."
            )

        source_manifest_metadata[manifest_name] = {
            "path": str(manifest_path),
            "rows": len(rows),
            "sha256": sha256_file(manifest_path),
        }

        for row in rows:
            row = dict(row)
            row["source_manifest"] = manifest_name
            all_rows.append(row)

    expected_total_rows = sum(EXPECTED_ROW_COUNTS.values())
    if len(all_rows) != expected_total_rows:
        raise RuntimeError(
            f"Expected {expected_total_rows} merged rows, "
            f"found {len(all_rows)}."
        )

    sample_ids = [row["sample_id"] for row in all_rows]
    duplicate_sample_ids = [
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if count > 1
    ]
    if duplicate_sample_ids:
        raise RuntimeError(
            "Duplicate sample IDs across manifests: "
            f"{duplicate_sample_ids[:20]}"
        )

    task_roles: dict[str, set[str]] = defaultdict(set)
    dataset_roles: dict[str, set[str]] = defaultdict(set)
    subject_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    subject_roles: dict[tuple[str, str], set[str]] = defaultdict(set)

    checked_images: set[Path] = set()
    checked_masks: set[Path] = set()
    checked_valid_masks: set[Path] = set()

    task_counts: Counter[str] = Counter()
    dataset_row_counts: Counter[str] = Counter()
    role_row_counts: Counter[str] = Counter()
    split_row_counts: Counter[tuple[str, str]] = Counter()
    foreground_counts: Counter[str] = Counter()

    for index, row in enumerate(all_rows, start=1):
        task_id = row["task_id"]
        task_role = row["task_role"]
        dataset = row["dataset"]
        subject_id = row["subject_id"]
        split = row["split"]

        if task_role not in ALLOWED_ROLES:
            raise ValueError(
                f"Invalid task_role={task_role!r} for {row['sample_id']}"
            )
        if split not in ALLOWED_SPLITS:
            raise ValueError(
                f"Invalid split={split!r} for {row['sample_id']}"
            )
        if row["dataset"] not in EXPECTED_DATASET_ROLE:
            raise ValueError(
                f"Unexpected dataset={dataset!r} for {row['sample_id']}"
            )
        if task_id not in EXPECTED_TASK_ROLE:
            raise ValueError(
                f"Unexpected task_id={task_id!r}"
            )
        if EXPECTED_TASK_ROLE[task_id] != task_role:
            raise ValueError(
                f"Role mismatch for {task_id}: "
                f"manifest={task_role}, expected={EXPECTED_TASK_ROLE[task_id]}"
            )
        if EXPECTED_DATASET_ROLE[dataset] != task_role:
            raise ValueError(
                f"Dataset-role mismatch for {dataset}: "
                f"manifest={task_role}, "
                f"expected={EXPECTED_DATASET_ROLE[dataset]}"
            )

        task_roles[task_id].add(task_role)
        dataset_roles[dataset].add(task_role)
        subject_splits[(dataset, subject_id)].add(split)
        subject_roles[(dataset, subject_id)].add(task_role)

        image_path = resolve_processed_path(
            processed_root,
            row["image_path"],
        )
        mask_path = resolve_processed_path(
            processed_root,
            row["mask_path"],
        )
        valid_mask_path = resolve_processed_path(
            processed_root,
            row["valid_mask_path"],
        )

        for path in (image_path, mask_path, valid_mask_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Manifest references missing file: {path}"
                )

        if image_path not in checked_images:
            validate_processed_image(image_path, args.image_size)
            checked_images.add(image_path)

        if mask_path not in checked_masks:
            actual_has_foreground = validate_binary_mask(
                mask_path,
                args.image_size,
            )
            checked_masks.add(mask_path)
        else:
            actual_has_foreground = bool(read_gray(mask_path).max() > 0)

        if valid_mask_path not in checked_valid_masks:
            validate_binary_mask(
                valid_mask_path,
                args.image_size,
                require_all_foreground=True,
            )
            checked_valid_masks.add(valid_mask_path)

        manifest_has_foreground = int(row["has_foreground"])
        if manifest_has_foreground not in {0, 1}:
            raise ValueError(
                f"Invalid has_foreground for {row['sample_id']}: "
                f"{row['has_foreground']}"
            )
        if manifest_has_foreground != int(actual_has_foreground):
            raise RuntimeError(
                f"has_foreground mismatch for {row['sample_id']}: "
                f"manifest={manifest_has_foreground}, "
                f"file={int(actual_has_foreground)}"
            )

        task_counts[task_id] += 1
        dataset_row_counts[dataset] += 1
        role_row_counts[task_role] += 1
        split_row_counts[(task_role, split)] += 1
        foreground_counts[task_id] += manifest_has_foreground

        if index % 250 == 0 or index == len(all_rows):
            print(
                f"Validated {index}/{len(all_rows)} manifest rows",
                flush=True,
            )

    inconsistent_subject_splits = {
        f"{dataset}:{subject_id}": sorted(splits)
        for (dataset, subject_id), splits in subject_splits.items()
        if len(splits) != 1
    }
    if inconsistent_subject_splits:
        raise RuntimeError(
            "Subjects occur in multiple splits: "
            f"{dict(list(inconsistent_subject_splits.items())[:20])}"
        )

    mixed_subject_roles = {
        f"{dataset}:{subject_id}": sorted(roles)
        for (dataset, subject_id), roles in subject_roles.items()
        if len(roles) != 1
    }
    if mixed_subject_roles:
        raise RuntimeError(
            "Subjects occur in both train and heldout roles: "
            f"{dict(list(mixed_subject_roles.items())[:20])}"
        )

    if any(len(roles) != 1 for roles in task_roles.values()):
        raise RuntimeError(
            "At least one task has mixed train/heldout roles."
        )
    if any(len(roles) != 1 for roles in dataset_roles.values()):
        raise RuntimeError(
            "At least one dataset has mixed train/heldout roles."
        )

    # Deterministic ordering improves reproducibility and diffs.
    role_order = {"train": 0, "heldout": 1}
    split_order = {"support": 0, "dev": 1, "test": 2}

    all_rows.sort(
        key=lambda row: (
            role_order[row["task_role"]],
            row["dataset"],
            row["task_id"],
            split_order[row["split"]],
            row["subject_id"],
            row["sample_id"],
        )
    )

    train_rows = [
        row for row in all_rows if row["task_role"] == "train"
    ]
    heldout_rows = [
        row for row in all_rows if row["task_role"] == "heldout"
    ]

    output_columns = list(REQUIRED_COLUMNS) + ["source_manifest"]

    write_csv(outputs["manifest"], all_rows, output_columns)
    write_csv(outputs["train_manifest"], train_rows, output_columns)
    write_csv(outputs["heldout_manifest"], heldout_rows, output_columns)

    rows_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    rows_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in all_rows:
        rows_by_task[row["task_id"]].append(row)
        rows_by_dataset[row["dataset"]].append(row)

    task_summary_rows: list[dict[str, object]] = []
    for task_id in sorted(rows_by_task):
        rows = rows_by_task[task_id]
        roles = {row["task_role"] for row in rows}
        datasets = {row["dataset"] for row in rows}
        modalities = {row["modality"] for row in rows}
        anatomies = {row["anatomy"] for row in rows}
        labels = {row["target_label"] for row in rows}

        if not (
            len(roles)
            == len(datasets)
            == len(modalities)
            == len(anatomies)
            == len(labels)
            == 1
        ):
            raise RuntimeError(
                f"Inconsistent metadata inside task {task_id}."
            )

        split_counts = Counter(row["split"] for row in rows)
        foreground_rows = sum(int(row["has_foreground"]) for row in rows)

        task_summary_rows.append(
            {
                "task_id": task_id,
                "task_role": next(iter(roles)),
                "dataset": next(iter(datasets)),
                "modality": next(iter(modalities)),
                "anatomy": next(iter(anatomies)),
                "target_label": next(iter(labels)),
                "total_rows": len(rows),
                "unique_subjects": len(
                    {row["subject_id"] for row in rows}
                ),
                "support_rows": split_counts["support"],
                "dev_rows": split_counts["dev"],
                "test_rows": split_counts["test"],
                "foreground_rows": foreground_rows,
                "empty_rows": len(rows) - foreground_rows,
            }
        )

    dataset_summary_rows: list[dict[str, object]] = []
    for dataset in sorted(rows_by_dataset):
        rows = rows_by_dataset[dataset]
        roles = {row["task_role"] for row in rows}
        if len(roles) != 1:
            raise RuntimeError(
                f"Dataset {dataset} has mixed roles."
            )

        subject_to_split: dict[str, str] = {}
        for row in rows:
            subject_to_split[row["subject_id"]] = row["split"]

        subject_split_counts = Counter(subject_to_split.values())

        dataset_summary_rows.append(
            {
                "dataset": dataset,
                "task_role": next(iter(roles)),
                "total_rows": len(rows),
                "unique_subjects": len(subject_to_split),
                "task_count": len({row["task_id"] for row in rows}),
                "support_subjects": subject_split_counts["support"],
                "dev_subjects": subject_split_counts["dev"],
                "test_subjects": subject_split_counts["test"],
                "foreground_rows": sum(
                    int(row["has_foreground"]) for row in rows
                ),
                "empty_rows": sum(
                    1 - int(row["has_foreground"]) for row in rows
                ),
            }
        )

    write_csv(
        outputs["task_summary"],
        task_summary_rows,
        [
            "task_id",
            "task_role",
            "dataset",
            "modality",
            "anatomy",
            "target_label",
            "total_rows",
            "unique_subjects",
            "support_rows",
            "dev_rows",
            "test_rows",
            "foreground_rows",
            "empty_rows",
        ],
    )

    write_csv(
        outputs["dataset_summary"],
        dataset_summary_rows,
        [
            "dataset",
            "task_role",
            "total_rows",
            "unique_subjects",
            "task_count",
            "support_subjects",
            "dev_subjects",
            "test_subjects",
            "foreground_rows",
            "empty_rows",
        ],
    )

    metadata = {
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "processed_root": str(processed_root),
        "image_size": [args.image_size, args.image_size],
        "source_manifests": source_manifest_metadata,
        "outputs": {
            name: str(path) for name, path in outputs.items()
        },
        "output_sha256": {
            "manifest.csv": sha256_file(outputs["manifest"]),
            "manifest_train.csv": sha256_file(
                outputs["train_manifest"]
            ),
            "manifest_heldout.csv": sha256_file(
                outputs["heldout_manifest"]
            ),
            "task_summary.csv": sha256_file(
                outputs["task_summary"]
            ),
            "dataset_summary.csv": sha256_file(
                outputs["dataset_summary"]
            ),
        },
        "row_counts": {
            "all": len(all_rows),
            "train": len(train_rows),
            "heldout": len(heldout_rows),
        },
        "role_row_counts": dict(sorted(role_row_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "dataset_row_counts": dict(sorted(dataset_row_counts.items())),
        "split_row_counts_by_role": {
            role: {
                split: split_row_counts[(role, split)]
                for split in ("support", "dev", "test")
            }
            for role in ("train", "heldout")
        },
        "foreground_counts_by_task": dict(
            sorted(foreground_counts.items())
        ),
        "unique_processed_files_checked": {
            "images": len(checked_images),
            "masks": len(checked_masks),
            "valid_masks": len(checked_valid_masks),
        },
        "training_tasks": sorted(
            task_id
            for task_id, role in EXPECTED_TASK_ROLE.items()
            if role == "train"
        ),
        "heldout_tasks": sorted(
            task_id
            for task_id, role in EXPECTED_TASK_ROLE.items()
            if role == "heldout"
        ),
        "qualitative_only_datasets_not_in_manifest": ["stare"],
        "excluded_datasets": {
            "stacked_carton_scd": "non-medical dataset",
        },
        "checks_completed": [
            "All four source manifests exist and have expected row counts.",
            "All required manifest columns are present.",
            "No global duplicate sample IDs exist.",
            "Every task and dataset has one fixed train/heldout role.",
            "No subject appears in multiple splits within a dataset.",
            "No subject appears in both train and heldout roles.",
            "Every processed image, mask, and valid mask exists.",
            "Every processed file is 128x128.",
            "Every task mask is binary 0/255.",
            "Every valid mask is binary and fully valid.",
            "Every has_foreground value matches its saved mask.",
            "Merged output order is deterministic.",
        ],
        "scientific_scope": (
            "This merged index supports the reduced, CPU-friendly, "
            "from-scratch methodological reproduction. It does not recreate "
            "the complete MegaMedical training collection or the exact "
            "numerical experiments of the original UniverSeg paper."
        ),
    }

    outputs["metadata"].write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n=== UNIFIED MANIFEST AUDIT PASSED ===")
    print(f"All rows:      {len(all_rows)}")
    print(f"Training rows: {len(train_rows)}")
    print(f"Held-out rows: {len(heldout_rows)}")
    print(f"Tasks:         {len(rows_by_task)}")
    print(f"Datasets:      {len(rows_by_dataset)}")
    print(f"Manifest:      {outputs['manifest']}")
    print(f"Metadata:      {outputs['metadata']}")


if __name__ == "__main__":
    main()
