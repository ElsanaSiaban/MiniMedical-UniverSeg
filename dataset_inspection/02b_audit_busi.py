from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_SIZE = 128

REQUIRED_COLUMNS = [
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
]

EXPECTED_CLASS_COUNTS = {
    "benign": 437,
    "malignant": 210,
    "normal": 133,
}

EXPECTED_TASK_COUNTS = {
    "busi_all_lesions": 780,
    "busi_benign_lesion": 570,
    "busi_malignant_lesion": 343,
}

EXPECTED_TASKS_BY_CLASS = {
    "benign": {
        "busi_all_lesions",
        "busi_benign_lesion",
    },
    "malignant": {
        "busi_all_lesions",
        "busi_malignant_lesion",
    },
    "normal": {
        "busi_all_lesions",
        "busi_benign_lesion",
        "busi_malignant_lesion",
    },
}

RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit an existing BUSI preprocessing run. This script is "
            "read-only except for writing a JSON audit report."
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
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in fieldnames
        ]
        if missing:
            raise ValueError(
                f"Manifest is missing columns: {missing}"
            )

        rows = list(reader)

    if not rows:
        raise RuntimeError("BUSI manifest is empty.")

    return rows


def class_from_subject(subject_id: str) -> str:
    match = re.fullmatch(
        r"busi_(benign|malignant|normal)_.+",
        subject_id,
    )
    if match is None:
        raise ValueError(
            f"Unexpected BUSI subject ID: {subject_id}"
        )
    return match.group(1)


def split_source_mask_paths(value: str) -> list[Path]:
    text = str(value).strip()
    if not text:
        return []
    return [Path(item) for item in text.split("|") if item]


def read_gray(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"))
    except Exception as exc:
        raise RuntimeError(f"Could not read image: {path}") from exc


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)

    if array.size == 0:
        return np.zeros(
            (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
            dtype=np.uint8,
        )

    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.uint8)

    minimum = float(array[finite].min())
    maximum = float(array[finite].max())

    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.uint8)

    normalized = (array - minimum) / (maximum - minimum)
    return np.clip(
        normalized * 255.0,
        0,
        255,
    ).astype(np.uint8)


def resize_gray_image(
    image: np.ndarray,
    image_size: int,
) -> np.ndarray:
    image_uint8 = normalize_to_uint8(image)
    pil_image = Image.fromarray(image_uint8, mode="L")
    return np.asarray(
        pil_image.resize(
            (image_size, image_size),
            resample=RESAMPLE_BILINEAR,
        ),
        dtype=np.uint8,
    )


def resize_binary_mask(
    mask: np.ndarray,
    image_size: int,
) -> np.ndarray:
    mask_uint8 = (
        np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    )
    pil_mask = Image.fromarray(mask_uint8, mode="L")
    resized = np.asarray(
        pil_mask.resize(
            (image_size, image_size),
            resample=RESAMPLE_NEAREST,
        ),
        dtype=np.uint8,
    )
    return resized > 0


def resolve_processed_path(
    processed_root: Path,
    relative_path: str,
) -> Path:
    path = processed_root / Path(relative_path)
    return path.resolve()


def assert_binary_png(
    path: Path,
    expected_size: int,
) -> np.ndarray:
    array = read_gray(path)

    if array.shape != (expected_size, expected_size):
        raise ValueError(
            f"Unexpected processed mask size for {path}: "
            f"{array.shape}"
        )

    unique_values = set(int(value) for value in np.unique(array))
    if not unique_values.issubset({0, 255}):
        raise ValueError(
            f"Non-binary processed mask {path}: "
            f"{sorted(unique_values)}"
        )

    return array > 0


def compare_exact(
    expected: np.ndarray,
    actual: np.ndarray,
    description: str,
) -> None:
    if expected.shape != actual.shape:
        raise ValueError(
            f"{description} shape mismatch: "
            f"{expected.shape} versus {actual.shape}"
        )

    if not np.array_equal(expected, actual):
        different_pixels = int(np.count_nonzero(expected != actual))
        raise ValueError(
            f"{description} differs in {different_pixels} pixels."
        )


def main() -> None:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    processed_root = project_root / "data" / "processed"
    manifest_path = processed_root / "manifest_busi.csv"

    rows = read_manifest(manifest_path)

    if len(rows) != 1693:
        raise RuntimeError(
            f"Expected 1693 BUSI rows, found {len(rows)}."
        )

    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate sample_id values found.")

    subject_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        subject_rows[row["subject_id"]].append(row)

    if len(subject_rows) != 780:
        raise RuntimeError(
            f"Expected 780 BUSI subjects, found {len(subject_rows)}."
        )

    class_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    split_subject_counts: Counter[tuple[str, str]] = Counter()
    split_task_counts: Counter[tuple[str, str]] = Counter()
    foreground_task_counts: Counter[str] = Counter()
    source_mask_count_histogram: Counter[int] = Counter()
    multiple_mask_subjects: list[str] = []

    processed_images_checked: set[Path] = set()
    processed_valid_masks_checked: set[Path] = set()

    for subject_index, (
        subject_id,
        subject_manifest_rows,
    ) in enumerate(
        sorted(subject_rows.items()),
        start=1,
    ):
        class_name = class_from_subject(subject_id)
        class_counts[class_name] += 1

        task_ids = {
            row["task_id"]
            for row in subject_manifest_rows
        }
        if task_ids != EXPECTED_TASKS_BY_CLASS[class_name]:
            raise RuntimeError(
                f"Unexpected task membership for {subject_id}: "
                f"{sorted(task_ids)}"
            )

        if len(subject_manifest_rows) != len(task_ids):
            raise RuntimeError(
                f"Duplicate subject-task row for {subject_id}."
            )

        splits = {
            row["split"]
            for row in subject_manifest_rows
        }
        if len(splits) != 1:
            raise RuntimeError(
                f"Split inconsistency for {subject_id}: {splits}"
            )
        split = next(iter(splits))

        if split not in {"support", "dev", "test"}:
            raise RuntimeError(
                f"Unexpected split for {subject_id}: {split}"
            )

        image_paths = {
            row["image_path"]
            for row in subject_manifest_rows
        }
        valid_paths = {
            row["valid_mask_path"]
            for row in subject_manifest_rows
        }
        source_image_paths = {
            row["source_image_path"]
            for row in subject_manifest_rows
        }
        source_mask_path_strings = {
            row["source_mask_paths"]
            for row in subject_manifest_rows
        }

        if len(image_paths) != 1:
            raise RuntimeError(
                f"Processed image inconsistency for {subject_id}."
            )
        if len(valid_paths) != 1:
            raise RuntimeError(
                f"Valid-mask inconsistency for {subject_id}."
            )
        if len(source_image_paths) != 1:
            raise RuntimeError(
                f"Source-image inconsistency for {subject_id}."
            )
        if len(source_mask_path_strings) != 1:
            raise RuntimeError(
                f"Source-mask inconsistency for {subject_id}."
            )

        source_image_path = Path(
            next(iter(source_image_paths))
        )
        source_mask_paths = split_source_mask_paths(
            next(iter(source_mask_path_strings))
        )

        if "_mask" in source_image_path.stem.lower():
            raise RuntimeError(
                f"A mask file was treated as an image: "
                f"{source_image_path}"
            )
        if not source_image_path.is_file():
            raise FileNotFoundError(source_image_path)

        if class_name in {"benign", "malignant"}:
            if not source_mask_paths:
                raise RuntimeError(
                    f"Positive BUSI subject has no source mask: "
                    f"{subject_id}"
                )

        for source_mask_path in source_mask_paths:
            if not source_mask_path.is_file():
                raise FileNotFoundError(source_mask_path)

        source_mask_count_histogram[
            len(source_mask_paths)
        ] += 1
        if len(source_mask_paths) > 1:
            multiple_mask_subjects.append(subject_id)

        source_image = read_gray(source_image_path)

        if source_mask_paths:
            source_masks = [
                read_gray(path)
                for path in source_mask_paths
            ]

            for source_mask in source_masks:
                if source_mask.shape != source_image.shape:
                    raise ValueError(
                        f"Source image-mask shape mismatch for "
                        f"{subject_id}: image={source_image.shape}, "
                        f"mask={source_mask.shape}"
                    )

            lesion_mask = np.zeros_like(
                source_image,
                dtype=bool,
            )
            for source_mask in source_masks:
                lesion_mask |= source_mask > 0
        else:
            lesion_mask = np.zeros_like(
                source_image,
                dtype=bool,
            )

        if class_name == "normal" and lesion_mask.any():
            raise RuntimeError(
                f"Normal BUSI source mask is not empty: "
                f"{subject_id}"
            )

        if (
            class_name in {"benign", "malignant"}
            and not lesion_mask.any()
        ):
            raise RuntimeError(
                f"Positive BUSI source lesion is empty: "
                f"{subject_id}"
            )

        expected_processed_image = resize_gray_image(
            source_image,
            args.image_size,
        )
        processed_image_path = resolve_processed_path(
            processed_root,
            next(iter(image_paths)),
        )

        if processed_image_path not in processed_images_checked:
            actual_processed_image = read_gray(
                processed_image_path
            )
            if actual_processed_image.shape != (
                args.image_size,
                args.image_size,
            ):
                raise ValueError(
                    f"Unexpected processed image size for "
                    f"{processed_image_path}: "
                    f"{actual_processed_image.shape}"
                )

            compare_exact(
                expected_processed_image,
                actual_processed_image,
                f"Processed image for {subject_id}",
            )
            processed_images_checked.add(
                processed_image_path
            )

        valid_mask_path = resolve_processed_path(
            processed_root,
            next(iter(valid_paths)),
        )
        if valid_mask_path not in processed_valid_masks_checked:
            valid_mask = assert_binary_png(
                valid_mask_path,
                args.image_size,
            )
            if not valid_mask.all():
                raise RuntimeError(
                    f"BUSI valid mask is not all foreground: "
                    f"{valid_mask_path}"
                )
            processed_valid_masks_checked.add(
                valid_mask_path
            )

        expected_masks = {
            "busi_all_lesions": lesion_mask,
        }
        if class_name in {"benign", "normal"}:
            expected_masks["busi_benign_lesion"] = (
                lesion_mask
                if class_name == "benign"
                else np.zeros_like(lesion_mask)
            )
        if class_name in {"malignant", "normal"}:
            expected_masks["busi_malignant_lesion"] = (
                lesion_mask
                if class_name == "malignant"
                else np.zeros_like(lesion_mask)
            )

        for row in subject_manifest_rows:
            task_id = row["task_id"]
            task_counts[task_id] += 1
            split_task_counts[(task_id, split)] += 1

            expected_processed_mask = resize_binary_mask(
                expected_masks[task_id],
                args.image_size,
            )
            processed_mask_path = resolve_processed_path(
                processed_root,
                row["mask_path"],
            )
            actual_processed_mask = assert_binary_png(
                processed_mask_path,
                args.image_size,
            )

            compare_exact(
                expected_processed_mask,
                actual_processed_mask,
                f"{task_id} mask for {subject_id}",
            )

            expected_has_foreground = int(
                expected_processed_mask.any()
            )
            actual_has_foreground = int(
                row["has_foreground"]
            )

            if (
                expected_has_foreground
                != actual_has_foreground
            ):
                raise RuntimeError(
                    f"has_foreground mismatch for "
                    f"{row['sample_id']}: "
                    f"expected={expected_has_foreground}, "
                    f"manifest={actual_has_foreground}"
                )

            foreground_task_counts[
                task_id
            ] += expected_has_foreground

        split_subject_counts[
            (class_name, split)
        ] += 1

        if subject_index % 25 == 0 or subject_index == len(
            subject_rows
        ):
            print(
                f"Audited {subject_index}/{len(subject_rows)} "
                "BUSI subjects",
                flush=True,
            )

    if dict(class_counts) != EXPECTED_CLASS_COUNTS:
        raise RuntimeError(
            f"Unexpected class counts: {dict(class_counts)}"
        )

    if dict(task_counts) != EXPECTED_TASK_COUNTS:
        raise RuntimeError(
            f"Unexpected task counts: {dict(task_counts)}"
        )

    expected_foreground_counts = {
        "busi_all_lesions": 647,
        "busi_benign_lesion": 437,
        "busi_malignant_lesion": 210,
    }
    if dict(foreground_task_counts) != (
        expected_foreground_counts
    ):
        raise RuntimeError(
            "Unexpected foreground counts: "
            f"{dict(foreground_task_counts)}"
        )

    summary: dict[str, Any] = {
        "status": "passed",
        "manifest": str(manifest_path),
        "manifest_rows": len(rows),
        "unique_subjects": len(subject_rows),
        "class_counts": dict(class_counts),
        "task_counts": dict(task_counts),
        "foreground_task_counts": dict(
            foreground_task_counts
        ),
        "subject_split_counts_by_class": {
            class_name: {
                split: split_subject_counts[
                    (class_name, split)
                ]
                for split in ("support", "dev", "test")
            }
            for class_name in (
                "benign",
                "malignant",
                "normal",
            )
        },
        "task_split_counts": {
            task_id: {
                split: split_task_counts[
                    (task_id, split)
                ]
                for split in ("support", "dev", "test")
            }
            for task_id in EXPECTED_TASK_COUNTS
        },
        "source_mask_count_histogram": {
            str(count): subjects
            for count, subjects in sorted(
                source_mask_count_histogram.items()
            )
        },
        "multiple_mask_subject_count": len(
            multiple_mask_subjects
        ),
        "multiple_mask_subjects": multiple_mask_subjects,
        "processed_images_checked": len(
            processed_images_checked
        ),
        "processed_valid_masks_checked": len(
            processed_valid_masks_checked
        ),
        "checks_completed": [
            "All 780 source images exist.",
            "No source mask was treated as an image.",
            "All source image-mask dimensions match.",
            "All multiple masks were recombined with logical OR.",
            "All positive subjects have nonempty source lesions.",
            "All normal source masks are empty.",
            "All processed images exactly match preprocessing.",
            "All processed masks exactly match expected masks.",
            "All processed masks are binary 0/255.",
            "All valid masks are 128x128 and fully valid.",
            "All subject splits are consistent across tasks.",
            "No duplicate sample IDs or subject-task rows exist.",
        ],
    }

    output_path = (
        project_root
        / "outputs"
        / "busi_audit_summary.json"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== BUSI AUDIT PASSED ===")
    print(f"Rows:     {len(rows)}")
    print(f"Subjects: {len(subject_rows)}")
    print(f"Classes:  {dict(class_counts)}")
    print(f"Tasks:    {dict(task_counts)}")
    print(
        "Foreground:",
        dict(foreground_task_counts),
    )
    print(
        "Multiple-mask subjects:",
        len(multiple_mask_subjects),
    )
    print(f"Report:   {output_path}")


if __name__ == "__main__":
    main()
