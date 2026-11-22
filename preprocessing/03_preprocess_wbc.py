from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WBC_FOLDER = "segmentation_WBC-master"
DEFAULT_IMAGE_SIZE = 128
DEFAULT_SEED = 42

IMAGE_EXTENSIONS = {".bmp"}
MASK_EXTENSIONS = {".png"}

RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST

DATASET_CONFIG = {
    1: {
        "role": "train",
        "dataset_name": "wbc_dataset1",
        "task_prefix": "wbc1",
        "expected_count": 300,
        "split_seed_offset": 201,
    },
    2: {
        "role": "heldout",
        "dataset_name": "wbc_dataset2",
        "task_prefix": "wbc2",
        "expected_count": 100,
        "split_seed_offset": 202,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the WBC segmentation dataset for the reduced "
            "UniverSeg reproduction. Dataset 1 is used for training and "
            "Dataset 2 is held out. Each source image produces whole-cell "
            "and nucleus binary tasks."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Repository root. Defaults to this repository directory.",
    )
    parser.add_argument(
        "--wbc-folder",
        type=str,
        default=DEFAULT_WBC_FOLDER,
        help=(
            "Folder inside MiniMedical. "
            "Default: segmentation_WBC-master"
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Square output size. Default: 128",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base deterministic split seed. Default: 42",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete and regenerate only WBC-specific processed outputs. "
            "Required when replacing an earlier WBC preprocessing run."
        ),
    )
    return parser.parse_args()


def read_gray(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"))
    except Exception as exc:
        raise RuntimeError(f"Could not read image: {path}") from exc


def normalize_image_to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, received {array.shape}."
        )
    if array.size == 0:
        raise ValueError("Encountered an empty WBC image.")
    if not np.isfinite(array).all():
        raise ValueError("WBC image contains non-finite values.")

    minimum = float(array.min())
    maximum = float(array.max())

    if maximum <= minimum:
        raise ValueError(
            "WBC microscopy image has no intensity variation."
        )

    normalized = (array - minimum) / (maximum - minimum)
    return np.clip(
        np.rint(normalized * 255.0),
        0,
        255,
    ).astype(np.uint8)


def resize_image(
    image: np.ndarray,
    image_size: int,
) -> np.ndarray:
    image_uint8 = normalize_image_to_uint8(image)
    pil_image = Image.fromarray(image_uint8, mode="L")
    resized = pil_image.resize(
        (image_size, image_size),
        resample=RESAMPLE_BILINEAR,
    )
    return np.asarray(resized, dtype=np.uint8)


def resize_mask(
    mask: np.ndarray,
    image_size: int,
) -> np.ndarray:
    mask_uint8 = (
        np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    )
    pil_mask = Image.fromarray(mask_uint8, mode="L")
    resized = pil_mask.resize(
        (image_size, image_size),
        resample=RESAMPLE_NEAREST,
    )
    return np.asarray(resized, dtype=np.uint8) > 0


def decode_ground_truth(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """
    Decode the supplied WBC grayscale annotation.

    Expected semantic ordering:
      darkest intensity  -> background
      middle intensity   -> cytoplasm
      brightest intensity -> nucleus

    Returned binary tasks:
      whole_cell = cytoplasm OR nucleus
      nucleus    = nucleus only
    """
    gray = np.asarray(mask, dtype=np.float32)

    if gray.ndim != 2:
        raise ValueError(
            f"Expected a 2D WBC annotation, received {gray.shape}."
        )
    if gray.size == 0:
        raise ValueError("Encountered an empty WBC annotation.")
    if not np.isfinite(gray).all():
        raise ValueError("WBC annotation contains non-finite values.")

    unique_values = np.unique(gray)
    if unique_values.size < 3:
        raise ValueError(
            "Expected at least background, cytoplasm, and nucleus "
            f"intensities; found {unique_values.tolist()}."
        )

    minimum = float(unique_values.min())
    maximum = float(unique_values.max())
    value_range = maximum - minimum

    if value_range <= 0:
        raise ValueError("WBC annotation has no intensity range.")

    if unique_values.size <= 10:
        sorted_values = np.sort(unique_values)
        background_value = float(sorted_values[0])
        nucleus_value = float(sorted_values[-1])

        whole_cell = gray > background_value
        nucleus = gray == nucleus_value
        decoding_mode = "discrete_unique_values"
    else:
        whole_cell_threshold = minimum + 0.10 * value_range
        nucleus_threshold = minimum + 0.75 * value_range

        whole_cell = gray > whole_cell_threshold
        nucleus = gray > nucleus_threshold
        background_value = minimum
        nucleus_value = nucleus_threshold
        decoding_mode = "range_thresholds"

    if not whole_cell.any():
        raise ValueError("Decoded whole-cell mask is empty.")
    if not nucleus.any():
        raise ValueError("Decoded nucleus mask is empty.")
    if np.any(nucleus & ~whole_cell):
        raise ValueError("Nucleus is not a subset of whole-cell mask.")

    cytoplasm = whole_cell & ~nucleus
    if not cytoplasm.any():
        raise ValueError(
            "Decoded whole-cell mask contains no cytoplasm pixels."
        )

    diagnostics: dict[str, object] = {
        "unique_value_count": int(unique_values.size),
        "unique_values": [
            float(value) for value in unique_values[:20]
        ],
        "unique_values_truncated": bool(
            unique_values.size > 20
        ),
        "minimum": minimum,
        "maximum": maximum,
        "background_reference": background_value,
        "nucleus_reference": nucleus_value,
        "decoding_mode": decoding_mode,
        "whole_cell_fraction": float(whole_cell.mean()),
        "nucleus_fraction": float(nucleus.mean()),
        "cytoplasm_fraction": float(cytoplasm.mean()),
    }

    return whole_cell, nucleus, diagnostics


def make_subject_splits(
    subject_ids: Iterable[str],
    seed: int,
) -> dict[str, str]:
    ids = sorted(set(str(value) for value in subject_ids))

    if len(ids) < 5:
        raise ValueError(
            "At least five subjects are required for 60/20/20."
        )

    rng = random.Random(seed)
    rng.shuffle(ids)

    total = len(ids)
    n_support = int(round(total * 0.60))
    n_dev = int(round(total * 0.20))
    n_test = total - n_support - n_dev

    if min(n_support, n_dev, n_test) < 1:
        raise ValueError(
            "Invalid split sizes: "
            f"support={n_support}, dev={n_dev}, test={n_test}."
        )

    split_map: dict[str, str] = {}

    for subject_id in ids[:n_support]:
        split_map[subject_id] = "support"

    for subject_id in ids[n_support:n_support + n_dev]:
        split_map[subject_id] = "dev"

    for subject_id in ids[n_support + n_dev:]:
        split_map[subject_id] = "test"

    return split_map


def discover_pairs(
    folder: Path,
    expected_count: int,
) -> list[tuple[str, Path, Path]]:
    image_map = {
        path.stem.lower(): path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    mask_map = {
        path.stem.lower(): path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in MASK_EXTENSIONS
    }

    image_only = sorted(set(image_map) - set(mask_map))
    mask_only = sorted(set(mask_map) - set(image_map))
    common_ids = sorted(set(image_map) & set(mask_map))

    if image_only or mask_only:
        raise RuntimeError(
            f"Unpaired WBC files in {folder}.\n"
            f"Images without masks: {image_only[:20]}\n"
            f"Masks without images: {mask_only[:20]}"
        )

    if len(common_ids) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} pairs in {folder}, "
            f"found {len(common_ids)}."
        )

    return [
        (item_id, image_map[item_id], mask_map[item_id])
        for item_id in common_ids
    ]


def save_png(
    array: np.ndarray,
    path: Path,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def relative_to_processed(
    path: Path,
    processed_root: Path,
) -> str:
    return path.relative_to(processed_root).as_posix()


def write_manifest(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    columns = [
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

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def wbc_output_locations(
    processed_root: Path,
) -> list[Path]:
    return [
        processed_root / "images" / "wbc_dataset1",
        processed_root / "images" / "wbc_dataset2",
        processed_root / "valid_masks" / "wbc_dataset1",
        processed_root / "valid_masks" / "wbc_dataset2",
        processed_root / "masks" / "wbc1_whole_cell",
        processed_root / "masks" / "wbc1_nucleus",
        processed_root / "masks" / "wbc1_cytoplasm",
        processed_root / "masks" / "wbc2_whole_cell",
        processed_root / "masks" / "wbc2_nucleus",
        processed_root / "masks" / "wbc2_cytoplasm",
        processed_root / "manifest_wbc.csv",
        processed_root / "wbc_preprocessing_summary.json",
    ]


def prepare_output_locations(
    processed_root: Path,
    overwrite: bool,
) -> None:
    locations = wbc_output_locations(processed_root)
    existing = [path for path in locations if path.exists()]

    if existing and not overwrite:
        details = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Existing WBC processed outputs were found:\n"
            f"{details}\n"
            "Rerun with --overwrite to replace only these WBC outputs."
        )

    if overwrite:
        for path in locations:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()


def main() -> None:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    data_root = project_root / "data" / "raw"
    processed_root = project_root / "data" / "processed"
    wbc_root = data_root / args.wbc_folder

    if not wbc_root.is_dir():
        raise FileNotFoundError(
            f"WBC root not found: {wbc_root}"
        )

    prepare_output_locations(
        processed_root,
        overwrite=args.overwrite,
    )

    rows: list[dict[str, object]] = []
    split_subject_counts: dict[str, Counter[str]] = {}
    task_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    decoding_mode_counts: Counter[str] = Counter()
    unique_value_count_histogram: Counter[int] = Counter()
    whole_cell_fractions: list[float] = []
    nucleus_fractions: list[float] = []

    for dataset_number in (1, 2):
        config = DATASET_CONFIG[dataset_number]
        folder = wbc_root / f"Dataset {dataset_number}"

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing WBC source folder: {folder}"
            )

        pairs = discover_pairs(
            folder,
            expected_count=int(config["expected_count"]),
        )

        split_map = make_subject_splits(
            (item_id for item_id, _, _ in pairs),
            seed=(
                args.seed
                + int(config["split_seed_offset"])
            ),
        )
        split_subject_counts[
            f"Dataset {dataset_number}"
        ] = Counter(split_map.values())

        for index, (
            item_id,
            source_image_path,
            source_mask_path,
        ) in enumerate(pairs, start=1):
            split = split_map[item_id]
            role = str(config["role"])
            dataset_name = str(config["dataset_name"])
            task_prefix = str(config["task_prefix"])
            subject_id = f"wbc{dataset_number}_{item_id}"

            print(
                f"WBC Dataset {dataset_number} "
                f"[{index:03d}/{len(pairs)}] "
                f"{subject_id} ({split}, {role})",
                flush=True,
            )

            source_image = read_gray(source_image_path)
            source_ground_truth = read_gray(source_mask_path)

            if source_image.shape != source_ground_truth.shape:
                raise ValueError(
                    f"Image-mask shape mismatch for {subject_id}: "
                    f"{source_image.shape} versus "
                    f"{source_ground_truth.shape}"
                )

            whole_cell, nucleus, diagnostics = (
                decode_ground_truth(source_ground_truth)
            )

            processed_image = resize_image(
                source_image,
                args.image_size,
            )
            processed_whole_cell = resize_mask(
                whole_cell,
                args.image_size,
            )
            processed_nucleus = resize_mask(
                nucleus,
                args.image_size,
            )
            processed_valid = np.ones(
                (args.image_size, args.image_size),
                dtype=np.uint8,
            ) * 255

            if not processed_whole_cell.any():
                raise ValueError(
                    f"Whole-cell mask became empty after resizing: "
                    f"{subject_id}"
                )
            if not processed_nucleus.any():
                raise ValueError(
                    f"Nucleus mask became empty after resizing: "
                    f"{subject_id}"
                )
            if np.any(
                processed_nucleus & ~processed_whole_cell
            ):
                raise ValueError(
                    "Processed nucleus is not contained in whole-cell "
                    f"mask: {subject_id}"
                )

            output_image = (
                processed_root
                / "images"
                / dataset_name
                / f"{subject_id}.png"
            )
            output_valid = (
                processed_root
                / "valid_masks"
                / dataset_name
                / f"{subject_id}.png"
            )

            save_png(
                processed_image,
                output_image,
                overwrite=args.overwrite,
            )
            save_png(
                processed_valid,
                output_valid,
                overwrite=args.overwrite,
            )

            task_definitions = {
                f"{task_prefix}_whole_cell": (
                    "whole_cell",
                    processed_whole_cell,
                ),
                f"{task_prefix}_nucleus": (
                    "nucleus",
                    processed_nucleus,
                ),
            }

            for task_id, (
                target_label,
                processed_mask,
            ) in task_definitions.items():
                output_mask = (
                    processed_root
                    / "masks"
                    / task_id
                    / f"{subject_id}.png"
                )
                save_png(
                    processed_mask.astype(np.uint8) * 255,
                    output_mask,
                    overwrite=args.overwrite,
                )

                rows.append(
                    {
                        "sample_id": (
                            f"{task_id}:{subject_id}"
                        ),
                        "task_id": task_id,
                        "task_role": role,
                        "dataset": dataset_name,
                        "modality": "microscopy",
                        "anatomy": "white_blood_cell",
                        "target_label": target_label,
                        "image_path": relative_to_processed(
                            output_image,
                            processed_root,
                        ),
                        "mask_path": relative_to_processed(
                            output_mask,
                            processed_root,
                        ),
                        "valid_mask_path": relative_to_processed(
                            output_valid,
                            processed_root,
                        ),
                        "subject_id": subject_id,
                        "split": split,
                        "has_foreground": int(
                            processed_mask.any()
                        ),
                        "source_image_path": str(
                            source_image_path
                        ),
                        "source_mask_paths": str(
                            source_mask_path
                        ),
                    }
                )

                task_counts[task_id] += 1
                role_counts[role] += 1

            decoding_mode_counts[
                str(diagnostics["decoding_mode"])
            ] += 1
            unique_value_count_histogram[
                int(diagnostics["unique_value_count"])
            ] += 1
            whole_cell_fractions.append(
                float(diagnostics["whole_cell_fraction"])
            )
            nucleus_fractions.append(
                float(diagnostics["nucleus_fraction"])
            )

    if len(rows) != 800:
        raise RuntimeError(
            f"Expected 800 WBC manifest rows, produced {len(rows)}."
        )

    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(
            "Duplicate WBC sample IDs were produced."
        )

    expected_task_counts = {
        "wbc1_whole_cell": 300,
        "wbc1_nucleus": 300,
        "wbc2_whole_cell": 100,
        "wbc2_nucleus": 100,
    }
    if dict(task_counts) != expected_task_counts:
        raise RuntimeError(
            "Unexpected WBC task counts: "
            f"{dict(task_counts)}"
        )

    manifest_path = processed_root / "manifest_wbc.csv"
    write_manifest(manifest_path, rows)

    whole_cell_array = np.asarray(
        whole_cell_fractions,
        dtype=np.float64,
    )
    nucleus_array = np.asarray(
        nucleus_fractions,
        dtype=np.float64,
    )

    summary = {
        "status": "completed",
        "project_scope": (
            "Reduced methodological reproduction. Dataset 1 is a training "
            "source; Dataset 2 is a held-out source subset."
        ),
        "source_root": str(wbc_root),
        "source_counts": {
            "Dataset 1": 300,
            "Dataset 2": 100,
        },
        "roles": {
            "Dataset 1": "train",
            "Dataset 2": "heldout",
        },
        "important_interpretation": (
            "Dataset 2 is held out at the image/source-subset level, but "
            "its modality, anatomy, and segmentation tasks are represented "
            "during training by Dataset 1. It is not a fully unseen task."
        ),
        "task_definitions": {
            "wbc1_whole_cell": (
                "Dataset 1 cytoplasm OR nucleus; training"
            ),
            "wbc1_nucleus": (
                "Dataset 1 nucleus; training"
            ),
            "wbc2_whole_cell": (
                "Dataset 2 cytoplasm OR nucleus; held out"
            ),
            "wbc2_nucleus": (
                "Dataset 2 nucleus; held out"
            ),
        },
        "removed_custom_tasks": [
            "wbc1_cytoplasm",
            "wbc2_cytoplasm",
        ],
        "output_image_size": [
            args.image_size,
            args.image_size,
        ],
        "image_normalization": (
            "Per-image min-max normalization to [0,255] before PNG "
            "storage; loaders should divide by 255."
        ),
        "image_interpolation": "bilinear",
        "mask_interpolation": "nearest-neighbor",
        "base_seed": args.seed,
        "dataset_specific_split_seeds": {
            "Dataset 1": (
                args.seed
                + int(
                    DATASET_CONFIG[1][
                        "split_seed_offset"
                    ]
                )
            ),
            "Dataset 2": (
                args.seed
                + int(
                    DATASET_CONFIG[2][
                        "split_seed_offset"
                    ]
                )
            ),
        },
        "subject_split_policy": (
            "Each source subset is split independently by subject into "
            "60% support, 20% development, and 20% test."
        ),
        "subject_split_counts": {
            key: dict(sorted(counter.items()))
            for key, counter in split_subject_counts.items()
        },
        "manifest_rows": len(rows),
        "task_counts": dict(task_counts),
        "role_row_counts": dict(role_counts),
        "decoding_mode_counts": dict(
            decoding_mode_counts
        ),
        "mask_unique_value_count_histogram": {
            str(key): value
            for key, value in sorted(
                unique_value_count_histogram.items()
            )
        },
        "whole_cell_fraction_source_resolution": {
            "minimum": float(whole_cell_array.min()),
            "median": float(
                np.median(whole_cell_array)
            ),
            "mean": float(whole_cell_array.mean()),
            "maximum": float(whole_cell_array.max()),
        },
        "nucleus_fraction_source_resolution": {
            "minimum": float(nucleus_array.min()),
            "median": float(
                np.median(nucleus_array)
            ),
            "mean": float(nucleus_array.mean()),
            "maximum": float(nucleus_array.max()),
        },
        "scientific_difference_from_original_paper": (
            "The original UniverSeg paper held out the complete WBC "
            "dataset. This reduced project trains on Dataset 1 and tests "
            "on Dataset 2, so its WBC score must not be directly compared "
            "with the paper's fully held-out WBC result."
        ),
    }

    summary_path = (
        processed_root / "wbc_preprocessing_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== WBC PREPROCESSING COMPLETE ===")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    print(f"Rows:     {len(rows)}")
    print(
        "Dataset 1 split:",
        dict(
            sorted(
                split_subject_counts[
                    "Dataset 1"
                ].items()
            )
        ),
    )
    print(
        "Dataset 2 split:",
        dict(
            sorted(
                split_subject_counts[
                    "Dataset 2"
                ].items()
            )
        ),
    )
    print("Tasks:", dict(task_counts))


if __name__ == "__main__":
    main()
