from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from PIL import Image


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACDC_FOLDER = "ACDC"
DEFAULT_IMAGE_SIZE = 128
DEFAULT_SEED = 42

RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST

LABEL_TO_TASK = {
    1: ("acdc_right_ventricle", "right_ventricle"),
    2: ("acdc_myocardium", "myocardium"),
    3: ("acdc_left_ventricle", "left_ventricle"),
}

VOLUME_FOLDERS = (
    "ACDC_training_volumes",
    "ACDC_testing_volumes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the HDF5 ACDC dataset into CPU-friendly, binary, "
            "128x128 UniverSeg tasks. For each patient, annotated frame, and "
            "cardiac structure, the slice with maximum foreground area is "
            "selected. The script is deterministic and uses subject-level "
            "60/20/20 splits."
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
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Square output size. Default: 128",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic subject split seed. Default: 42",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing ACDC processed files.",
    )
    return parser.parse_args()


def parse_patient_and_frame(path: Path) -> tuple[str, str]:
    match = re.fullmatch(
        r"(?i)(patient\d+)_(frame\d+)",
        path.stem,
    )
    if match is None:
        raise ValueError(
            "Unexpected ACDC volume filename. Expected "
            f"'patientNNN_frameNN.h5', received: {path.name}"
        )

    patient_id = match.group(1).lower()
    frame_id = match.group(2).lower()
    return patient_id, frame_id


def discover_volume_files(acdc_root: Path) -> list[Path]:
    files: list[Path] = []

    for folder_name in VOLUME_FOLDERS:
        folder = acdc_root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing ACDC folder: {folder}")

        folder_files = sorted(
            path for path in folder.glob("*.h5") if path.is_file()
        )
        if not folder_files:
            raise RuntimeError(f"No HDF5 files found in: {folder}")

        files.extend(folder_files)

    patients = Counter(parse_patient_and_frame(path)[0] for path in files)

    if len(files) != 300:
        raise RuntimeError(
            f"Expected 300 ACDC volume files, found {len(files)}."
        )

    if len(patients) != 150:
        raise RuntimeError(
            f"Expected 150 unique patients, found {len(patients)}."
        )

    unexpected_frame_counts = {
        patient_id: count
        for patient_id, count in patients.items()
        if count != 2
    }
    if unexpected_frame_counts:
        raise RuntimeError(
            "Expected two annotated frames per patient. Unexpected counts: "
            f"{unexpected_frame_counts}"
        )

    return sorted(files)


def make_subject_splits(
    subject_ids: Iterable[str],
    seed: int,
) -> dict[str, str]:
    ids = sorted(set(subject_ids))

    if len(ids) != 150:
        raise ValueError(
            f"Expected 150 subjects before splitting, found {len(ids)}."
        )

    rng = random.Random(seed)
    rng.shuffle(ids)

    n_support = int(round(len(ids) * 0.60))
    n_dev = int(round(len(ids) * 0.20))
    n_test = len(ids) - n_support - n_dev

    if (n_support, n_dev, n_test) != (90, 30, 30):
        raise RuntimeError(
            "Unexpected ACDC split sizes: "
            f"{n_support}/{n_dev}/{n_test}"
        )

    split_map: dict[str, str] = {}

    for subject_id in ids[:n_support]:
        split_map[subject_id] = "support"

    for subject_id in ids[n_support:n_support + n_dev]:
        split_map[subject_id] = "dev"

    for subject_id in ids[n_support + n_dev:]:
        split_map[subject_id] = "test"

    return split_map


def load_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with h5py.File(path, "r") as handle:
            if "image" not in handle:
                raise KeyError("Missing HDF5 dataset: image")
            if "label" not in handle:
                raise KeyError("Missing HDF5 dataset: label")

            image = np.asarray(handle["image"], dtype=np.float32)
            label = np.asarray(handle["label"], dtype=np.uint8)
    except Exception as exc:
        raise RuntimeError(f"Could not read ACDC volume: {path}") from exc

    if image.ndim != 3 or label.ndim != 3:
        raise ValueError(
            f"Expected 3D [slice, height, width] arrays in {path.name}; "
            f"received image={image.shape}, label={label.shape}."
        )

    if image.shape != label.shape:
        raise ValueError(
            f"Image-label shape mismatch in {path.name}: "
            f"{image.shape} versus {label.shape}."
        )

    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite image values in {path.name}.")

    image_minimum = float(image.min())
    image_maximum = float(image.max())
    if image_minimum < -1e-6 or image_maximum > 1.0 + 1e-6:
        raise ValueError(
            f"Expected normalized images in [0,1] in {path.name}; "
            f"found [{image_minimum}, {image_maximum}]."
        )

    unique_labels = set(int(value) for value in np.unique(label))
    if not unique_labels.issubset({0, 1, 2, 3}):
        raise ValueError(
            f"Unexpected labels in {path.name}: {sorted(unique_labels)}"
        )

    return image, label


def select_max_area_slice(
    label_volume: np.ndarray,
    target_value: int,
) -> tuple[int, int]:
    binary_volume = label_volume == target_value
    areas = binary_volume.reshape(binary_volume.shape[0], -1).sum(axis=1)

    slice_index = int(np.argmax(areas))
    maximum_area = int(areas[slice_index])

    if maximum_area <= 0:
        raise ValueError(
            f"Target label {target_value} is absent from the volume."
        )

    return slice_index, maximum_area


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return np.rint(clipped * 255.0).astype(np.uint8)


def resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    pil_image = Image.fromarray(image_to_uint8(image), mode="L")
    resized = pil_image.resize(
        (image_size, image_size),
        resample=RESAMPLE_BILINEAR,
    )
    return np.asarray(resized, dtype=np.uint8)


def resize_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    mask_uint8 = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    pil_mask = Image.fromarray(mask_uint8, mode="L")
    resized = pil_mask.resize(
        (image_size, image_size),
        resample=RESAMPLE_NEAREST,
    )
    return np.asarray(resized, dtype=np.uint8) > 0


def save_png(array: np.ndarray, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}\n"
            "Use --overwrite only when you intentionally want to regenerate "
            "the ACDC processed dataset."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def relative_to_processed(path: Path, processed_root: Path) -> str:
    return path.relative_to(processed_root).as_posix()


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
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
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def ensure_clean_output_locations(
    processed_root: Path,
    overwrite: bool,
) -> None:
    expected_locations = [
        processed_root / "images" / "acdc",
        processed_root / "valid_masks" / "acdc",
        processed_root / "masks" / "acdc_right_ventricle",
        processed_root / "masks" / "acdc_myocardium",
        processed_root / "masks" / "acdc_left_ventricle",
        processed_root / "manifest_acdc.csv",
        processed_root / "acdc_preprocessing_summary.json",
    ]

    existing = [path for path in expected_locations if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "ACDC processed outputs already exist:\n"
            f"{formatted}\n"
            "Review them first, or rerun with --overwrite to regenerate."
        )


def main() -> None:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    acdc_root = project_root / "data" / "raw" / args.acdc_folder
    processed_root = project_root / "data" / "processed"

    if not acdc_root.is_dir():
        raise FileNotFoundError(f"ACDC root not found: {acdc_root}")

    ensure_clean_output_locations(processed_root, args.overwrite)

    volume_files = discover_volume_files(acdc_root)
    patient_ids = [
        parse_patient_and_frame(path)[0]
        for path in volume_files
    ]
    split_map = make_subject_splits(patient_ids, args.seed)

    rows: list[dict[str, object]] = []
    split_subject_counts = Counter(split_map.values())
    split_row_counts: Counter[str] = Counter()
    task_row_counts: Counter[str] = Counter()
    source_folder_counts: Counter[str] = Counter()
    selected_slice_counts: Counter[int] = Counter()
    skipped_targets: list[dict[str, object]] = []

    written_image_paths: set[Path] = set()
    written_valid_paths: set[Path] = set()

    for file_index, volume_path in enumerate(volume_files, start=1):
        patient_id, frame_id = parse_patient_and_frame(volume_path)
        split = split_map[patient_id]
        source_folder = volume_path.parent.name

        print(
            f"[{file_index:03d}/{len(volume_files)}] "
            f"{patient_id} {frame_id} ({split})",
            flush=True,
        )

        image_volume, label_volume = load_volume(volume_path)
        source_folder_counts[source_folder] += 1

        for label_value, (
            task_id,
            target_label,
        ) in LABEL_TO_TASK.items():
            try:
                slice_index, original_foreground_pixels = (
                    select_max_area_slice(label_volume, label_value)
                )
            except ValueError as exc:
                skipped_targets.append(
                    {
                        "source_file": str(volume_path),
                        "patient_id": patient_id,
                        "frame_id": frame_id,
                        "task_id": task_id,
                        "reason": str(exc),
                    }
                )
                continue

            image_slice = image_volume[slice_index]
            binary_mask = label_volume[slice_index] == label_value

            processed_image = resize_image(
                image_slice,
                args.image_size,
            )
            processed_mask = resize_mask(
                binary_mask,
                args.image_size,
            )
            processed_valid = np.ones(
                (args.image_size, args.image_size),
                dtype=np.uint8,
            ) * 255

            base_id = (
                f"{patient_id}_{frame_id}_slice_{slice_index:02d}"
            )

            output_image = (
                processed_root
                / "images"
                / "acdc"
                / f"{base_id}.png"
            )
            output_valid = (
                processed_root
                / "valid_masks"
                / "acdc"
                / f"{base_id}.png"
            )
            output_mask = (
                processed_root
                / "masks"
                / task_id
                / f"{base_id}.png"
            )

            if output_image not in written_image_paths:
                save_png(
                    processed_image,
                    output_image,
                    overwrite=args.overwrite,
                )
                written_image_paths.add(output_image)

            if output_valid not in written_valid_paths:
                save_png(
                    processed_valid,
                    output_valid,
                    overwrite=args.overwrite,
                )
                written_valid_paths.add(output_valid)

            save_png(
                processed_mask.astype(np.uint8) * 255,
                output_mask,
                overwrite=args.overwrite,
            )

            sample_id = f"{task_id}:{base_id}"

            rows.append(
                {
                    "sample_id": sample_id,
                    "task_id": task_id,
                    "task_role": "train",
                    "dataset": "acdc",
                    "modality": "cine_mri",
                    "anatomy": "heart",
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
                    "subject_id": patient_id,
                    "split": split,
                    "has_foreground": int(processed_mask.any()),
                    "source_image_path": (
                        f"{volume_path}::image[{slice_index}]"
                    ),
                    "source_mask_paths": (
                        f"{volume_path}::label[{slice_index}]"
                        f"=={label_value};"
                        f"original_foreground_pixels="
                        f"{original_foreground_pixels}"
                    ),
                }
            )

            split_row_counts[split] += 1
            task_row_counts[task_id] += 1
            selected_slice_counts[slice_index] += 1

    if not rows:
        raise RuntimeError("No ACDC samples were produced.")

    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate sample IDs were produced.")

    for row in rows:
        if int(row["has_foreground"]) != 1:
            raise RuntimeError(
                f"Unexpected empty processed mask: {row['sample_id']}"
            )

    manifest_path = processed_root / "manifest_acdc.csv"
    write_manifest(manifest_path, rows)

    summary = {
        "status": "completed",
        "project_role": "training dataset in reduced reproduction",
        "source": "preprocessed ACDC HDF5 Kaggle copy",
        "input_root": str(acdc_root),
        "input_volume_files": len(volume_files),
        "input_patients": len(set(patient_ids)),
        "source_folder_counts": dict(sorted(source_folder_counts.items())),
        "ignored_duplicate_folder": (
            "ACDC_training_slices was intentionally ignored because it "
            "duplicates slices from ACDC_training_volumes."
        ),
        "hdf5_image_key": "image",
        "hdf5_label_key": "label",
        "ignored_hdf5_key": "scribble",
        "label_mapping": {
            "0": "background",
            "1": "right_ventricle",
            "2": "myocardium",
            "3": "left_ventricle",
        },
        "selection_policy": (
            "For each patient, annotated frame, and target label, select "
            "the axial slice containing the maximum number of target pixels."
        ),
        "output_image_size": [args.image_size, args.image_size],
        "normalization": (
            "Input HDF5 images were already float32 in [0,1]; they were "
            "clipped to [0,1], converted to uint8 PNG, and later reload "
            "to the same normalized range."
        ),
        "mask_interpolation": "nearest-neighbor",
        "image_interpolation": "bilinear",
        "subject_split_seed": args.seed,
        "subject_split_policy": "deterministic 60/20/20",
        "subject_split_counts": dict(sorted(split_subject_counts.items())),
        "manifest_row_counts_by_split": dict(
            sorted(split_row_counts.items())
        ),
        "manifest_row_counts_by_task": dict(
            sorted(task_row_counts.items())
        ),
        "manifest_rows": len(rows),
        "unique_processed_images": len(written_image_paths),
        "selected_slice_index_histogram": {
            str(index): count
            for index, count in sorted(selected_slice_counts.items())
        },
        "skipped_target_count": len(skipped_targets),
        "skipped_targets": skipped_targets,
        "scientific_scope": (
            "This ACDC adapter supports a CPU-friendly methodological "
            "reproduction. ACDC is used for training here, unlike the "
            "original UniverSeg paper where ACDC was held out."
        ),
    }

    summary_path = (
        processed_root / "acdc_preprocessing_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== ACDC PREPROCESSING COMPLETE ===")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    print(f"Rows:     {len(rows)}")
    print(f"Images:   {len(written_image_paths)}")
    print(
        "Subjects: ",
        dict(sorted(split_subject_counts.items())),
    )
    print("Tasks:    ", dict(sorted(task_row_counts.items())))

    if skipped_targets:
        print(
            f"Warning: {len(skipped_targets)} target-volume pairs "
            "contained no foreground and were skipped."
        )


if __name__ == "__main__":
    main()
