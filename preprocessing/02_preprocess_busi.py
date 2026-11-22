import argparse
from pathlib import Path
import csv
import json
import random
import re
import shutil

import numpy as np
from PIL import Image

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_SIZE = 128
SEED = 42

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".gif", ".ppm", ".pgm",
}

RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST


def is_image_file(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def normalize_key(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def read_gray(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def normalize_to_uint8(image):
    image = np.asarray(image, dtype=np.float32)

    if image.size == 0:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.uint8)

    minimum = float(image[finite].min())
    maximum = float(image[finite].max())

    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = (image - minimum) / (maximum - minimum)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def resize_gray_image(image):
    image = normalize_to_uint8(image)
    pil = Image.fromarray(image, mode="L")
    return np.asarray(
        pil.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            resample=RESAMPLE_BILINEAR,
        )
    )


def resize_binary_mask(mask):
    mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    pil = Image.fromarray(mask, mode="L")
    resized = np.asarray(
        pil.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            resample=RESAMPLE_NEAREST,
        )
    )
    return resized > 0


def save_png(array, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def make_splits(subject_ids, seed):
    ids = list(dict.fromkeys(str(item) for item in subject_ids))

    if len(ids) < 3:
        raise ValueError("At least three subjects are required.")

    rng = random.Random(seed)
    rng.shuffle(ids)

    total = len(ids)
    n_support = max(1, round(total * 0.60))
    n_dev = max(1, round(total * 0.20))

    if n_support + n_dev >= total:
        n_support = total - 2
        n_dev = 1

    split_map = {}

    for item in ids[:n_support]:
        split_map[item] = "support"

    for item in ids[n_support:n_support + n_dev]:
        split_map[item] = "dev"

    for item in ids[n_support + n_dev:]:
        split_map[item] = "test"

    return split_map


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

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

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def relative_processed(path, processed_root):
    return Path(path).relative_to(processed_root).as_posix()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess the BUSI breast ultrasound dataset."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Repository root. Defaults to this repository directory.",
    )
    return parser.parse_args()


def mask_base_name(path):
    return re.sub(
        r"_mask(?:_\d+)?$",
        "",
        path.stem,
        flags=re.IGNORECASE,
    )


def main():
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    data_root = project_root / "data" / "raw"
    processed_root = project_root / "data" / "processed"
    busi_root = data_root / "Dataset_BUSI_with_GT"
    output_manifest = processed_root / "manifest_busi.csv"

    print("Starting BUSI preprocessing...", flush=True)

    rows = []

    class_subjects = {
        "benign": [],
        "malignant": [],
        "normal": [],
    }

    records = []

    for class_name in ("benign", "malignant", "normal"):
        class_dir = busi_root / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(class_dir)

        files = sorted(
            path for path in class_dir.iterdir()
            if is_image_file(path)
        )

        images = [
            path for path in files
            if "_mask" not in path.stem.lower()
        ]

        masks = [
            path for path in files
            if "_mask" in path.stem.lower()
        ]

        masks_by_key = {}

        for mask_path in masks:
            key = normalize_key(mask_base_name(mask_path))
            masks_by_key.setdefault(key, []).append(mask_path)

        for image_path in images:
            key = normalize_key(image_path.stem)
            subject_id = f"busi_{class_name}_{key}"

            records.append(
                (
                    image_path,
                    masks_by_key.get(key, []),
                    class_name,
                    subject_id,
                )
            )

            class_subjects[class_name].append(subject_id)

    split_map = {}

    for offset, class_name in enumerate(
        ("benign", "malignant", "normal")
    ):
        split_map.update(
            make_splits(
                class_subjects[class_name],
                SEED + 100 + offset,
            )
        )

    for index, (
        image_path,
        mask_paths,
        class_name,
        subject_id,
    ) in enumerate(records, start=1):
        print(
            f"BUSI {index}/{len(records)}: {image_path.name}",
            flush=True,
        )

        image = read_gray(image_path)

        if mask_paths:
            lesion_mask = np.zeros_like(
                read_gray(mask_paths[0]),
                dtype=bool,
            )

            for mask_path in mask_paths:
                current = read_gray(mask_path) > 0

                if current.shape != lesion_mask.shape:
                    current = np.asarray(
                        Image.fromarray(
                            current.astype(np.uint8) * 255
                        ).resize(
                            (
                                lesion_mask.shape[1],
                                lesion_mask.shape[0],
                            ),
                            resample=RESAMPLE_NEAREST,
                        )
                    ) > 0

                lesion_mask |= current
        else:
            lesion_mask = np.zeros_like(image, dtype=bool)

        valid = np.ones_like(image, dtype=bool)

        processed_image = resize_gray_image(image)
        processed_valid = resize_binary_mask(valid)

        output_image = (
            processed_root / "images" / "busi"
            / f"{subject_id}.png"
        )

        output_valid = (
            processed_root / "valid_masks" / "busi"
            / f"{subject_id}.png"
        )

        save_png(processed_image, output_image)
        save_png(processed_valid.astype(np.uint8) * 255, output_valid)

        tasks = [
            (
                "busi_all_lesions",
                "lesion",
                lesion_mask,
            )
        ]

        if class_name in {"benign", "normal"}:
            tasks.append(
                (
                    "busi_benign_lesion",
                    "benign_lesion",
                    lesion_mask
                    if class_name == "benign"
                    else np.zeros_like(lesion_mask),
                )
            )

        if class_name in {"malignant", "normal"}:
            tasks.append(
                (
                    "busi_malignant_lesion",
                    "malignant_lesion",
                    lesion_mask
                    if class_name == "malignant"
                    else np.zeros_like(lesion_mask),
                )
            )

        for task_id, target_label, task_mask in tasks:
            processed_mask = resize_binary_mask(task_mask)

            output_mask = (
                processed_root / "masks" / task_id
                / f"{subject_id}.png"
            )

            save_png(
                processed_mask.astype(np.uint8) * 255,
                output_mask,
            )

            rows.append(
                {
                    "sample_id": f"{task_id}:{subject_id}",
                    "task_id": task_id,
                    "task_role": "train",
                    "dataset": "busi",
                    "modality": "ultrasound",
                    "anatomy": "breast",
                    "target_label": target_label,
                    "image_path": relative_processed(output_image, processed_root),
                    "mask_path": relative_processed(output_mask, processed_root),
                    "valid_mask_path": relative_processed(output_valid, processed_root),
                    "subject_id": subject_id,
                    "split": split_map[subject_id],
                    "has_foreground": int(processed_mask.any()),
                    "source_image_path": str(image_path),
                    "source_mask_paths": "|".join(
                        str(path) for path in mask_paths
                    ),
                }
            )

    write_manifest(output_manifest, rows)

    print("\nBUSI completed.")
    print("Rows:", len(rows))
    print("Manifest:", output_manifest)


if __name__ == "__main__":
    main()
