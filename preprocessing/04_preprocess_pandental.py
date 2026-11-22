from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    from scipy import ndimage
except ImportError as exc:
    raise ImportError(
        "This script requires SciPy for connected-component and hole-filling "
        "operations. Install it with:\n"
        "python -m pip install scipy"
    ) from exc


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANDENTAL_FOLDER = "Panoramic_Dental_Xrays"
DEFAULT_IMAGE_SIZE = 128
DEFAULT_SEED = 42

TASK_ID = "pandental_mandible"
SECOND_ANNOTATOR_TASK_ID = "pandental_mandible_dentist2"


KNOWN_SHAPE_MISMATCH_IDS = {"109"}

RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the PanDental masked-radiograph annotations into binary "
            "mandible masks for held-out UniverSeg evaluation. Segmentation1 "
            "is used as the primary ground truth; Segmentation2 is preserved "
            "for inter-annotator quality analysis."
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
            "Folder inside MiniMedical. "
            "Default: Panoramic_Dental_Xrays"
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
        help="Deterministic subject split seed. Default: 42",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing PanDental processed outputs.",
    )
    return parser.parse_args()


def numeric_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10**9, value


def discover_ids(folder: Path) -> set[str]:
    return {
        path.stem
        for path in folder.glob("*.png")
        if path.is_file()
    }


def load_gray(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)
    except Exception as exc:
        raise RuntimeError(f"Could not read image: {path}") from exc


def most_frequent_intensity(image: np.ndarray) -> tuple[int, int]:
    counts = np.bincount(
        np.asarray(image, dtype=np.uint8).ravel(),
        minlength=256,
    )
    mode = int(np.argmax(counts))
    count = int(counts[mode])
    return mode, count


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labels, component_count = ndimage.label(mask)

    if component_count == 0:
        raise ValueError("No foreground component was detected.")

    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    largest_label = int(np.argmax(component_sizes))

    return labels == largest_label


def extract_binary_mask(
    segmented_radiograph: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """
    Recover a binary mandible mask from a masked radiograph.

    Encoding observed in the downloaded dataset:
      - pixels outside the dentist annotation are replaced by one uniform
        dark background intensity;
      - pixels inside the annotation retain radiographic intensities.

    The modal intensity is therefore treated as the masked background.
    Small disconnected artifacts are removed, narrow internal gaps are
    closed, and enclosed holes are filled.
    """
    image = np.asarray(segmented_radiograph, dtype=np.uint8)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale annotation, got {image.shape}."
        )

    background_value, background_count = most_frequent_intensity(image)
    background_fraction = background_count / image.size
    minimum = int(image.min())
    fifth_percentile = float(np.percentile(image, 5))

    if background_fraction < 0.20:
        raise ValueError(
            "The modal intensity occupies less than 20% of the image, so "
            "the expected uniform masked background was not detected. "
            f"mode={background_value}, fraction={background_fraction:.4f}"
        )

    if background_value > fifth_percentile + 2:
        raise ValueError(
            "The modal intensity is not in the dark tail of the annotation. "
            f"mode={background_value}, p5={fifth_percentile:.2f}"
        )

    raw_mask = image > background_value

    # Remove isolated pixels and join narrow discontinuities at the boundary.
    structure = np.ones((5, 5), dtype=bool)
    cleaned = ndimage.binary_opening(
        raw_mask,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    )
    cleaned = ndimage.binary_closing(
        cleaned,
        structure=structure,
        iterations=2,
    )

    cleaned = keep_largest_component(cleaned)


    cleaned = ndimage.binary_fill_holes(cleaned)

    foreground_fraction = float(cleaned.mean())

    if not 0.10 <= foreground_fraction <= 0.70:
        raise ValueError(
            "Extracted mask has an implausible foreground fraction: "
            f"{foreground_fraction:.4f}"
        )

    diagnostics: dict[str, float | int] = {
        "background_value": background_value,
        "minimum_value": minimum,
        "background_fraction": float(background_fraction),
        "foreground_fraction_before_cleanup": float(raw_mask.mean()),
        "foreground_fraction_after_cleanup": foreground_fraction,
    }
    return cleaned.astype(bool), diagnostics


def dice_score(first: np.ndarray, second: np.ndarray) -> float:
    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)

    denominator = int(first_bool.sum()) + int(second_bool.sum())
    if denominator == 0:
        return 1.0

    intersection = int(np.logical_and(first_bool, second_bool).sum())
    return (2.0 * intersection) / denominator


def make_subject_splits(
    subject_ids: Iterable[str],
    seed: int,
) -> dict[str, str]:
    ids = sorted(set(subject_ids), key=numeric_sort_key)

    if len(ids) != 115:
        raise ValueError(
            "Expected 115 usable subjects after excluding patient 109; "
            f"found {len(ids)}."
        )

    rng = random.Random(seed)
    rng.shuffle(ids)

    n_support = int(round(len(ids) * 0.60))
    n_dev = int(round(len(ids) * 0.20))
    n_test = len(ids) - n_support - n_dev

    if (n_support, n_dev, n_test) != (69, 23, 23):
        raise RuntimeError(
            "Unexpected split sizes: "
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


def resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    pil_image = Image.fromarray(
        np.asarray(image, dtype=np.uint8),
        mode="L",
    )
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


def save_png(
    array: np.ndarray,
    path: Path,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}\n"
            "Use --overwrite only when intentionally regenerating outputs."
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


def write_inter_annotator_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    columns = [
        "subject_id",
        "split",
        "original_height",
        "original_width",
        "dentist1_background_value",
        "dentist2_background_value",
        "dentist1_foreground_fraction",
        "dentist2_foreground_fraction",
        "inter_annotator_dice_original_resolution",
        "inter_annotator_dice_128",
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


def ensure_clean_outputs(
    processed_root: Path,
    overwrite: bool,
) -> None:
    locations = [
        processed_root / "images" / "pandental",
        processed_root / "valid_masks" / "pandental",
        processed_root / "masks" / TASK_ID,
        processed_root / "masks" / SECOND_ANNOTATOR_TASK_ID,
        processed_root / "manifest_pandental.csv",
        processed_root / "pandental_inter_annotator.csv",
        processed_root / "pandental_preprocessing_summary.json",
    ]

    existing = [path for path in locations if path.exists()]
    if existing and not overwrite:
        details = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "PanDental processed outputs already exist:\n"
            f"{details}\n"
            "Review them or rerun with --overwrite."
        )


def main() -> None:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    dental_root = (
        project_root
        / "data"
        / "raw"
        / args.pandental_folder
    )
    processed_root = project_root / "data" / "processed"

    images_dir = dental_root / "Images"
    segmentation1_dir = dental_root / "Segmentation1"
    segmentation2_dir = dental_root / "Segmentation2"

    for folder in (
        images_dir,
        segmentation1_dir,
        segmentation2_dir,
    ):
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing required folder: {folder}")

    ensure_clean_outputs(processed_root, args.overwrite)

    image_ids = discover_ids(images_dir)
    dentist1_ids = discover_ids(segmentation1_dir)
    dentist2_ids = discover_ids(segmentation2_dir)

    common_ids = image_ids & dentist1_ids & dentist2_ids

    if len(image_ids) != 116:
        raise RuntimeError(f"Expected 116 images, found {len(image_ids)}.")
    if len(dentist1_ids) != 116:
        raise RuntimeError(
            f"Expected 116 Segmentation1 files, found {len(dentist1_ids)}."
        )
    if len(dentist2_ids) != 116:
        raise RuntimeError(
            f"Expected 116 Segmentation2 files, found {len(dentist2_ids)}."
        )
    if len(common_ids) != 116:
        raise RuntimeError(
            "Images, Segmentation1, and Segmentation2 do not share "
            "the same 116 numeric filenames."
        )

    usable_ids = sorted(
        common_ids - KNOWN_SHAPE_MISMATCH_IDS,
        key=numeric_sort_key,
    )
    split_map = make_subject_splits(usable_ids, args.seed)

    manifest_rows: list[dict[str, object]] = []
    inter_annotator_rows: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()

    original_resolution_dice_values: list[float] = []
    resized_dice_values: list[float] = []
    dentist1_background_values: Counter[int] = Counter()
    dentist2_background_values: Counter[int] = Counter()

    for index, subject_id in enumerate(usable_ids, start=1):
        split = split_map[subject_id]
        print(
            f"[{index:03d}/{len(usable_ids)}] "
            f"patient {subject_id} ({split})",
            flush=True,
        )

        image_path = images_dir / f"{subject_id}.png"
        dentist1_path = segmentation1_dir / f"{subject_id}.png"
        dentist2_path = segmentation2_dir / f"{subject_id}.png"

        image = load_gray(image_path)
        dentist1_segmented = load_gray(dentist1_path)
        dentist2_segmented = load_gray(dentist2_path)

        if not (
            image.shape
            == dentist1_segmented.shape
            == dentist2_segmented.shape
        ):
            raise ValueError(
                f"Unexpected shape mismatch for patient {subject_id}: "
                f"image={image.shape}, "
                f"dentist1={dentist1_segmented.shape}, "
                f"dentist2={dentist2_segmented.shape}"
            )

        dentist1_mask, dentist1_diagnostics = extract_binary_mask(
            dentist1_segmented
        )
        dentist2_mask, dentist2_diagnostics = extract_binary_mask(
            dentist2_segmented
        )

        original_dice = dice_score(
            dentist1_mask,
            dentist2_mask,
        )

        processed_image = resize_image(image, args.image_size)
        processed_mask1 = resize_mask(
            dentist1_mask,
            args.image_size,
        )
        processed_mask2 = resize_mask(
            dentist2_mask,
            args.image_size,
        )
        processed_valid = np.ones(
            (args.image_size, args.image_size),
            dtype=np.uint8,
        ) * 255

        resized_dice = dice_score(
            processed_mask1,
            processed_mask2,
        )

        output_image = (
            processed_root
            / "images"
            / "pandental"
            / f"pandental_{int(subject_id):03d}.png"
        )
        output_valid = (
            processed_root
            / "valid_masks"
            / "pandental"
            / f"pandental_{int(subject_id):03d}.png"
        )
        output_mask1 = (
            processed_root
            / "masks"
            / TASK_ID
            / f"pandental_{int(subject_id):03d}.png"
        )
        output_mask2 = (
            processed_root
            / "masks"
            / SECOND_ANNOTATOR_TASK_ID
            / f"pandental_{int(subject_id):03d}.png"
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
        save_png(
            processed_mask1.astype(np.uint8) * 255,
            output_mask1,
            overwrite=args.overwrite,
        )
        save_png(
            processed_mask2.astype(np.uint8) * 255,
            output_mask2,
            overwrite=args.overwrite,
        )

        sample_id = (
            f"{TASK_ID}:pandental_{int(subject_id):03d}"
        )

        manifest_rows.append(
            {
                "sample_id": sample_id,
                "task_id": TASK_ID,
                "task_role": "heldout",
                "dataset": "pandental",
                "modality": "panoramic_dental_xray",
                "anatomy": "mandible",
                "target_label": "mandible",
                "image_path": relative_to_processed(
                    output_image,
                    processed_root,
                ),
                "mask_path": relative_to_processed(
                    output_mask1,
                    processed_root,
                ),
                "valid_mask_path": relative_to_processed(
                    output_valid,
                    processed_root,
                ),
                "subject_id": f"pandental_{int(subject_id):03d}",
                "split": split,
                "has_foreground": int(processed_mask1.any()),
                "source_image_path": str(image_path),
                "source_mask_paths": (
                    f"primary={dentist1_path};"
                    f"secondary={dentist2_path};"
                    "primary_rule=Segmentation1;"
                    "secondary_reserved_for_inter_annotator_audit"
                ),
            }
        )

        inter_annotator_rows.append(
            {
                "subject_id": f"pandental_{int(subject_id):03d}",
                "split": split,
                "original_height": image.shape[0],
                "original_width": image.shape[1],
                "dentist1_background_value": (
                    dentist1_diagnostics["background_value"]
                ),
                "dentist2_background_value": (
                    dentist2_diagnostics["background_value"]
                ),
                "dentist1_foreground_fraction": (
                    dentist1_diagnostics[
                        "foreground_fraction_after_cleanup"
                    ]
                ),
                "dentist2_foreground_fraction": (
                    dentist2_diagnostics[
                        "foreground_fraction_after_cleanup"
                    ]
                ),
                "inter_annotator_dice_original_resolution": (
                    original_dice
                ),
                "inter_annotator_dice_128": resized_dice,
            }
        )

        split_counts[split] += 1
        original_resolution_dice_values.append(original_dice)
        resized_dice_values.append(resized_dice)
        dentist1_background_values[
            int(dentist1_diagnostics["background_value"])
        ] += 1
        dentist2_background_values[
            int(dentist2_diagnostics["background_value"])
        ] += 1

    if len(manifest_rows) != 115:
        raise RuntimeError(
            f"Expected 115 manifest rows, produced {len(manifest_rows)}."
        )

    sample_ids = [str(row["sample_id"]) for row in manifest_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate PanDental sample IDs were produced.")

    if any(int(row["has_foreground"]) != 1 for row in manifest_rows):
        raise RuntimeError("At least one processed primary mask is empty.")

    manifest_path = processed_root / "manifest_pandental.csv"
    inter_annotator_path = (
        processed_root / "pandental_inter_annotator.csv"
    )

    write_manifest(manifest_path, manifest_rows)
    write_inter_annotator_csv(
        inter_annotator_path,
        inter_annotator_rows,
    )

    original_dice_array = np.asarray(
        original_resolution_dice_values,
        dtype=np.float64,
    )
    resized_dice_array = np.asarray(
        resized_dice_values,
        dtype=np.float64,
    )

    summary = {
        "status": "completed",
        "project_role": "held-out unseen-domain evaluation dataset",
        "source_root": str(dental_root),
        "original_subject_count": 116,
        "usable_subject_count": 115,
        "excluded_subjects": {
            "109": (
                "Original image is 1310x2950 while both dentist annotations "
                "are 1320x2950. The undocumented 10-row alignment was not "
                "guessed."
            )
        },
        "pairing_rule": (
            "Images/N.png pairs with Segmentation1/N.png and "
            "Segmentation2/N.png."
        ),
        "annotation_encoding": (
            "Each Segmentation file is a masked radiograph: the annotated "
            "mandible retains radiographic intensities and the outside "
            "region is replaced by a uniform dark modal intensity."
        ),
        "binary_mask_extraction": (
            "Foreground = pixels above the annotation's modal dark "
            "background, followed by small morphological opening/closing, "
            "largest-component retention, and enclosed-hole filling."
        ),
        "primary_ground_truth": (
            "Segmentation1. No official fusion rule for the two dentists "
            "was supplied, so annotations were not arbitrarily merged."
        ),
        "secondary_annotation": (
            "Segmentation2 is preserved in "
            f"masks/{SECOND_ANNOTATOR_TASK_ID} and used for inter-annotator "
            "Dice analysis."
        ),
        "task_id": TASK_ID,
        "task_role": "heldout",
        "output_image_size": [args.image_size, args.image_size],
        "image_interpolation": "bilinear",
        "mask_interpolation": "nearest-neighbor",
        "subject_split_seed": args.seed,
        "subject_split_policy": "deterministic 60/20/20",
        "subject_split_counts": dict(sorted(split_counts.items())),
        "manifest_rows": len(manifest_rows),
        "dentist1_background_value_counts": {
            str(key): value
            for key, value in sorted(
                dentist1_background_values.items()
            )
        },
        "dentist2_background_value_counts": {
            str(key): value
            for key, value in sorted(
                dentist2_background_values.items()
            )
        },
        "inter_annotator_dice_original_resolution": {
            "minimum": float(original_dice_array.min()),
            "median": float(np.median(original_dice_array)),
            "mean": float(original_dice_array.mean()),
            "maximum": float(original_dice_array.max()),
        },
        "inter_annotator_dice_128": {
            "minimum": float(resized_dice_array.min()),
            "median": float(np.median(resized_dice_array)),
            "mean": float(resized_dice_array.mean()),
            "maximum": float(resized_dice_array.max()),
        },
        "scientific_scope": (
            "This is a transparent local preprocessing rule for the "
            "downloaded masked-radiograph annotations. The public UniverSeg "
            "repository does not provide its PanDental preprocessing code."
        ),
    }

    summary_path = (
        processed_root / "pandental_preprocessing_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== PANDENTAL PREPROCESSING COMPLETE ===")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    print(f"Audit:    {inter_annotator_path}")
    print(f"Rows:     {len(manifest_rows)}")
    print(f"Splits:   {dict(sorted(split_counts.items()))}")
    print(
        "Mean inter-annotator Dice at original resolution: "
        f"{original_dice_array.mean():.4f}"
    )
    print(
        "Mean inter-annotator Dice at 128x128: "
        f"{resized_dice_array.mean():.4f}"
    )


if __name__ == "__main__":
    main()
