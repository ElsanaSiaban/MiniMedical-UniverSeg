from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANDENTAL_FOLDER = "Panoramic_Dental_Xrays"
DEFAULT_EXAMPLE_IDS = ("1", "10", "50", "100", "109", "116")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PanDental Images, Segmentation1, and Segmentation2 "
            "pixel-by-pixel. This is a read-only diagnostic step used to "
            "determine how the two dentist annotations are encoded."
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
        "--example-ids",
        nargs="*",
        default=list(DEFAULT_EXAMPLE_IDS),
        help=(
            "Patient IDs used for visual contact sheets. "
            "Default: 1 10 50 100 109 116"
        ),
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=640,
        help="Width of each contact-sheet panel. Default: 640",
    )
    return parser.parse_args()


def numeric_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return math.inf, value


def load_gray(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)
    except Exception as exc:
        raise RuntimeError(f"Could not read: {path}") from exc


def array_statistics(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    return {
        "shape": list(values.shape),
        "minimum": int(values.min()),
        "maximum": int(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "zero_fraction": float(np.mean(values == 0)),
        "near_zero_fraction_le_5": float(np.mean(values <= 5)),
    }


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    if np.std(a) == 0 or np.std(b) == 0:
        return None

    return float(np.corrcoef(a, b)[0, 1])


def comparison_statistics(
    original: np.ndarray,
    segmentation: np.ndarray,
) -> dict[str, Any]:
    if original.shape != segmentation.shape:
        return {
            "shape_match": False,
            "original_shape": list(original.shape),
            "segmentation_shape": list(segmentation.shape),
        }

    difference = np.abs(
        original.astype(np.int16) - segmentation.astype(np.int16)
    )

    nonzero_original = original > 0
    ratio = np.divide(
        segmentation.astype(np.float64),
        original.astype(np.float64),
        out=np.full(original.shape, np.nan, dtype=np.float64),
        where=nonzero_original,
    )
    finite_ratio = ratio[np.isfinite(ratio)]

    result: dict[str, Any] = {
        "shape_match": True,
        "exact_match_fraction": float(np.mean(difference == 0)),
        "within_1_fraction": float(np.mean(difference <= 1)),
        "within_2_fraction": float(np.mean(difference <= 2)),
        "within_5_fraction": float(np.mean(difference <= 5)),
        "within_10_fraction": float(np.mean(difference <= 10)),
        "difference_mean": float(difference.mean()),
        "difference_median": float(np.median(difference)),
        "difference_maximum": int(difference.max()),
        "pearson_correlation": safe_correlation(
            original,
            segmentation,
        ),
        "segmentation_zero_where_original_nonzero_fraction": float(
            np.mean((segmentation == 0) & nonzero_original)
        ),
    }

    if finite_ratio.size:
        result["segmentation_to_original_ratio_median"] = float(
            np.median(finite_ratio)
        )
        result["segmentation_to_original_ratio_p10"] = float(
            np.percentile(finite_ratio, 10)
        )
        result["segmentation_to_original_ratio_p90"] = float(
            np.percentile(finite_ratio, 90)
        )

    return result


def compare_two_segmentations(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, Any]:
    if first.shape != second.shape:
        return {
            "shape_match": False,
            "segmentation1_shape": list(first.shape),
            "segmentation2_shape": list(second.shape),
        }

    difference = np.abs(
        first.astype(np.int16) - second.astype(np.int16)
    )

    return {
        "shape_match": True,
        "exact_match_fraction": float(np.mean(difference == 0)),
        "within_5_fraction": float(np.mean(difference <= 5)),
        "difference_mean": float(difference.mean()),
        "difference_median": float(np.median(difference)),
        "difference_maximum": int(difference.max()),
        "pearson_correlation": safe_correlation(first, second),
    }


def discover_ids(folder: Path) -> set[str]:
    return {
        path.stem
        for path in folder.glob("*.png")
        if path.is_file()
    }


def normalize_for_display(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    minimum = float(values.min())
    maximum = float(values.max())

    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.uint8)

    normalized = (values - minimum) / (maximum - minimum)
    return np.clip(np.rint(normalized * 255.0), 0, 255).astype(
        np.uint8
    )


def resize_for_panel(
    array: np.ndarray,
    panel_width: int,
    resample: int,
) -> Image.Image:
    image = Image.fromarray(
        np.asarray(array, dtype=np.uint8),
        mode="L",
    )

    scale = panel_width / image.width
    height = max(1, int(round(image.height * scale)))

    return image.resize(
        (panel_width, height),
        resample=resample,
    )


def make_title_band(
    width: int,
    title: str,
    height: int = 42,
) -> Image.Image:
    band = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(band)
    font = ImageFont.load_default()
    draw.text((10, 14), title, fill=0, font=font)
    return band


def make_contact_sheet(
    patient_id: str,
    original: np.ndarray,
    segmentation1: np.ndarray,
    segmentation2: np.ndarray,
    output_path: Path,
    panel_width: int,
) -> None:
    # Resize annotations only for visual comparison when source dimensions
    # differ. This diagnostic resizing is never used to create final masks.
    original_pil = Image.fromarray(original, mode="L")

    seg1_visual = Image.fromarray(segmentation1, mode="L").resize(
        original_pil.size,
        resample=Image.Resampling.BILINEAR,
    )
    seg2_visual = Image.fromarray(segmentation2, mode="L").resize(
        original_pil.size,
        resample=Image.Resampling.BILINEAR,
    )

    seg1_array = np.asarray(seg1_visual, dtype=np.uint8)
    seg2_array = np.asarray(seg2_visual, dtype=np.uint8)

    diff1 = np.abs(
        original.astype(np.int16) - seg1_array.astype(np.int16)
    ).astype(np.uint8)
    diff2 = np.abs(
        original.astype(np.int16) - seg2_array.astype(np.int16)
    ).astype(np.uint8)
    inter_annotator_diff = np.abs(
        seg1_array.astype(np.int16) - seg2_array.astype(np.int16)
    ).astype(np.uint8)

    panels = [
        ("Original", original),
        ("Segmentation1", segmentation1),
        ("Segmentation2", segmentation2),
        ("|Original - Seg1|", normalize_for_display(diff1)),
        ("|Original - Seg2|", normalize_for_display(diff2)),
        (
            "|Seg1 - Seg2|",
            normalize_for_display(inter_annotator_diff),
        ),
    ]

    rendered: list[Image.Image] = []
    for title, array in panels:
        panel = resize_for_panel(
            normalize_for_display(array),
            panel_width=panel_width,
            resample=Image.Resampling.BILINEAR,
        )
        title_band = make_title_band(panel.width, title)
        combined = Image.new(
            "L",
            (panel.width, title_band.height + panel.height),
            color=255,
        )
        combined.paste(title_band, (0, 0))
        combined.paste(panel, (0, title_band.height))
        rendered.append(combined)

    maximum_height = max(panel.height for panel in rendered)
    total_width = sum(panel.width for panel in rendered)

    sheet = Image.new(
        "L",
        (total_width, maximum_height + 60),
        color=255,
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (10, 16),
        (
            f"PanDental patient {patient_id} — diagnostic only. "
            "Any resized annotation is for visualization, not preprocessing."
        ),
        fill=0,
        font=ImageFont.load_default(),
    )

    x = 0
    for panel in rendered:
        sheet.paste(panel, (x, 60))
        x += panel.width

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    dental_root = (
        project_root
        / "data"
        / "raw"
        / args.pandental_folder
    )

    images_dir = dental_root / "Images"
    segmentation1_dir = dental_root / "Segmentation1"
    segmentation2_dir = dental_root / "Segmentation2"

    for folder in (
        images_dir,
        segmentation1_dir,
        segmentation2_dir,
    ):
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing expected folder: {folder}")

    image_ids = discover_ids(images_dir)
    segmentation1_ids = discover_ids(segmentation1_dir)
    segmentation2_ids = discover_ids(segmentation2_dir)

    common_ids = (
        image_ids
        & segmentation1_ids
        & segmentation2_ids
    )

    if len(image_ids) != 116:
        raise RuntimeError(
            f"Expected 116 image IDs, found {len(image_ids)}."
        )
    if len(segmentation1_ids) != 116:
        raise RuntimeError(
            "Expected 116 Segmentation1 IDs, found "
            f"{len(segmentation1_ids)}."
        )
    if len(segmentation2_ids) != 116:
        raise RuntimeError(
            "Expected 116 Segmentation2 IDs, found "
            f"{len(segmentation2_ids)}."
        )
    if len(common_ids) != 116:
        raise RuntimeError(
            "The three folders do not share exactly the same 116 stems."
        )

    output_root = project_root / "outputs"
    visual_root = output_root / "pandental_annotation_diagnostics"

    records: list[dict[str, Any]] = []
    shape_pattern_counts: Counter[str] = Counter()
    mismatch_ids: list[str] = []

    sorted_ids = sorted(common_ids, key=numeric_sort_key)

    for index, patient_id in enumerate(sorted_ids, start=1):
        print(
            f"[{index:03d}/{len(sorted_ids)}] Comparing {patient_id}.png",
            flush=True,
        )

        image_path = images_dir / f"{patient_id}.png"
        segmentation1_path = (
            segmentation1_dir / f"{patient_id}.png"
        )
        segmentation2_path = (
            segmentation2_dir / f"{patient_id}.png"
        )

        original = load_gray(image_path)
        segmentation1 = load_gray(segmentation1_path)
        segmentation2 = load_gray(segmentation2_path)

        shape_pattern = (
            f"image={original.shape};"
            f"seg1={segmentation1.shape};"
            f"seg2={segmentation2.shape}"
        )
        shape_pattern_counts[shape_pattern] += 1

        all_shapes_match = (
            original.shape
            == segmentation1.shape
            == segmentation2.shape
        )
        if not all_shapes_match:
            mismatch_ids.append(patient_id)

        records.append(
            {
                "patient_id": patient_id,
                "all_shapes_match": all_shapes_match,
                "image": array_statistics(original),
                "segmentation1": array_statistics(segmentation1),
                "segmentation2": array_statistics(segmentation2),
                "image_vs_segmentation1": comparison_statistics(
                    original,
                    segmentation1,
                ),
                "image_vs_segmentation2": comparison_statistics(
                    original,
                    segmentation2,
                ),
                "segmentation1_vs_segmentation2": (
                    compare_two_segmentations(
                        segmentation1,
                        segmentation2,
                    )
                ),
            }
        )

        if patient_id in set(args.example_ids):
            output_path = (
                visual_root
                / f"pandental_{patient_id}_comparison.png"
            )
            make_contact_sheet(
                patient_id=patient_id,
                original=original,
                segmentation1=segmentation1,
                segmentation2=segmentation2,
                output_path=output_path,
                panel_width=args.panel_width,
            )

    same_shape_records = [
        record
        for record in records
        if record["all_shapes_match"]
    ]

    def aggregate_metric(
        section: str,
        metric: str,
    ) -> dict[str, float] | None:
        values = [
            record[section].get(metric)
            for record in same_shape_records
            if record[section].get(metric) is not None
        ]

        if not values:
            return None

        array = np.asarray(values, dtype=np.float64)
        return {
            "minimum": float(array.min()),
            "median": float(np.median(array)),
            "mean": float(array.mean()),
            "maximum": float(array.max()),
        }

    aggregate_metrics: dict[str, Any] = {}
    requested_metrics = (
        "exact_match_fraction",
        "within_5_fraction",
        "difference_mean",
        "difference_median",
        "pearson_correlation",
    )

    for section in (
        "image_vs_segmentation1",
        "image_vs_segmentation2",
        "segmentation1_vs_segmentation2",
    ):
        aggregate_metrics[section] = {
            metric: aggregate_metric(section, metric)
            for metric in requested_metrics
        }

    report = {
        "dental_root": str(dental_root),
        "pairing_result": {
            "image_count": len(image_ids),
            "segmentation1_count": len(segmentation1_ids),
            "segmentation2_count": len(segmentation2_ids),
            "common_numeric_filename_count": len(common_ids),
            "pairing_rule": (
                "Images/N.png pairs with Segmentation1/N.png and "
                "Segmentation2/N.png."
            ),
        },
        "all_three_shapes_match_count": len(same_shape_records),
        "shape_mismatch_count": len(mismatch_ids),
        "shape_mismatch_ids": mismatch_ids,
        "shape_pattern_counts": dict(shape_pattern_counts),
        "aggregate_metrics_same_shape_only": aggregate_metrics,
        "records": records,
        "visual_output_directory": str(visual_root),
        "interpretation_warning": (
            "Do not derive binary masks until the contact sheets and "
            "pixel-comparison statistics confirm whether Segmentation1 and "
            "Segmentation2 are binary masks, masked radiographs, contour "
            "overlays, or another representation."
        ),
    }

    output_path = output_root / "pandental_annotation_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== PANDENTAL ANNOTATION ANALYSIS COMPLETE ===")
    print(f"Common IDs: {len(common_ids)}")
    print(f"Same-shape cases: {len(same_shape_records)}")
    print(f"Shape mismatches: {len(mismatch_ids)}")
    if mismatch_ids:
        print(f"Mismatch IDs: {mismatch_ids}")
    print(f"JSON report: {output_path}")
    print(f"Visual checks: {visual_root}")


if __name__ == "__main__":
    main()
