from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from episodic_dataset import (
    ManifestIndex,
    ManifestRecord,
    load_binary_tensor,
    load_grayscale_tensor,
)
from universeg_model_v1 import (
    build_reduced_universeg,
    count_trainable_parameters,
)


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_NAME = "cpu_main"
DEFAULT_REPEATS = 5
DEFAULT_THRESHOLD = 0.5
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_SEED = 42

EXPECTED_HELDOUT_TASKS = {
    "pandental_mandible",
    "wbc2_nucleus",
    "wbc2_whole_cell",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen best UniverSeg checkpoint on every held-out "
            "test query. Each query prediction is averaged across repeated, "
            "independently sampled support sets."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help=(
            "Training run under outputs/training/. Default: cpu_main"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional explicit checkpoint. Default: "
            "outputs/training/<run-name>/best.pt"
        ),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Evaluation folder under outputs/evaluation/. Default: "
            "<run-name>_heldout"
        ),
    )
    parser.add_argument(
        "--support-size",
        type=int,
        default=None,
        help=(
            "Support size. Default: value stored in the checkpoint."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Independent support sets per test query. Default: 5",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Evaluate at most two test queries per task with two support "
            "repeats. Use only to validate the evaluation pipeline."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    required = {
        "epoch",
        "model_state_dict",
        "settings",
        "best_validation_macro_soft_dice",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(
            f"Checkpoint is missing keys: {missing}"
        )

    return checkpoint


def prepare_output_directory(
    project_root: Path,
    output_name: str,
    overwrite: bool,
) -> Path:
    if not output_name.strip():
        raise ValueError("output-name cannot be empty.")

    output_directory = (
        project_root
        / "outputs"
        / "evaluation"
        / output_name
    )

    if output_directory.exists():
        if not overwrite:
            raise FileExistsError(
                f"Evaluation directory already exists: {output_directory}\n"
                "Use another --output-name or pass --overwrite."
            )
        shutil.rmtree(output_directory)

    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def choose_distinct_support_records(
    candidates: Sequence[ManifestRecord],
    support_size: int,
    rng: random.Random,
) -> list[ManifestRecord]:
    by_subject: dict[str, list[ManifestRecord]] = defaultdict(list)

    for record in candidates:
        by_subject[record.subject_id].append(record)

    positive_subjects = [
        subject_id
        for subject_id, records in by_subject.items()
        if any(record.has_foreground for record in records)
    ]

    subject_ids = sorted(by_subject)

    if len(subject_ids) < support_size:
        raise RuntimeError(
            f"Need {support_size} distinct support subjects, "
            f"but only {len(subject_ids)} are available."
        )
    if not positive_subjects:
        raise RuntimeError(
            "No positive support subject is available."
        )

    selected_subjects = rng.sample(
        subject_ids,
        k=support_size,
    )

    if not any(
        subject_id in positive_subjects
        for subject_id in selected_subjects
    ):
        selected_subjects[0] = rng.choice(
            positive_subjects
        )

        if len(set(selected_subjects)) != support_size:
            remaining = [
                subject_id
                for subject_id in subject_ids
                if subject_id not in set(selected_subjects)
            ]
            if not remaining:
                raise RuntimeError(
                    "Could not maintain distinct support subjects."
                )
            selected_subjects[-1] = rng.choice(remaining)

    selected_records = [
        rng.choice(by_subject[subject_id])
        for subject_id in selected_subjects
    ]

    if len(
        {
            record.subject_id
            for record in selected_records
        }
    ) != support_size:
        raise RuntimeError(
            "Support subjects are not distinct."
        )

    return selected_records


def load_support_batch(
    records: Sequence[ManifestRecord],
) -> tuple[torch.Tensor, torch.Tensor]:
    support_images = torch.stack(
        [
            load_grayscale_tensor(record.image_path)
            for record in records
        ],
        dim=0,
    ).unsqueeze(0)

    support_masks = torch.stack(
        [
            load_binary_tensor(record.mask_path)
            for record in records
        ],
        dim=0,
    ).unsqueeze(0)

    return support_images, support_masks


def dice_from_probability(
    probability: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float,
    smooth: float = 1e-5,
) -> tuple[float, float, float]:
    probability = probability * valid_mask
    target = target * valid_mask

    intersection_soft = torch.sum(
        probability * target
    )
    denominator_soft = (
        torch.sum(probability)
        + torch.sum(target)
    )
    soft_dice = (
        2.0 * intersection_soft + smooth
    ) / (
        denominator_soft + smooth
    )

    prediction = (
        probability >= threshold
    ).to(dtype=target.dtype)
    prediction = prediction * valid_mask

    intersection_hard = torch.sum(
        prediction * target
    )
    denominator_hard = (
        torch.sum(prediction)
        + torch.sum(target)
    )
    hard_dice = (
        2.0 * intersection_hard + smooth
    ) / (
        denominator_hard + smooth
    )

    predicted_fraction = float(
        prediction.sum().item()
        / max(1.0, valid_mask.sum().item())
    )

    return (
        float(soft_dice.item()),
        float(hard_dice.item()),
        predicted_fraction,
    )


def save_probability_png(
    probability: torch.Tensor,
    path: Path,
) -> None:
    array = (
        probability.squeeze()
        .detach()
        .cpu()
        .clamp(0, 1)
        .numpy()
    )
    uint8 = np.rint(array * 255.0).astype(np.uint8)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    Image.fromarray(uint8, mode="L").save(path)


def save_binary_png(
    probability: torch.Tensor,
    threshold: float,
    path: Path,
) -> None:
    binary = (
        probability.squeeze()
        .detach()
        .cpu()
        .numpy()
        >= threshold
    ).astype(np.uint8) * 255

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    Image.fromarray(binary, mode="L").save(path)


def bootstrap_mean_interval(
    values: Sequence[float],
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1 or array.size < 1:
        raise ValueError(
            "Bootstrap input must be a nonempty one-dimensional array."
        )
    if samples < 1:
        raise ValueError(
            "bootstrap-samples must be positive."
        )
    if not 0 < confidence < 1:
        raise ValueError(
            "confidence must lie between zero and one."
        )

    if array.size == 1:
        value = float(array[0])
        return {
            "mean": value,
            "lower": value,
            "upper": value,
        }

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(
        samples,
        dtype=np.float64,
    )

    for index in range(samples):
        resampled = rng.choice(
            array,
            size=array.size,
            replace=True,
        )
        bootstrap_means[index] = np.mean(
            resampled
        )

    alpha = (1.0 - confidence) / 2.0

    return {
        "mean": float(np.mean(array)),
        "lower": float(
            np.quantile(
                bootstrap_means,
                alpha,
            )
        ),
        "upper": float(
            np.quantile(
                bootstrap_means,
                1.0 - alpha,
            )
        ),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if args.repeats < 1:
        raise ValueError("repeats must be positive.")
    if not 0 < args.threshold < 1:
        raise ValueError(
            "threshold must lie strictly between zero and one."
        )
    if args.bootstrap_samples < 1:
        raise ValueError(
            "bootstrap-samples must be positive."
        )
    if args.cpu_threads < 1:
        raise ValueError(
            "cpu-threads must be positive."
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(
        max(
            1,
            min(
                args.cpu_threads,
                torch.get_num_threads(),
            ),
        )
    )

    project_root = args.project_root.expanduser().resolve()
    processed_root = project_root / "data" / "processed"

    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else (
            project_root
            / "outputs"
            / "training"
            / args.run_name
            / "best.pt"
        )
    )

    checkpoint = load_checkpoint(
        checkpoint_path
    )
    checkpoint_settings = checkpoint["settings"]

    feature_width = int(
        checkpoint_settings["feature_width"]
    )
    support_size = (
        int(args.support_size)
        if args.support_size is not None
        else int(checkpoint_settings["support_size"])
    )
    repeats = args.repeats

    if args.smoke_test:
        repeats = 2

    output_name = (
        args.output_name
        if args.output_name is not None
        else f"{args.run_name}_heldout"
    )
    if args.smoke_test:
        output_name = f"{output_name}_smoke"

    output_directory = prepare_output_directory(
        project_root,
        output_name,
        args.overwrite,
    )

    index = ManifestIndex(
        manifest_path=processed_root / "manifest.csv",
        processed_root=processed_root,
    )

    heldout_tasks = {
        task_id
        for dataset in index.datasets("heldout")
        for task_id in index.tasks(
            "heldout",
            dataset,
        )
    }

    if heldout_tasks != EXPECTED_HELDOUT_TASKS:
        raise RuntimeError(
            "Unexpected held-out task set: "
            f"{sorted(heldout_tasks)}"
        )

    model = build_reduced_universeg(
        feature_width=feature_width
    )
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model state mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    query_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    task_scores: dict[
        str,
        dict[str, list[float]],
    ] = defaultdict(
        lambda: {
            "soft_dice": [],
            "hard_dice": [],
        }
    )

    evaluation_start = time.perf_counter()
    task_number = 0

    with torch.inference_mode():
        for dataset in index.datasets("heldout"):
            for task_id in index.tasks(
                "heldout",
                dataset,
            ):
                task_number += 1

                query_records = sorted(
                    index.records_for(
                        role="heldout",
                        dataset=dataset,
                        task_id=task_id,
                        split="test",
                    ),
                    key=lambda record: (
                        record.subject_id,
                        record.sample_id,
                    ),
                )
                support_candidates = (
                    index.records_for(
                        role="heldout",
                        dataset=dataset,
                        task_id=task_id,
                        split="support",
                    )
                )

                if args.smoke_test:
                    query_records = query_records[:2]

                print(
                    f"\nTask {task_number}/3: {task_id} "
                    f"| queries={len(query_records)} "
                    f"| support={support_size} "
                    f"| repeats={repeats}",
                    flush=True,
                )

                for query_index, query_record in enumerate(
                    query_records,
                    start=1,
                ):
                    query_image = load_grayscale_tensor(
                        query_record.image_path
                    ).unsqueeze(0)
                    query_mask = load_binary_tensor(
                        query_record.mask_path
                    ).unsqueeze(0)
                    query_valid_mask = load_binary_tensor(
                        query_record.valid_mask_path
                    ).unsqueeze(0)

                    probabilities: list[torch.Tensor] = []
                    individual_soft_dices: list[float] = []
                    individual_hard_dices: list[float] = []

                    for repeat_index in range(repeats):
                        repeat_seed = (
                            args.seed
                            + task_number * 10_000_019
                            + query_index * 100_003
                            + repeat_index * 1_009
                        )
                        rng = random.Random(
                            repeat_seed
                        )

                        support_records = (
                            choose_distinct_support_records(
                                candidates=support_candidates,
                                support_size=support_size,
                                rng=rng,
                            )
                        )
                        support_images, support_masks = (
                            load_support_batch(
                                support_records
                            )
                        )

                        logits = model(
                            query_image,
                            support_images,
                            support_masks,
                        )
                        probability = torch.sigmoid(
                            logits
                        )
                        probabilities.append(
                            probability.cpu()
                        )

                        (
                            individual_soft,
                            individual_hard,
                            _,
                        ) = dice_from_probability(
                            probability=probability,
                            target=query_mask,
                            valid_mask=query_valid_mask,
                            threshold=args.threshold,
                        )
                        individual_soft_dices.append(
                            individual_soft
                        )
                        individual_hard_dices.append(
                            individual_hard
                        )

                        support_rows.append(
                            {
                                "dataset": dataset,
                                "task_id": task_id,
                                "query_sample_id": (
                                    query_record.sample_id
                                ),
                                "query_subject_id": (
                                    query_record.subject_id
                                ),
                                "repeat_index": (
                                    repeat_index + 1
                                ),
                                "repeat_seed": repeat_seed,
                                "support_sample_ids": "|".join(
                                    record.sample_id
                                    for record in support_records
                                ),
                                "support_subject_ids": "|".join(
                                    record.subject_id
                                    for record in support_records
                                ),
                                "individual_soft_dice": (
                                    individual_soft
                                ),
                                "individual_hard_dice": (
                                    individual_hard
                                ),
                            }
                        )

                    ensemble_probability = torch.stack(
                        probabilities,
                        dim=0,
                    ).mean(dim=0)

                    (
                        ensemble_soft_dice,
                        ensemble_hard_dice,
                        predicted_fraction,
                    ) = dice_from_probability(
                        probability=ensemble_probability,
                        target=query_mask,
                        valid_mask=query_valid_mask,
                        threshold=args.threshold,
                    )

                    target_fraction = float(
                        query_mask.sum().item()
                        / max(
                            1.0,
                            query_valid_mask.sum().item(),
                        )
                    )

                    safe_sample_name = (
                        query_record.sample_id
                        .replace(":", "__")
                        .replace("/", "_")
                        .replace("\\", "_")
                    )
                    probability_path = (
                        output_directory
                        / "probabilities"
                        / task_id
                        / f"{safe_sample_name}.png"
                    )
                    prediction_path = (
                        output_directory
                        / "predictions"
                        / task_id
                        / f"{safe_sample_name}.png"
                    )

                    save_probability_png(
                        ensemble_probability,
                        probability_path,
                    )
                    save_binary_png(
                        ensemble_probability,
                        args.threshold,
                        prediction_path,
                    )

                    query_rows.append(
                        {
                            "dataset": dataset,
                            "task_id": task_id,
                            "sample_id": (
                                query_record.sample_id
                            ),
                            "subject_id": (
                                query_record.subject_id
                            ),
                            "support_size": support_size,
                            "support_repeats": repeats,
                            "threshold": args.threshold,
                            "ensemble_soft_dice": (
                                ensemble_soft_dice
                            ),
                            "ensemble_hard_dice": (
                                ensemble_hard_dice
                            ),
                            "individual_soft_dice_mean": (
                                float(
                                    np.mean(
                                        individual_soft_dices
                                    )
                                )
                            ),
                            "individual_soft_dice_std": (
                                float(
                                    np.std(
                                        individual_soft_dices
                                    )
                                )
                            ),
                            "individual_hard_dice_mean": (
                                float(
                                    np.mean(
                                        individual_hard_dices
                                    )
                                )
                            ),
                            "individual_hard_dice_std": (
                                float(
                                    np.std(
                                        individual_hard_dices
                                    )
                                )
                            ),
                            "target_foreground_fraction": (
                                target_fraction
                            ),
                            "prediction_foreground_fraction": (
                                predicted_fraction
                            ),
                            "probability_path": str(
                                probability_path
                            ),
                            "prediction_path": str(
                                prediction_path
                            ),
                        }
                    )

                    task_scores[
                        task_id
                    ]["soft_dice"].append(
                        ensemble_soft_dice
                    )
                    task_scores[
                        task_id
                    ]["hard_dice"].append(
                        ensemble_hard_dice
                    )

                    print(
                        f"  query "
                        f"{query_index:02d}/"
                        f"{len(query_records):02d} "
                        f"| soft={ensemble_soft_dice:.4f} "
                        f"| hard={ensemble_hard_dice:.4f}",
                        flush=True,
                    )

    elapsed_seconds = (
        time.perf_counter() - evaluation_start
    )

    task_summary_rows: list[dict[str, Any]] = []
    task_summary_json: dict[str, Any] = {}

    for task_index, task_id in enumerate(
        sorted(task_scores),
        start=1,
    ):
        task_query_rows = [
            row
            for row in query_rows
            if row["task_id"] == task_id
        ]
        dataset = str(
            task_query_rows[0]["dataset"]
        )

        soft_interval = bootstrap_mean_interval(
            task_scores[task_id]["soft_dice"],
            samples=args.bootstrap_samples,
            seed=args.seed + task_index * 2_003,
        )
        hard_interval = bootstrap_mean_interval(
            task_scores[task_id]["hard_dice"],
            samples=args.bootstrap_samples,
            seed=args.seed + task_index * 3_001,
        )

        task_row = {
            "dataset": dataset,
            "task_id": task_id,
            "test_queries": len(
                task_scores[task_id]["hard_dice"]
            ),
            "support_size": support_size,
            "support_repeats": repeats,
            "mean_soft_dice": soft_interval["mean"],
            "soft_dice_ci95_lower": (
                soft_interval["lower"]
            ),
            "soft_dice_ci95_upper": (
                soft_interval["upper"]
            ),
            "mean_hard_dice": hard_interval["mean"],
            "hard_dice_ci95_lower": (
                hard_interval["lower"]
            ),
            "hard_dice_ci95_upper": (
                hard_interval["upper"]
            ),
            "median_hard_dice": float(
                np.median(
                    task_scores[task_id][
                        "hard_dice"
                    ]
                )
            ),
            "minimum_hard_dice": float(
                np.min(
                    task_scores[task_id][
                        "hard_dice"
                    ]
                )
            ),
            "maximum_hard_dice": float(
                np.max(
                    task_scores[task_id][
                        "hard_dice"
                    ]
                )
            ),
        }

        task_summary_rows.append(task_row)
        task_summary_json[task_id] = task_row

    macro_soft_dice = float(
        np.mean(
            [
                row["mean_soft_dice"]
                for row in task_summary_rows
            ]
        )
    )
    macro_hard_dice = float(
        np.mean(
            [
                row["mean_hard_dice"]
                for row in task_summary_rows
            ]
        )
    )
    pooled_soft_dice = float(
        np.mean(
            [
                row["ensemble_soft_dice"]
                for row in query_rows
            ]
        )
    )
    pooled_hard_dice = float(
        np.mean(
            [
                row["ensemble_hard_dice"]
                for row in query_rows
            ]
        )
    )

    query_csv_path = (
        output_directory / "per_query_results.csv"
    )
    support_csv_path = (
        output_directory / "support_sets.csv"
    )
    task_csv_path = (
        output_directory / "task_summary.csv"
    )
    summary_path = (
        output_directory / "evaluation_summary.json"
    )

    write_csv(
        query_csv_path,
        query_rows,
        [
            "dataset",
            "task_id",
            "sample_id",
            "subject_id",
            "support_size",
            "support_repeats",
            "threshold",
            "ensemble_soft_dice",
            "ensemble_hard_dice",
            "individual_soft_dice_mean",
            "individual_soft_dice_std",
            "individual_hard_dice_mean",
            "individual_hard_dice_std",
            "target_foreground_fraction",
            "prediction_foreground_fraction",
            "probability_path",
            "prediction_path",
        ],
    )
    write_csv(
        support_csv_path,
        support_rows,
        [
            "dataset",
            "task_id",
            "query_sample_id",
            "query_subject_id",
            "repeat_index",
            "repeat_seed",
            "support_sample_ids",
            "support_subject_ids",
            "individual_soft_dice",
            "individual_hard_dice",
        ],
    )
    write_csv(
        task_csv_path,
        task_summary_rows,
        [
            "dataset",
            "task_id",
            "test_queries",
            "support_size",
            "support_repeats",
            "mean_soft_dice",
            "soft_dice_ci95_lower",
            "soft_dice_ci95_upper",
            "mean_hard_dice",
            "hard_dice_ci95_lower",
            "hard_dice_ci95_upper",
            "median_hard_dice",
            "minimum_hard_dice",
            "maximum_hard_dice",
        ],
    )

    summary = {
        "status": "completed",
        "smoke_test": bool(args.smoke_test),
        "project_root": str(project_root),
        "manifest": str(
            processed_root / "manifest.csv"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch_one_based": (
            int(checkpoint["epoch"]) + 1
        ),
        "checkpoint_validation_macro_soft_dice": (
            float(
                checkpoint[
                    "best_validation_macro_soft_dice"
                ]
            )
        ),
        "feature_width": feature_width,
        "trainable_parameters": (
            count_trainable_parameters(model)
        ),
        "support_size": support_size,
        "support_repeats": repeats,
        "ensemble_rule": (
            "Mean of sigmoid probability maps across independently "
            "sampled support sets, followed by thresholding."
        ),
        "threshold": args.threshold,
        "query_policy": (
            "Every held-out test query is evaluated exactly once."
        ),
        "support_policy": (
            "Support examples are drawn without replacement from distinct "
            "support-split subjects. Test queries are never support items."
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "elapsed_seconds": elapsed_seconds,
        "test_query_count": len(query_rows),
        "inference_count": len(support_rows),
        "task_results": task_summary_json,
        "macro_mean_soft_dice": macro_soft_dice,
        "macro_mean_hard_dice": macro_hard_dice,
        "pooled_query_mean_soft_dice": (
            pooled_soft_dice
        ),
        "pooled_query_mean_hard_dice": (
            pooled_hard_dice
        ),
        "outputs": {
            "per_query_results": str(query_csv_path),
            "support_sets": str(support_csv_path),
            "task_summary": str(task_csv_path),
            "probabilities_directory": str(
                output_directory / "probabilities"
            ),
            "predictions_directory": str(
                output_directory / "predictions"
            ),
        },
        "scientific_interpretation": {
            "pandental_mandible": (
                "True held-out modality/anatomy task in this reduced "
                "project."
            ),
            "wbc2_nucleus": (
                "Held-out source subset, but the same microscopy anatomy "
                "and nucleus task are represented by WBC Dataset 1 during "
                "training."
            ),
            "wbc2_whole_cell": (
                "Held-out source subset, but the same microscopy anatomy "
                "and whole-cell task are represented by WBC Dataset 1 "
                "during training."
            ),
            "paper_comparison": (
                "These scores must not be directly equated with the "
                "original UniverSeg paper because the training collection, "
                "feature width, support size, and WBC holdout definition "
                "differ."
            ),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(
        "\n=== HELD-OUT EVALUATION COMPLETE ==="
    )
    print(
        "Checkpoint epoch:",
        summary["checkpoint_epoch_one_based"],
    )
    print(
        "Test queries / inferences:",
        len(query_rows),
        len(support_rows),
    )
    print(
        "Macro soft/hard Dice:",
        macro_soft_dice,
        macro_hard_dice,
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
