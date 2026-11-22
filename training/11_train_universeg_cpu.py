from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from episodic_dataset import (
    Episode,
    EpisodeConfig,
    EpisodeSampler,
    ManifestIndex,
)
from universeg_losses import (
    SoftDiceLoss,
    hard_dice_score_from_logits,
    soft_dice_score_from_logits,
)
from universeg_model_v1 import (
    build_reduced_universeg,
    count_trainable_parameters,
)


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainingSettings:
    feature_width: int
    support_size: int
    epochs: int
    episodes_per_epoch: int
    validation_episodes_per_task: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    early_stopping_patience: int
    seed: int
    cpu_threads: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the reduced from-scratch UniverSeg model on BUSI, "
            "WBC Dataset 1, and ACDC. Checkpoint selection uses fixed "
            "development episodes from the known training tasks."
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
        default="cpu_main",
    )
    parser.add_argument(
        "--feature-width",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--support-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--episodes-per-epoch",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--validation-episodes-per-task",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
            "Use a fast end-to-end validation configuration: width 8, "
            "support 2, 2 epochs, 12 training episodes per epoch, and "
            "1 fixed validation episode per task."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last.pt in the selected run directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing run directory before starting.",
    )
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> TrainingSettings:
    if args.smoke_test:
        settings = TrainingSettings(
            feature_width=8,
            support_size=2,
            epochs=2,
            episodes_per_epoch=12,
            validation_episodes_per_task=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip_norm=5.0,
            early_stopping_patience=2,
            seed=args.seed,
            cpu_threads=args.cpu_threads,
        )
    else:
        settings = TrainingSettings(
            feature_width=args.feature_width,
            support_size=args.support_size,
            epochs=args.epochs,
            episodes_per_epoch=args.episodes_per_epoch,
            validation_episodes_per_task=(
                args.validation_episodes_per_task
            ),
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip_norm=args.gradient_clip_norm,
            early_stopping_patience=args.early_stopping_patience,
            seed=args.seed,
            cpu_threads=args.cpu_threads,
        )

    integer_fields = {
        "feature_width": settings.feature_width,
        "support_size": settings.support_size,
        "epochs": settings.epochs,
        "episodes_per_epoch": settings.episodes_per_epoch,
        "validation_episodes_per_task": (
            settings.validation_episodes_per_task
        ),
        "early_stopping_patience": (
            settings.early_stopping_patience
        ),
        "cpu_threads": settings.cpu_threads,
    }
    for name, value in integer_fields.items():
        if value < 1:
            raise ValueError(f"{name} must be positive.")

    if settings.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if settings.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")
    if settings.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive.")

    return settings


def set_global_seed(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(
        max(1, min(cpu_threads, torch.get_num_threads()))
    )


def prepare_run_directory(
    project_root: Path,
    run_name: str,
    resume: bool,
    overwrite: bool,
) -> Path:
    if not run_name.strip():
        raise ValueError("run-name cannot be empty.")

    run_directory = (
        project_root
        / "outputs"
        / "training"
        / run_name
    )

    if resume and overwrite:
        raise ValueError(
            "--resume and --overwrite cannot be used together."
        )

    if overwrite and run_directory.exists():
        shutil.rmtree(run_directory)

    if resume:
        checkpoint = run_directory / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Cannot resume; checkpoint not found: {checkpoint}"
            )
    elif run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError(
            f"Run directory already contains files: {run_directory}\n"
            "Use a new --run-name or pass --overwrite."
        )

    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory


def episode_to_batch(
    episode: Episode,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    query_image = episode.query_image.unsqueeze(0)
    query_mask = episode.query_mask.unsqueeze(0)
    query_valid_mask = episode.query_valid_mask.unsqueeze(0)
    support_images = episode.support_images.unsqueeze(0)
    support_masks = episode.support_masks.unsqueeze(0)

    return (
        query_image,
        query_mask,
        query_valid_mask,
        support_images,
        support_masks,
    )


def sample_distinct_subject_episode(
    sampler: EpisodeSampler,
    task_id: str | None = None,
    maximum_attempts: int = 100,
) -> Episode:
    """
    Require distinct support subjects.

    This matters for ACDC because one patient contributes two annotated
    frames. The base sampler already excludes the query patient; this
    wrapper also prevents two support frames from the same patient.
    """
    for _ in range(maximum_attempts):
        episode = sampler.sample(task_id=task_id)
        support_subjects = episode.metadata[
            "support_subject_ids"
        ]

        if len(support_subjects) == len(set(support_subjects)):
            return episode

    raise RuntimeError(
        "Could not sample an episode with distinct support subjects "
        f"after {maximum_attempts} attempts. Task={task_id!r}."
    )


def finite_scalar(value: torch.Tensor, name: str) -> float:
    if value.ndim != 0:
        raise RuntimeError(
            f"{name} must be scalar, got {tuple(value.shape)}."
        )
    if not torch.isfinite(value):
        raise RuntimeError(f"{name} is not finite.")
    return float(value.item())


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: SoftDiceLoss,
    index: ManifestIndex,
    settings: TrainingSettings,
    epoch_index: int,
) -> dict[str, Any]:
    model.train()

    sampler = EpisodeSampler(
        index=index,
        config=EpisodeConfig(
            task_role="train",
            query_split="support",
            support_split="support",
            support_size=settings.support_size,
            support_with_replacement=False,
            require_positive_support=True,
        ),
        seed=settings.seed + epoch_index * 100_003,
    )

    loss_values: list[float] = []
    hard_dice_values: list[float] = []
    soft_dice_values: list[float] = []
    gradient_norm_values: list[float] = []
    task_episode_counts: defaultdict[str, int] = defaultdict(int)

    epoch_start = time.perf_counter()

    for episode_index in range(settings.episodes_per_epoch):
        episode = sample_distinct_subject_episode(sampler)

        (
            query_image,
            query_mask,
            query_valid_mask,
            support_images,
            support_masks,
        ) = episode_to_batch(episode)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            query_image,
            support_images,
            support_masks,
        )
        loss = criterion(
            logits,
            query_mask,
            valid_mask=query_valid_mask,
        )

        loss_value = finite_scalar(loss, "training loss")
        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=settings.gradient_clip_norm,
        )
        gradient_norm_float = float(gradient_norm.item())

        if not math.isfinite(gradient_norm_float):
            raise RuntimeError(
                "Non-finite gradient norm during training."
            )

        optimizer.step()

        with torch.no_grad():
            soft_dice = soft_dice_score_from_logits(
                logits,
                query_mask,
                valid_mask=query_valid_mask,
            )
            hard_dice = hard_dice_score_from_logits(
                logits,
                query_mask,
                valid_mask=query_valid_mask,
            )

        loss_values.append(loss_value)
        soft_dice_values.append(
            finite_scalar(soft_dice, "training soft Dice")
        )
        hard_dice_values.append(
            finite_scalar(hard_dice, "training hard Dice")
        )
        gradient_norm_values.append(gradient_norm_float)

        task_id = str(episode.metadata["task_id"])
        task_episode_counts[task_id] += 1

        if (
            (episode_index + 1) % 5 == 0
            or episode_index + 1 == settings.episodes_per_epoch
        ):
            print(
                f"  train episode "
                f"{episode_index + 1:03d}/"
                f"{settings.episodes_per_epoch:03d} "
                f"| task={task_id} "
                f"| loss={loss_value:.4f}",
                flush=True,
            )

    elapsed_seconds = time.perf_counter() - epoch_start

    return {
        "mean_loss": float(np.mean(loss_values)),
        "mean_soft_dice": float(np.mean(soft_dice_values)),
        "mean_hard_dice": float(np.mean(hard_dice_values)),
        "mean_gradient_norm_before_clipping": float(
            np.mean(gradient_norm_values)
        ),
        "maximum_gradient_norm_before_clipping": float(
            np.max(gradient_norm_values)
        ),
        "elapsed_seconds": elapsed_seconds,
        "task_episode_counts": dict(
            sorted(task_episode_counts.items())
        ),
    }


@torch.no_grad()
def evaluate_known_tasks(
    model: torch.nn.Module,
    criterion: SoftDiceLoss,
    index: ManifestIndex,
    settings: TrainingSettings,
) -> dict[str, Any]:
    """
    Fixed development evaluation for checkpoint selection.

    The same task-specific seeds are used after every epoch, so checkpoints
    are compared on identical episodes.
    """
    model.eval()

    task_results: dict[str, dict[str, Any]] = {}
    task_counter = 0

    for dataset in index.datasets("train"):
        for task_id in index.tasks("train", dataset):
            sampler = EpisodeSampler(
                index=index,
                config=EpisodeConfig(
                    task_role="train",
                    query_split="dev",
                    support_split="support",
                    support_size=settings.support_size,
                    support_with_replacement=False,
                    require_positive_support=True,
                ),
                seed=(
                    settings.seed
                    + 5_000_000
                    + task_counter * 10_007
                ),
            )
            task_counter += 1

            losses: list[float] = []
            soft_dices: list[float] = []
            hard_dices: list[float] = []
            query_sample_ids: list[str] = []

            for _ in range(
                settings.validation_episodes_per_task
            ):
                episode = sample_distinct_subject_episode(
                    sampler,
                    task_id=task_id,
                )

                (
                    query_image,
                    query_mask,
                    query_valid_mask,
                    support_images,
                    support_masks,
                ) = episode_to_batch(episode)

                logits = model(
                    query_image,
                    support_images,
                    support_masks,
                )

                loss = criterion(
                    logits,
                    query_mask,
                    valid_mask=query_valid_mask,
                )
                soft_dice = soft_dice_score_from_logits(
                    logits,
                    query_mask,
                    valid_mask=query_valid_mask,
                )
                hard_dice = hard_dice_score_from_logits(
                    logits,
                    query_mask,
                    valid_mask=query_valid_mask,
                )

                losses.append(
                    finite_scalar(loss, "validation loss")
                )
                soft_dices.append(
                    finite_scalar(
                        soft_dice,
                        "validation soft Dice",
                    )
                )
                hard_dices.append(
                    finite_scalar(
                        hard_dice,
                        "validation hard Dice",
                    )
                )
                query_sample_ids.append(
                    str(
                        episode.metadata[
                            "query_sample_id"
                        ]
                    )
                )

            task_results[task_id] = {
                "dataset": dataset,
                "episodes": (
                    settings.validation_episodes_per_task
                ),
                "mean_loss": float(np.mean(losses)),
                "mean_soft_dice": float(
                    np.mean(soft_dices)
                ),
                "mean_hard_dice": float(
                    np.mean(hard_dices)
                ),
                "query_sample_ids": query_sample_ids,
            }

    macro_loss = float(
        np.mean(
            [
                result["mean_loss"]
                for result in task_results.values()
            ]
        )
    )
    macro_soft_dice = float(
        np.mean(
            [
                result["mean_soft_dice"]
                for result in task_results.values()
            ]
        )
    )
    macro_hard_dice = float(
        np.mean(
            [
                result["mean_hard_dice"]
                for result in task_results.values()
            ]
        )
    )

    return {
        "macro_mean_loss": macro_loss,
        "macro_mean_soft_dice": macro_soft_dice,
        "macro_mean_hard_dice": macro_hard_dice,
        "task_results": task_results,
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    epochs_without_improvement: int,
    history: list[dict[str, Any]],
    settings: TrainingSettings,
    source_manifest: Path,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_macro_soft_dice": best_metric,
        "epochs_without_improvement": (
            epochs_without_improvement
        ),
        "history": history,
        "settings": asdict(settings),
        "source_manifest": str(source_manifest),
        "rng_state": capture_rng_state(),
    }
    torch.save(checkpoint, path)


def write_history_csv(
    path: Path,
    history: list[dict[str, Any]],
) -> None:
    columns = [
        "epoch",
        "train_mean_loss",
        "train_mean_soft_dice",
        "train_mean_hard_dice",
        "train_mean_gradient_norm",
        "train_elapsed_seconds",
        "validation_macro_mean_loss",
        "validation_macro_mean_soft_dice",
        "validation_macro_mean_hard_dice",
        "is_best",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    column: row[column]
                    for column in columns
                }
            )


def main() -> None:
    args = parse_args()
    settings = build_settings(args)

    project_root = args.project_root.expanduser().resolve()
    processed_root = project_root / "data" / "processed"
    manifest_path = processed_root / "manifest.csv"

    set_global_seed(
        settings.seed,
        settings.cpu_threads,
    )

    run_directory = prepare_run_directory(
        project_root=project_root,
        run_name=args.run_name,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    settings_path = run_directory / "settings.json"
    history_path = run_directory / "history.csv"
    last_checkpoint_path = run_directory / "last.pt"
    best_checkpoint_path = run_directory / "best.pt"
    summary_path = run_directory / "training_summary.json"

    index = ManifestIndex(
        manifest_path=manifest_path,
        processed_root=processed_root,
    )

    model = build_reduced_universeg(
        feature_width=settings.feature_width
    )
    criterion = SoftDiceLoss(
        smooth=1e-5,
        reduction="mean",
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    start_epoch = 0
    best_metric = -math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    if args.resume:
        checkpoint = torch.load(
            last_checkpoint_path,
            map_location="cpu",
        )

        saved_settings = TrainingSettings(
            **checkpoint["settings"]
        )
        if saved_settings != settings:
            raise RuntimeError(
                "Current settings do not match the saved checkpoint.\n"
                f"Current: {asdict(settings)}\n"
                f"Saved:   {asdict(saved_settings)}"
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(
            checkpoint[
                "best_validation_macro_soft_dice"
            ]
        )
        epochs_without_improvement = int(
            checkpoint["epochs_without_improvement"]
        )
        history = list(checkpoint["history"])
        restore_rng_state(checkpoint["rng_state"])

    settings_payload = {
        "settings": asdict(settings),
        "smoke_test": bool(args.smoke_test),
        "run_name": args.run_name,
        "manifest": str(manifest_path),
        "training_datasets": list(
            index.datasets("train")
        ),
        "training_tasks": {
            dataset: list(
                index.tasks("train", dataset)
            )
            for dataset in index.datasets("train")
        },
        "checkpoint_selection": (
            "Highest macro mean soft Dice across the eight fixed "
            "training-task development evaluations."
        ),
        "model_trainable_parameters": (
            count_trainable_parameters(model)
        ),
        "device": "cpu",
    }
    settings_path.write_text(
        json.dumps(settings_payload, indent=2),
        encoding="utf-8",
    )

    print("\n=== UNIVERSEG TRAINING START ===")
    print(f"Run directory: {run_directory}")
    print(f"Device: CPU")
    print(
        "Model parameters:",
        count_trainable_parameters(model),
    )
    print(
        "Settings:",
        json.dumps(asdict(settings)),
    )

    training_start = time.perf_counter()
    stopped_early = False

    for epoch_index in range(
        start_epoch,
        settings.epochs,
    ):
        print(
            f"\nEpoch {epoch_index + 1}/{settings.epochs}",
            flush=True,
        )

        training_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            index=index,
            settings=settings,
            epoch_index=epoch_index,
        )

        validation_metrics = evaluate_known_tasks(
            model=model,
            criterion=criterion,
            index=index,
            settings=settings,
        )

        validation_metric = float(
            validation_metrics[
                "macro_mean_soft_dice"
            ]
        )
        is_best = validation_metric > best_metric

        if is_best:
            best_metric = validation_metric
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history_row = {
            "epoch": epoch_index + 1,
            "train_mean_loss": (
                training_metrics["mean_loss"]
            ),
            "train_mean_soft_dice": (
                training_metrics["mean_soft_dice"]
            ),
            "train_mean_hard_dice": (
                training_metrics["mean_hard_dice"]
            ),
            "train_mean_gradient_norm": (
                training_metrics[
                    "mean_gradient_norm_before_clipping"
                ]
            ),
            "train_elapsed_seconds": (
                training_metrics["elapsed_seconds"]
            ),
            "validation_macro_mean_loss": (
                validation_metrics[
                    "macro_mean_loss"
                ]
            ),
            "validation_macro_mean_soft_dice": (
                validation_metrics[
                    "macro_mean_soft_dice"
                ]
            ),
            "validation_macro_mean_hard_dice": (
                validation_metrics[
                    "macro_mean_hard_dice"
                ]
            ),
            "is_best": int(is_best),
            "training_task_episode_counts": (
                training_metrics[
                    "task_episode_counts"
                ]
            ),
            "validation_task_results": (
                validation_metrics[
                    "task_results"
                ]
            ),
        }
        history.append(history_row)

        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch_index,
            best_metric=best_metric,
            epochs_without_improvement=(
                epochs_without_improvement
            ),
            history=history,
            settings=settings,
            source_manifest=manifest_path,
        )

        if is_best:
            shutil.copy2(
                last_checkpoint_path,
                best_checkpoint_path,
            )

        write_history_csv(
            history_path,
            history,
        )

        print(
            "  train loss/soft/hard: "
            f"{training_metrics['mean_loss']:.4f} / "
            f"{training_metrics['mean_soft_dice']:.4f} / "
            f"{training_metrics['mean_hard_dice']:.4f}"
        )
        print(
            "  dev macro loss/soft/hard: "
            f"{validation_metrics['macro_mean_loss']:.4f} / "
            f"{validation_metrics['macro_mean_soft_dice']:.4f} / "
            f"{validation_metrics['macro_mean_hard_dice']:.4f}"
        )
        print(
            f"  best macro soft Dice: {best_metric:.4f}"
            + ("  [NEW BEST]" if is_best else "")
        )

        if (
            epochs_without_improvement
            >= settings.early_stopping_patience
        ):
            stopped_early = True
            print(
                "Early stopping: no validation improvement for "
                f"{settings.early_stopping_patience} epochs."
            )
            break

    total_elapsed = time.perf_counter() - training_start

    if not best_checkpoint_path.is_file():
        raise RuntimeError(
            "Training ended without producing best.pt."
        )

    best_epoch_row = max(
        history,
        key=lambda row: row[
            "validation_macro_mean_soft_dice"
        ],
    )

    summary = {
        "status": "completed",
        "smoke_test": bool(args.smoke_test),
        "run_name": args.run_name,
        "run_directory": str(run_directory),
        "settings": asdict(settings),
        "device": "cpu",
        "trainable_parameters": (
            count_trainable_parameters(model)
        ),
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_epoch": best_epoch_row["epoch"],
        "best_validation_macro_soft_dice": (
            best_epoch_row[
                "validation_macro_mean_soft_dice"
            ]
        ),
        "best_validation_macro_hard_dice": (
            best_epoch_row[
                "validation_macro_mean_hard_dice"
            ]
        ),
        "best_validation_macro_loss": (
            best_epoch_row[
                "validation_macro_mean_loss"
            ]
        ),
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "history_csv": str(history_path),
        "total_elapsed_seconds": total_elapsed,
        "scientific_scope": (
            "Checkpoint selection uses only development episodes from "
            "known training tasks. Held-out WBC Dataset 2 and PanDental "
            "are not accessed during training or checkpoint selection."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n UNIVERSEG TRAINING COMPLETE ")
    print(f"Epochs completed: {len(history)}")
    print(
        "Best epoch / macro soft Dice:",
        summary["best_epoch"],
        summary["best_validation_macro_soft_dice"],
    )
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
