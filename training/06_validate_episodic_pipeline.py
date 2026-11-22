from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from episodic_dataset import (
    EpisodeConfig,
    EpisodeSampler,
    EpisodicSegmentationDataset,
    ManifestIndex,
)


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUPPORT_SIZE = 8
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the merged manifest-backed episodic data pipeline "
            "before implementing or training UniverSeg."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Repository root. Defaults to this repository directory.",
    )
    parser.add_argument(
        "--support-size",
        type=int,
        default=DEFAULT_SUPPORT_SIZE,
        help="CPU-friendly smoke-test support size. Default: 8",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic sampler seed. Default: 42",
    )
    return parser.parse_args()


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "finite": bool(torch.isfinite(tensor).all().item()),
    }


def assert_image_tensor(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    name: str,
) -> None:
    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"{name} has shape {tuple(tensor.shape)}, "
            f"expected {expected_shape}."
        )
    if tensor.dtype != torch.float32:
        raise RuntimeError(
            f"{name} must be float32, received {tensor.dtype}."
        )
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains non-finite values.")
    if tensor.min() < 0 or tensor.max() > 1:
        raise RuntimeError(f"{name} is not normalized to [0,1].")


def assert_binary_tensor(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    name: str,
) -> None:
    assert_image_tensor(tensor, expected_shape, name)
    unique_values = set(float(value) for value in torch.unique(tensor))
    if not unique_values.issubset({0.0, 1.0}):
        raise RuntimeError(
            f"{name} is not binary: {sorted(unique_values)}"
        )


def validate_episode(
    episode: Any,
    support_size: int,
) -> dict[str, Any]:
    assert_image_tensor(
        episode.query_image,
        (1, 128, 128),
        "query_image",
    )
    assert_binary_tensor(
        episode.query_mask,
        (1, 128, 128),
        "query_mask",
    )
    assert_binary_tensor(
        episode.query_valid_mask,
        (1, 128, 128),
        "query_valid_mask",
    )
    assert_image_tensor(
        episode.support_images,
        (support_size, 1, 128, 128),
        "support_images",
    )
    assert_binary_tensor(
        episode.support_masks,
        (support_size, 1, 128, 128),
        "support_masks",
    )
    assert_binary_tensor(
        episode.support_valid_masks,
        (support_size, 1, 128, 128),
        "support_valid_masks",
    )

    metadata = episode.metadata

    if metadata["query_sample_id"] in metadata["support_sample_ids"]:
        raise RuntimeError(
            "The query sample was included in its own support set."
        )

    if metadata["query_subject_id"] in metadata["support_subject_ids"]:
        raise RuntimeError(
            "Another sample from the query subject was included in the "
            "support set."
        )

    if not any(metadata["support_has_foreground"]):
        raise RuntimeError(
            "Episode support set contains no positive mask."
        )

    return {
        "metadata": metadata,
        "query_image": tensor_summary(episode.query_image),
        "query_mask": tensor_summary(episode.query_mask),
        "support_images": tensor_summary(episode.support_images),
        "support_masks": tensor_summary(episode.support_masks),
        "support_positive_count": int(
            sum(metadata["support_has_foreground"])
        ),
    }


def main() -> None:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    processed_root = project_root / "data" / "processed"
    manifest_path = processed_root / "manifest.csv"

    index = ManifestIndex(
        manifest_path=manifest_path,
        processed_root=processed_root,
    )

    report: dict[str, Any] = {
        "status": "passed",
        "manifest": str(manifest_path),
        "support_size": args.support_size,
        "seed": args.seed,
        "datasets_by_role": {
            role: list(index.datasets(role))
            for role in ("train", "heldout")
        },
        "tasks_by_role_and_dataset": {
            role: {
                dataset: list(index.tasks(role, dataset))
                for dataset in index.datasets(role)
            }
            for role in ("train", "heldout")
        },
        "episodes": [],
    }

    seed_counter = args.seed

    # 1. Training smoke test:
    # query and support both come from the training task's support split.
    for dataset in index.datasets("train"):
        for task_id in index.tasks("train", dataset):
            sampler = EpisodeSampler(
                index=index,
                config=EpisodeConfig(
                    task_role="train",
                    query_split="support",
                    support_split="support",
                    support_size=args.support_size,
                    support_with_replacement=True,
                    require_positive_support=True,
                ),
                seed=seed_counter,
            )
            seed_counter += 1
            episode = sampler.sample(task_id=task_id)
            result = validate_episode(
                episode,
                support_size=args.support_size,
            )
            result["stage"] = "training_smoke_test"
            report["episodes"].append(result)

    # 2. Training-task validation:
    # support from support split, query from development split.
    for dataset in index.datasets("train"):
        for task_id in index.tasks("train", dataset):
            sampler = EpisodeSampler(
                index=index,
                config=EpisodeConfig(
                    task_role="train",
                    query_split="dev",
                    support_split="support",
                    support_size=args.support_size,
                    support_with_replacement=True,
                    require_positive_support=True,
                ),
                seed=seed_counter,
            )
            seed_counter += 1
            episode = sampler.sample(task_id=task_id)
            result = validate_episode(
                episode,
                support_size=args.support_size,
            )
            result["stage"] = "training_task_validation"
            report["episodes"].append(result)

    # 3. Held-out development and test:
    # support always comes from support split.
    for query_split in ("dev", "test"):
        for dataset in index.datasets("heldout"):
            for task_id in index.tasks("heldout", dataset):
                sampler = EpisodeSampler(
                    index=index,
                    config=EpisodeConfig(
                        task_role="heldout",
                        query_split=query_split,
                        support_split="support",
                        support_size=args.support_size,
                        support_with_replacement=True,
                        require_positive_support=True,
                    ),
                    seed=seed_counter,
                )
                seed_counter += 1
                episode = sampler.sample(task_id=task_id)
                result = validate_episode(
                    episode,
                    support_size=args.support_size,
                )
                result["stage"] = (
                    f"heldout_{query_split}_smoke_test"
                )
                report["episodes"].append(result)

    # 4. Verify DataLoader adds the expected batch dimension.
    iterable_dataset = EpisodicSegmentationDataset(
        index=index,
        config=EpisodeConfig(
            task_role="train",
            query_split="support",
            support_split="support",
            support_size=args.support_size,
            support_with_replacement=True,
            require_positive_support=True,
        ),
        episodes_per_epoch=2,
        base_seed=args.seed,
    )
    loader = DataLoader(
        iterable_dataset,
        batch_size=1,
        num_workers=0,
    )
    batch = next(iter(loader))

    expected_batch_shapes = {
        "query_image": (1, 1, 128, 128),
        "query_mask": (1, 1, 128, 128),
        "support_images": (
            1,
            args.support_size,
            1,
            128,
            128,
        ),
        "support_masks": (
            1,
            args.support_size,
            1,
            128,
            128,
        ),
    }

    for key, expected_shape in expected_batch_shapes.items():
        actual_shape = tuple(batch[key].shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"DataLoader batch {key} has shape {actual_shape}, "
                f"expected {expected_shape}."
            )

    report["dataloader_batch_shapes"] = {
        key: list(batch[key].shape)
        for key in expected_batch_shapes
    }
    report["episode_count"] = len(report["episodes"])

    output_path = (
        project_root
        / "outputs"
        / "episodic_pipeline_validation.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== EPISODIC PIPELINE VALIDATION PASSED ===")
    print(f"Manifest: {manifest_path}")
    print(f"Episodes checked: {len(report['episodes'])}")
    print("DataLoader shapes:")
    for key, shape in report["dataloader_batch_shapes"].items():
        print(f"  {key}: {shape}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
