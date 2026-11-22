from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info


REQUIRED_COLUMNS = {
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
}

VALID_ROLES = {"train", "heldout"}
VALID_SPLITS = {"support", "dev", "test"}


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    task_id: str
    task_role: str
    dataset: str
    modality: str
    anatomy: str
    target_label: str
    image_path: Path
    mask_path: Path
    valid_mask_path: Path
    subject_id: str
    split: str
    has_foreground: bool


@dataclass(frozen=True)
class EpisodeConfig:
    task_role: str
    query_split: str
    support_split: str = "support"
    support_size: int = 8
    support_with_replacement: bool = True
    require_positive_support: bool = True

    def __post_init__(self) -> None:
        if self.task_role not in VALID_ROLES:
            raise ValueError(
                f"task_role must be one of {sorted(VALID_ROLES)}, "
                f"received {self.task_role!r}."
            )
        if self.query_split not in VALID_SPLITS:
            raise ValueError(
                f"query_split must be one of {sorted(VALID_SPLITS)}, "
                f"received {self.query_split!r}."
            )
        if self.support_split not in VALID_SPLITS:
            raise ValueError(
                f"support_split must be one of {sorted(VALID_SPLITS)}, "
                f"received {self.support_split!r}."
            )
        if self.support_size < 1:
            raise ValueError("support_size must be at least 1.")


@dataclass
class Episode:
    query_image: torch.Tensor
    query_mask: torch.Tensor
    query_valid_mask: torch.Tensor
    support_images: torch.Tensor
    support_masks: torch.Tensor
    support_valid_masks: torch.Tensor
    metadata: dict[str, Any]


class ManifestIndex:
    """Validated in-memory index over the merged MiniMedical manifest."""

    def __init__(
        self,
        manifest_path: Path,
        processed_root: Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        if processed_root is None:
            processed_root = self.manifest_path.parent

        self.processed_root = Path(processed_root).expanduser().resolve()
        if not self.processed_root.is_dir():
            raise FileNotFoundError(
                f"Processed root not found: {self.processed_root}"
            )

        self.records = self._load_records()
        self.by_role_dataset_task_split = self._build_index()
        self.datasets_by_role = self._build_dataset_index()
        self.tasks_by_role_dataset = self._build_task_index()
        self._validate_index()

    def _resolve_relative_path(self, value: str) -> Path:
        relative = Path(value)
        if relative.is_absolute():
            raise ValueError(
                "Manifest paths must be relative to processed/: "
                f"{value}"
            )

        resolved = (self.processed_root / relative).resolve()
        try:
            resolved.relative_to(self.processed_root)
        except ValueError as exc:
            raise ValueError(
                f"Manifest path escapes processed root: {value}"
            ) from exc

        return resolved

    def _load_records(self) -> list[ManifestRecord]:
        with self.manifest_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS - fieldnames)
            if missing:
                raise ValueError(
                    f"Manifest is missing required columns: {missing}"
                )

            rows = list(reader)

        if not rows:
            raise RuntimeError("The merged manifest is empty.")

        records: list[ManifestRecord] = []
        seen_sample_ids: set[str] = set()

        for row_number, row in enumerate(rows, start=2):
            sample_id = row["sample_id"]
            if sample_id in seen_sample_ids:
                raise ValueError(
                    f"Duplicate sample_id at row {row_number}: {sample_id}"
                )
            seen_sample_ids.add(sample_id)

            role = row["task_role"]
            split = row["split"]

            if role not in VALID_ROLES:
                raise ValueError(
                    f"Invalid role at row {row_number}: {role}"
                )
            if split not in VALID_SPLITS:
                raise ValueError(
                    f"Invalid split at row {row_number}: {split}"
                )

            has_foreground_value = int(row["has_foreground"])
            if has_foreground_value not in {0, 1}:
                raise ValueError(
                    f"Invalid has_foreground at row {row_number}: "
                    f"{row['has_foreground']}"
                )

            record = ManifestRecord(
                sample_id=sample_id,
                task_id=row["task_id"],
                task_role=role,
                dataset=row["dataset"],
                modality=row["modality"],
                anatomy=row["anatomy"],
                target_label=row["target_label"],
                image_path=self._resolve_relative_path(row["image_path"]),
                mask_path=self._resolve_relative_path(row["mask_path"]),
                valid_mask_path=self._resolve_relative_path(
                    row["valid_mask_path"]
                ),
                subject_id=row["subject_id"],
                split=split,
                has_foreground=bool(has_foreground_value),
            )

            for path in (
                record.image_path,
                record.mask_path,
                record.valid_mask_path,
            ):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Manifest references a missing file: {path}"
                    )

            records.append(record)

        return records

    def _build_index(
        self,
    ) -> dict[str, dict[str, dict[str, dict[str, list[ManifestRecord]]]]]:
        index: dict[
            str,
            dict[str, dict[str, dict[str, list[ManifestRecord]]]],
        ] = {}

        for record in self.records:
            (
                index
                .setdefault(record.task_role, {})
                .setdefault(record.dataset, {})
                .setdefault(record.task_id, {})
                .setdefault(record.split, [])
                .append(record)
            )

        return index

    def _build_dataset_index(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, tuple[str, ...]] = {}

        for role, datasets in self.by_role_dataset_task_split.items():
            result[role] = tuple(sorted(datasets))

        return result

    def _build_task_index(
        self,
    ) -> dict[str, dict[str, tuple[str, ...]]]:
        result: dict[str, dict[str, tuple[str, ...]]] = {}

        for role, datasets in self.by_role_dataset_task_split.items():
            result[role] = {}
            for dataset, tasks in datasets.items():
                result[role][dataset] = tuple(sorted(tasks))

        return result

    def _validate_index(self) -> None:
        for role, datasets in self.by_role_dataset_task_split.items():
            for dataset, tasks in datasets.items():
                for task_id, split_records in tasks.items():
                    if "support" not in split_records:
                        raise RuntimeError(
                            f"Task {task_id} has no support split."
                        )

                    for split, records in split_records.items():
                        if not records:
                            raise RuntimeError(
                                f"Empty index list: "
                                f"{role}/{dataset}/{task_id}/{split}"
                            )

                        sample_ids = [
                            record.sample_id for record in records
                        ]
                        if len(sample_ids) != len(set(sample_ids)):
                            raise RuntimeError(
                                "A sample appears more than once inside "
                                f"{task_id}/{split}."
                            )

    def datasets(self, role: str) -> tuple[str, ...]:
        try:
            return self.datasets_by_role[role]
        except KeyError as exc:
            raise KeyError(f"No datasets found for role={role!r}.") from exc

    def tasks(self, role: str, dataset: str) -> tuple[str, ...]:
        try:
            return self.tasks_by_role_dataset[role][dataset]
        except KeyError as exc:
            raise KeyError(
                f"No tasks found for role={role!r}, dataset={dataset!r}."
            ) from exc

    def records_for(
        self,
        role: str,
        dataset: str,
        task_id: str,
        split: str,
    ) -> list[ManifestRecord]:
        try:
            return self.by_role_dataset_task_split[
                role
            ][dataset][task_id][split]
        except KeyError as exc:
            raise KeyError(
                "No records found for "
                f"role={role!r}, dataset={dataset!r}, "
                f"task={task_id!r}, split={split!r}."
            ) from exc

    def locate_task(
        self,
        role: str,
        task_id: str,
    ) -> str:
        matches = [
            dataset
            for dataset in self.datasets(role)
            if task_id in self.tasks(role, dataset)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one dataset for task={task_id!r}, "
                f"role={role!r}; found {matches}."
            )
        return matches[0]


def load_grayscale_tensor(path: Path) -> torch.Tensor:
    """Load a grayscale PNG as float32 tensor [1,H,W] in [0,1]."""
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Could not load image: {path}") from exc

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image at {path}, got {array.shape}."
        )

    tensor = torch.from_numpy(array / 255.0).unsqueeze(0)
    return tensor.contiguous()


def load_binary_tensor(path: Path) -> torch.Tensor:
    """Load a binary mask as float32 tensor [1,H,W] with values {0,1}."""
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8)
    except Exception as exc:
        raise RuntimeError(f"Could not load mask: {path}") from exc

    unique_values = set(int(value) for value in np.unique(array))
    if not unique_values.issubset({0, 255}):
        raise ValueError(
            f"Mask must contain only 0/255 values: {path}; "
            f"found {sorted(unique_values)}."
        )

    tensor = torch.from_numpy((array > 0).astype(np.float32)).unsqueeze(0)
    return tensor.contiguous()


class EpisodeSampler:
    """
    Samples UniverSeg episodes from the merged manifest.

    Hierarchical task sampling:
      1. sample a dataset uniformly;
      2. sample a task uniformly within that dataset;
      3. sample one query;
      4. sample S supports from the same task.

    This prevents BUSI or ACDC from dominating merely because they contain
    more manifest rows.
    """

    def __init__(
        self,
        index: ManifestIndex,
        config: EpisodeConfig,
        seed: int = 42,
    ) -> None:
        self.index = index
        self.config = config
        self.rng = random.Random(seed)

    def _choose_dataset_and_task(
        self,
        task_id: str | None = None,
    ) -> tuple[str, str]:
        role = self.config.task_role

        if task_id is not None:
            dataset = self.index.locate_task(role, task_id)
            return dataset, task_id

        dataset = self.rng.choice(self.index.datasets(role))
        selected_task = self.rng.choice(
            self.index.tasks(role, dataset)
        )
        return dataset, selected_task

    def _sample_support_records(
        self,
        candidates: Sequence[ManifestRecord],
        query_record: ManifestRecord,
    ) -> list[ManifestRecord]:
        # Exclude every sample from the query subject, not only the exact
        # query row. This matters for ACDC, where each patient contributes
        # two annotated cardiac frames to the same task.
        available = [
            record
            for record in candidates
            if record.subject_id != query_record.subject_id
        ]

        if not available:
            raise RuntimeError(
                "No support candidate remains after excluding the query: "
                f"{query_record.sample_id}"
            )

        if (
            not self.config.support_with_replacement
            and len(available) < self.config.support_size
        ):
            raise RuntimeError(
                f"Need {self.config.support_size} distinct supports but "
                f"only {len(available)} are available."
            )

        def draw() -> list[ManifestRecord]:
            if self.config.support_with_replacement:
                return [
                    self.rng.choice(available)
                    for _ in range(self.config.support_size)
                ]
            return self.rng.sample(
                available,
                k=self.config.support_size,
            )

        selected = draw()

        if self.config.require_positive_support:
            for _ in range(100):
                if any(record.has_foreground for record in selected):
                    break
                selected = draw()
            else:
                positive_candidates = [
                    record for record in available if record.has_foreground
                ]
                if not positive_candidates:
                    raise RuntimeError(
                        "Task has no positive support candidate: "
                        f"{query_record.task_id}"
                    )
                selected[0] = self.rng.choice(positive_candidates)

        return selected

    def sample(
        self,
        task_id: str | None = None,
    ) -> Episode:
        dataset, selected_task = self._choose_dataset_and_task(task_id)
        role = self.config.task_role

        query_candidates = self.index.records_for(
            role=role,
            dataset=dataset,
            task_id=selected_task,
            split=self.config.query_split,
        )
        support_candidates = self.index.records_for(
            role=role,
            dataset=dataset,
            task_id=selected_task,
            split=self.config.support_split,
        )

        query_record = self.rng.choice(query_candidates)
        support_records = self._sample_support_records(
            support_candidates,
            query_record,
        )

        query_image = load_grayscale_tensor(query_record.image_path)
        query_mask = load_binary_tensor(query_record.mask_path)
        query_valid_mask = load_binary_tensor(
            query_record.valid_mask_path
        )

        support_images = torch.stack(
            [
                load_grayscale_tensor(record.image_path)
                for record in support_records
            ],
            dim=0,
        )
        support_masks = torch.stack(
            [
                load_binary_tensor(record.mask_path)
                for record in support_records
            ],
            dim=0,
        )
        support_valid_masks = torch.stack(
            [
                load_binary_tensor(record.valid_mask_path)
                for record in support_records
            ],
            dim=0,
        )

        if query_image.shape != query_mask.shape:
            raise RuntimeError(
                "Query image-mask shape mismatch: "
                f"{query_image.shape} versus {query_mask.shape}"
            )
        if support_images.shape != support_masks.shape:
            raise RuntimeError(
                "Support image-mask shape mismatch: "
                f"{support_images.shape} versus {support_masks.shape}"
            )

        metadata = {
            "task_role": role,
            "dataset": dataset,
            "task_id": selected_task,
            "query_split": self.config.query_split,
            "support_split": self.config.support_split,
            "query_sample_id": query_record.sample_id,
            "query_subject_id": query_record.subject_id,
            "query_has_foreground": query_record.has_foreground,
            "support_sample_ids": [
                record.sample_id for record in support_records
            ],
            "support_subject_ids": [
                record.subject_id for record in support_records
            ],
            "support_has_foreground": [
                record.has_foreground for record in support_records
            ],
        }

        return Episode(
            query_image=query_image,
            query_mask=query_mask,
            query_valid_mask=query_valid_mask,
            support_images=support_images,
            support_masks=support_masks,
            support_valid_masks=support_valid_masks,
            metadata=metadata,
        )


class EpisodicSegmentationDataset(IterableDataset[dict[str, Any]]):
    """
    Iterable PyTorch dataset producing a fixed number of episodes per epoch.

    Returned shapes before DataLoader batching:
      query_image:    [1,H,W]
      query_mask:     [1,H,W]
      support_images: [S,1,H,W]
      support_masks:  [S,1,H,W]

    With DataLoader(batch_size=1), shapes become:
      query_image:    [1,1,H,W]
      query_mask:     [1,1,H,W]
      support_images: [1,S,1,H,W]
      support_masks:  [1,S,1,H,W]
    """

    def __init__(
        self,
        index: ManifestIndex,
        config: EpisodeConfig,
        episodes_per_epoch: int,
        base_seed: int = 42,
    ) -> None:
        super().__init__()

        if episodes_per_epoch < 1:
            raise ValueError("episodes_per_epoch must be at least 1.")

        self.index = index
        self.config = config
        self.episodes_per_epoch = episodes_per_epoch
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = epoch

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        worker_count = worker_info.num_workers if worker_info is not None else 1

        seed = (
            self.base_seed
            + self.epoch * 1_000_003
            + worker_id * 10_007
        )
        sampler = EpisodeSampler(
            index=self.index,
            config=self.config,
            seed=seed,
        )

        for episode_index in range(
            worker_id,
            self.episodes_per_epoch,
            worker_count,
        ):
            episode = sampler.sample()

            yield {
                "query_image": episode.query_image,
                "query_mask": episode.query_mask,
                "query_valid_mask": episode.query_valid_mask,
                "support_images": episode.support_images,
                "support_masks": episode.support_masks,
                "support_valid_masks": episode.support_valid_masks,
                "metadata": {
                    **episode.metadata,
                    "episode_index": episode_index,
                    "epoch": self.epoch,
                },
            }
