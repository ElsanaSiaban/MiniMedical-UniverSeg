from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from episodic_dataset import (
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
DEFAULT_SEED = 42
DEFAULT_SUPPORT_SIZE = 2
DEFAULT_FEATURE_WIDTH = 8
DEFAULT_LEARNING_RATE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Soft Dice loss and one real Adam optimization step "
            "before starting episodic training."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--support-size",
        type=int,
        default=DEFAULT_SUPPORT_SIZE,
    )
    parser.add_argument(
        "--feature-width",
        type=int,
        default=DEFAULT_FEATURE_WIDTH,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    return parser.parse_args()


def require_finite_scalar(
    value: torch.Tensor,
    name: str,
) -> float:
    if value.ndim != 0:
        raise RuntimeError(
            f"{name} must be scalar, received shape "
            f"{tuple(value.shape)}."
        )
    if not torch.isfinite(value):
        raise RuntimeError(f"{name} is not finite.")

    return float(value.item())


def require_positive_gradient(
    gradient: torch.Tensor | None,
    name: str,
) -> float:
    if gradient is None:
        raise RuntimeError(
            f"No gradient was produced for {name}."
        )
    if not torch.isfinite(gradient).all():
        raise RuntimeError(
            f"Non-finite gradient detected for {name}."
        )

    norm = float(gradient.norm().item())
    if norm <= 0:
        raise RuntimeError(
            f"Zero gradient detected for {name}."
        )

    return norm


def main() -> None:
    args = parse_args()

    if args.support_size < 1:
        raise ValueError("support-size must be positive.")
    if args.feature_width < 1:
        raise ValueError("feature-width must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive.")

    torch.manual_seed(args.seed)
    torch.set_num_threads(
        max(1, min(torch.get_num_threads(), 4))
    )

    project_root = args.project_root.expanduser().resolve()
    criterion = SoftDiceLoss(
        smooth=1e-5,
        reduction="mean",
    )

    report: dict[str, Any] = {
        "status": "passed",
        "seed": args.seed,
        "device": "cpu",
        "support_size": args.support_size,
        "feature_width": args.feature_width,
        "learning_rate": args.learning_rate,
        "tests": {},
    }

# 1. Near-perfect prediction.

    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 5:13] = 1.0

    perfect_logits = torch.where(
        target > 0,
        torch.tensor(20.0),
        torch.tensor(-20.0),
    )

    perfect_score = soft_dice_score_from_logits(
        perfect_logits,
        target,
    )
    perfect_loss = criterion(
        perfect_logits,
        target,
    )
    perfect_hard = hard_dice_score_from_logits(
        perfect_logits,
        target,
    )

    if perfect_score < 0.9999:
        raise RuntimeError(
            f"Perfect soft Dice is too low: {perfect_score.item()}."
        )
    if perfect_loss > 1e-4:
        raise RuntimeError(
            f"Perfect Dice loss is too high: {perfect_loss.item()}."
        )
    if perfect_hard < 0.9999:
        raise RuntimeError(
            f"Perfect hard Dice is too low: {perfect_hard.item()}."
        )

    report["tests"]["perfect_prediction"] = {
        "soft_dice": float(perfect_score.item()),
        "loss": float(perfect_loss.item()),
        "hard_dice": float(perfect_hard.item()),
    }

# 2. Opposite prediction.

    opposite_logits = -perfect_logits
    opposite_score = soft_dice_score_from_logits(
        opposite_logits,
        target,
    )
    opposite_loss = criterion(
        opposite_logits,
        target,
    )
    opposite_hard = hard_dice_score_from_logits(
        opposite_logits,
        target,
    )

    if opposite_score > 1e-3:
        raise RuntimeError(
            f"Opposite soft Dice is unexpectedly high: "
            f"{opposite_score.item()}."
        )
    if opposite_loss < 0.999:
        raise RuntimeError(
            f"Opposite Dice loss is unexpectedly low: "
            f"{opposite_loss.item()}."
        )
    if opposite_hard > 1e-3:
        raise RuntimeError(
            f"Opposite hard Dice is unexpectedly high: "
            f"{opposite_hard.item()}."
        )

    report["tests"]["opposite_prediction"] = {
        "soft_dice": float(opposite_score.item()),
        "loss": float(opposite_loss.item()),
        "hard_dice": float(opposite_hard.item()),
    }

# 3. Empty target with an almost-empty prediction.

    empty_target = torch.zeros(1, 1, 16, 16)
    empty_logits = torch.full_like(
        empty_target,
        -30.0,
    )

    empty_score = soft_dice_score_from_logits(
        empty_logits,
        empty_target,
    )
    empty_hard = hard_dice_score_from_logits(
        empty_logits,
        empty_target,
    )

    if empty_score < 0.99:
        raise RuntimeError(
            "Empty-empty soft Dice should be close to one, "
            f"received {empty_score.item()}."
        )
    if empty_hard < 0.9999:
        raise RuntimeError(
            "Empty-empty hard Dice should equal one."
        )

    report["tests"]["empty_target"] = {
        "soft_dice": float(empty_score.item()),
        "hard_dice": float(empty_hard.item()),
    }

# 4. Invalid pixels must not affect the score.

    masked_target = torch.zeros(1, 1, 8, 8)
    masked_target[:, :, 2:6, 2:6] = 1.0

    valid_mask = torch.zeros_like(masked_target)
    valid_mask[:, :, 1:7, 1:7] = 1.0

    masked_logits = torch.where(
        masked_target > 0,
        torch.tensor(20.0),
        torch.tensor(-20.0),
    )
    masked_logits[:, :, 0, :] = 20.0
    masked_logits[:, :, -1, :] = 20.0
    masked_logits[:, :, :, 0] = 20.0
    masked_logits[:, :, :, -1] = 20.0

    valid_score = soft_dice_score_from_logits(
        masked_logits,
        masked_target,
        valid_mask=valid_mask,
    )

    if valid_score < 0.9999:
        raise RuntimeError(
            "Invalid-region disagreements affected Dice: "
            f"{valid_score.item()}."
        )

    report["tests"]["valid_mask"] = {
        "soft_dice": float(valid_score.item()),
    }

# 5. Standalone loss gradient.

    gradient_logits = torch.randn(
        2,
        1,
        12,
        10,
        requires_grad=True,
    )
    gradient_targets = torch.randint(
        0,
        2,
        (2, 1, 12, 10),
    ).float()

    gradient_loss = criterion(
        gradient_logits,
        gradient_targets,
    )
    gradient_loss.backward()

    standalone_gradient_norm = require_positive_gradient(
        gradient_logits.grad,
        "standalone logits",
    )

    report["tests"]["standalone_gradient"] = {
        "loss": float(gradient_loss.item()),
        "gradient_norm": standalone_gradient_norm,
    }

# 6. One real optimization step.

    index = ManifestIndex(
        manifest_path=project_root / "data" / "processed" / "manifest.csv",
        processed_root=project_root / "data" / "processed",
    )
    sampler = EpisodeSampler(
        index=index,
        config=EpisodeConfig(
            task_role="train",
            query_split="support",
            support_split="support",
            support_size=args.support_size,
            support_with_replacement=False,
            require_positive_support=True,
        ),
        seed=args.seed,
    )
    episode = sampler.sample(
        task_id="wbc1_whole_cell"
    )

    query_image = episode.query_image.unsqueeze(0)
    query_mask = episode.query_mask.unsqueeze(0)
    query_valid_mask = episode.query_valid_mask.unsqueeze(0)
    support_images = episode.support_images.unsqueeze(0)
    support_masks = episode.support_masks.unsqueeze(0)

    model = build_reduced_universeg(
        feature_width=args.feature_width
    )
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    tracked_parameter = (
        model.encoder_blocks[
            0
        ].cross.cross_conv.conv.weight
    )
    parameter_before = tracked_parameter.detach().clone()

    optimizer.zero_grad(set_to_none=True)

    logits_before = model(
        query_image,
        support_images,
        support_masks,
    )
    loss_before_tensor = criterion(
        logits_before,
        query_mask,
        valid_mask=query_valid_mask,
    )
    loss_before = require_finite_scalar(
        loss_before_tensor,
        "loss_before",
    )

    loss_before_tensor.backward()

    gradient_norms = {
        "first_encoder_cross": require_positive_gradient(
            model.encoder_blocks[
                0
            ].cross.cross_conv.conv.weight.grad,
            "first encoder cross weight",
        ),
        "bottleneck_cross": require_positive_gradient(
            model.encoder_blocks[
                -1
            ].cross.cross_conv.conv.weight.grad,
            "bottleneck cross weight",
        ),
        "last_decoder_cross": require_positive_gradient(
            model.decoder_blocks[
                -1
            ].cross.cross_conv.conv.weight.grad,
            "last decoder cross weight",
        ),
        "output_conv": require_positive_gradient(
            model.output_conv.weight.grad,
            "output convolution weight",
        ),
    }

    optimizer.step()

    parameter_after = tracked_parameter.detach()
    parameter_update_norm = float(
        (parameter_after - parameter_before).norm().item()
    )
    if parameter_update_norm <= 0:
        raise RuntimeError(
            "Adam did not update the tracked model parameter."
        )

    optimizer_state_entries = len(optimizer.state)
    if optimizer_state_entries < 1:
        raise RuntimeError(
            "Adam optimizer state was not initialized."
        )

    model.eval()
    with torch.no_grad():
        logits_after = model(
            query_image,
            support_images,
            support_masks,
        )
        loss_after_tensor = criterion(
            logits_after,
            query_mask,
            valid_mask=query_valid_mask,
        )
        loss_after = require_finite_scalar(
            loss_after_tensor,
            "loss_after",
        )
        hard_dice_after = require_finite_scalar(
            hard_dice_score_from_logits(
                logits_after,
                query_mask,
                valid_mask=query_valid_mask,
            ),
            "hard_dice_after",
        )

    report["tests"]["real_adam_step"] = {
        "dataset": episode.metadata["dataset"],
        "task_id": episode.metadata["task_id"],
        "query_sample_id": (
            episode.metadata["query_sample_id"]
        ),
        "support_sample_ids": (
            episode.metadata["support_sample_ids"]
        ),
        "distinct_supports": (
            len(set(episode.metadata["support_sample_ids"]))
            == args.support_size
        ),
        "query_image_shape": list(query_image.shape),
        "query_mask_shape": list(query_mask.shape),
        "support_image_shape": list(support_images.shape),
        "support_mask_shape": list(support_masks.shape),
        "logit_shape": list(logits_before.shape),
        "loss_before": loss_before,
        "loss_after_same_episode": loss_after,
        "hard_dice_after": hard_dice_after,
        "gradient_norms": gradient_norms,
        "tracked_parameter_update_norm": (
            parameter_update_norm
        ),
        "optimizer_state_entries": (
            optimizer_state_entries
        ),
        "trainable_parameters": (
            count_trainable_parameters(model)
        ),
    }

    output_path = (
        project_root
        / "outputs"
        / "loss_training_step_validation.json"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        "\n SOFT DICE AND TRAINING STEP VALIDATION PASSED "
    )
    print(
        "Perfect/opposite soft Dice:",
        float(perfect_score.item()),
        float(opposite_score.item()),
    )
    print(
        "Empty-target soft Dice:",
        float(empty_score.item()),
    )
    print(
        "Real episode loss before/after:",
        loss_before,
        loss_after,
    )
    print(
        "Tracked parameter update norm:",
        parameter_update_norm,
    )
    print(
        "Trainable parameters:",
        count_trainable_parameters(model),
    )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
