from __future__ import annotations

from typing import Literal

import torch
from torch import nn


Reduction = Literal["none", "mean", "sum"]


def _validate_segmentation_tensors(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    if logits.ndim < 3:
        raise ValueError(
            "logits must have at least [B,C,spatial...] dimensions; "
            f"received {tuple(logits.shape)}."
        )
    if logits.shape != targets.shape:
        raise ValueError(
            "logits and targets must have identical shapes; "
            f"received {tuple(logits.shape)} and {tuple(targets.shape)}."
        )
    if not logits.is_floating_point():
        raise TypeError("logits must be floating-point.")
    if not targets.is_floating_point():
        raise TypeError("targets must be floating-point.")
    if logits.device != targets.device:
        raise ValueError("logits and targets must be on the same device.")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain non-finite values.")
    if not torch.isfinite(targets).all():
        raise ValueError("targets contain non-finite values.")
    if targets.min() < 0 or targets.max() > 1:
        raise ValueError("targets must lie in [0,1].")

    if valid_mask is None:
        return torch.ones_like(targets)

    if valid_mask.shape != targets.shape:
        raise ValueError(
            "valid_mask must have the same shape as targets; "
            f"received {tuple(valid_mask.shape)} and "
            f"{tuple(targets.shape)}."
        )
    if valid_mask.device != targets.device:
        raise ValueError(
            "valid_mask and targets must be on the same device."
        )
    if not valid_mask.is_floating_point():
        valid_mask = valid_mask.float()
    if not torch.isfinite(valid_mask).all():
        raise ValueError("valid_mask contains non-finite values.")
    if valid_mask.min() < 0 or valid_mask.max() > 1:
        raise ValueError("valid_mask must lie in [0,1].")

    return valid_mask.to(dtype=targets.dtype)


def soft_dice_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    smooth: float = 1e-5,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Soft Dice computed from logits.

    Dice is calculated independently for each [batch, channel] pair across
    all spatial dimensions, then reduced.

    The smoothing term is added to numerator and denominator. Consequently,
    an empty target paired with an almost-empty prediction receives a score
    close to 1, while false-positive foreground lowers the score.
    """
    if smooth <= 0:
        raise ValueError("smooth must be positive.")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(
            "reduction must be 'none', 'mean', or 'sum'."
        )

    valid = _validate_segmentation_tensors(
        logits,
        targets,
        valid_mask,
    )

    probabilities = torch.sigmoid(logits)
    probabilities = probabilities * valid
    masked_targets = targets * valid

    spatial_dims = tuple(range(2, logits.ndim))

    intersection = torch.sum(
        probabilities * masked_targets,
        dim=spatial_dims,
    )
    denominator = torch.sum(
        probabilities,
        dim=spatial_dims,
    ) + torch.sum(
        masked_targets,
        dim=spatial_dims,
    )

    score = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    if reduction == "none":
        return score
    if reduction == "sum":
        return score.sum()
    return score.mean()


class SoftDiceLoss(nn.Module):
    """One minus soft Dice score, computed directly from logits."""

    def __init__(
        self,
        smooth: float = 1e-5,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()

        if smooth <= 0:
            raise ValueError("smooth must be positive.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                "reduction must be 'none', 'mean', or 'sum'."
            )

        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        score = soft_dice_score_from_logits(
            logits=logits,
            targets=targets,
            valid_mask=valid_mask,
            smooth=self.smooth,
            reduction=self.reduction,
        )
        return 1.0 - score


@torch.no_grad()
def hard_dice_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
    smooth: float = 1e-5,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Hard Dice after sigmoid and thresholding.

    This is intended for evaluation, not optimization.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between 0 and 1.")
    if smooth <= 0:
        raise ValueError("smooth must be positive.")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(
            "reduction must be 'none', 'mean', or 'sum'."
        )

    valid = _validate_segmentation_tensors(
        logits,
        targets,
        valid_mask,
    )

    predictions = (
        torch.sigmoid(logits) >= threshold
    ).to(dtype=targets.dtype)

    predictions = predictions * valid
    masked_targets = targets * valid

    spatial_dims = tuple(range(2, logits.ndim))

    intersection = torch.sum(
        predictions * masked_targets,
        dim=spatial_dims,
    )
    denominator = torch.sum(
        predictions,
        dim=spatial_dims,
    ) + torch.sum(
        masked_targets,
        dim=spatial_dims,
    )

    score = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    if reduction == "none":
        return score
    if reduction == "sum":
        return score.sum()
    return score.mean()
