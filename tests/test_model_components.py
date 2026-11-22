from __future__ import annotations

import torch

from universeg_core import CrossConv2d, explicit_cross_conv_reference
from universeg_losses import hard_dice_score_from_logits, soft_dice_score_from_logits
from universeg_model_v1 import build_reduced_universeg


def test_cross_conv_matches_reference_implementation() -> None:
    torch.manual_seed(7)
    layer = CrossConv2d(
        in_channels=(2, 3),
        out_channels=4,
        kernel_size=3,
        padding=1,
    )
    query = torch.randn(1, 2, 2, 8, 8)
    support = torch.randn(1, 3, 3, 8, 8)

    actual = layer(query, support)
    expected = explicit_cross_conv_reference(layer, query, support)

    assert actual.shape == (1, 2, 3, 4, 8, 8)
    torch.testing.assert_close(actual, expected)


def test_reduced_model_forward_shape() -> None:
    torch.manual_seed(11)
    model = build_reduced_universeg(feature_width=4)
    target = torch.randn(1, 1, 16, 16)
    support_images = torch.randn(1, 2, 1, 16, 16)
    support_labels = torch.randint(0, 2, (1, 2, 1, 16, 16)).float()

    with torch.no_grad():
        logits = model(target, support_images, support_labels)

    assert logits.shape == (1, 1, 16, 16)
    assert torch.isfinite(logits).all()


def test_dice_metrics_return_finite_values() -> None:
    logits = torch.zeros(2, 1, 8, 8)
    targets = torch.zeros_like(logits)

    soft_score = soft_dice_score_from_logits(logits, targets)
    hard_score = hard_dice_score_from_logits(logits, targets)

    assert torch.isfinite(soft_score)
    assert torch.isfinite(hard_score)
    assert 0.0 <= float(soft_score) <= 1.0
    assert 0.0 <= float(hard_score) <= 1.0
