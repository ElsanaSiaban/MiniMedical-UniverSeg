from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from universeg_core import CrossBlock


BlockSpec = int | tuple[int, int]


def _normalize_block_spec(value: BlockSpec) -> tuple[int, int]:
    if isinstance(value, int):
        if value < 1:
            raise ValueError("Block feature width must be positive.")
        return value, value

    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, int) and item > 0 for item in value)
    ):
        return value

    raise TypeError(
        "Each block specification must be a positive integer or "
        "(cross_features, conv_features)."
    )


def _resize_set(
    tensor: torch.Tensor,
    size: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    if tensor.ndim != 5:
        raise ValueError(
            "Set tensor must have shape [B,N,C,H,W], "
            f"received {tuple(tensor.shape)}."
        )

    batch_size, set_size, channels, height, width = tensor.shape
    flat = tensor.reshape(
        batch_size * set_size,
        channels,
        height,
        width,
    )

    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        resized = F.interpolate(
            flat,
            size=size,
            mode=mode,
            align_corners=False,
        )
    else:
        resized = F.interpolate(
            flat,
            size=size,
            mode=mode,
        )

    return resized.reshape(
        batch_size,
        set_size,
        channels,
        size[0],
        size[1],
    )


def _max_pool_set(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 5:
        raise ValueError(
            "Set tensor must have shape [B,N,C,H,W], "
            f"received {tuple(tensor.shape)}."
        )

    batch_size, set_size, channels, height, width = tensor.shape
    flat = tensor.reshape(
        batch_size * set_size,
        channels,
        height,
        width,
    )
    pooled = F.max_pool2d(
        flat,
        kernel_size=2,
        stride=2,
    )

    output_height, output_width = pooled.shape[-2:]

    return pooled.reshape(
        batch_size,
        set_size,
        channels,
        output_height,
        output_width,
    )


@dataclass(frozen=True)
class UniverSegConfig:
    """
    Configuration for the from-scratch UniverSeg architecture.

    The released v1 structure corresponds to:
        encoder_blocks=(64, 64, 64, 64)
        decoder_blocks=None

    When decoder_blocks is None, it mirrors all encoder levels except
    the bottleneck, producing three decoder CrossBlocks for four encoder
    CrossBlocks.
    """

    encoder_blocks: tuple[BlockSpec, ...] = (64, 64, 64, 64)
    decoder_blocks: tuple[BlockSpec, ...] | None = None
    output_channels: int = 1
    kernel_size: int = 3
    negative_slope: float = 0.01

    def __post_init__(self) -> None:
        if len(self.encoder_blocks) < 2:
            raise ValueError(
                "At least two encoder blocks are required."
            )
        if self.output_channels < 1:
            raise ValueError(
                "output_channels must be positive."
            )
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be a positive odd integer."
            )

        normalized_encoder = tuple(
            _normalize_block_spec(value)
            for value in self.encoder_blocks
        )

        if self.decoder_blocks is None:
            normalized_decoder = tuple(
                normalized_encoder[-2::-1]
            )
        else:
            normalized_decoder = tuple(
                _normalize_block_spec(value)
                for value in self.decoder_blocks
            )

        expected_decoder_count = len(normalized_encoder) - 1
        if len(normalized_decoder) != expected_decoder_count:
            raise ValueError(
                "The decoder must contain exactly one fewer block than "
                "the encoder. "
                f"Expected {expected_decoder_count}, "
                f"received {len(normalized_decoder)}."
            )


class UniverSegFromScratch(nn.Module):
    """
    UNet-like UniverSeg architecture implemented without the official package.

    Inputs
    ------
    target_image:
        [B,1,H,W]
    support_images:
        [B,S,1,H,W]
    support_labels:
        [B,S,1,H,W]

    Output
    ------
    logits:
        [B,output_channels,H,W]

    The model returns logits. Apply torch.sigmoid only for probabilities or
    thresholded predictions; use logits directly in numerically stable loss
    functions when appropriate.
    """

    def __init__(
        self,
        config: UniverSegConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or UniverSegConfig()

        encoder_specs = tuple(
            _normalize_block_spec(value)
            for value in self.config.encoder_blocks
        )

        if self.config.decoder_blocks is None:
            decoder_specs = tuple(
                encoder_specs[-2::-1]
            )
        else:
            decoder_specs = tuple(
                _normalize_block_spec(value)
                for value in self.config.decoder_blocks
            )

        if len(decoder_specs) != len(encoder_specs) - 1:
            raise ValueError(
                "Decoder block count must equal encoder block count minus one."
            )

        self.encoder_specs = encoder_specs
        self.decoder_specs = decoder_specs

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        input_channels: int | tuple[int, int] = (1, 2)
        encoder_output_channels: list[int] = []

        for cross_features, conv_features in encoder_specs:
            block = CrossBlock(
                in_channels=input_channels,
                cross_features=cross_features,
                conv_features=conv_features,
                kernel_size=self.config.kernel_size,
                cross_activation=None,
                conv_activation="LeakyReLU",
                negative_slope=self.config.negative_slope,
            )
            self.encoder_blocks.append(block)
            encoder_output_channels.append(conv_features)
            input_channels = conv_features

        skip_channels = encoder_output_channels[-2::-1]
        current_channels = encoder_output_channels[-1]

        for (
            cross_features,
            conv_features,
        ), skip_features in zip(
            decoder_specs,
            skip_channels,
            strict=True,
        ):
            block = CrossBlock(
                in_channels=current_channels + skip_features,
                cross_features=cross_features,
                conv_features=conv_features,
                kernel_size=self.config.kernel_size,
                cross_activation=None,
                conv_activation="LeakyReLU",
                negative_slope=self.config.negative_slope,
            )
            self.decoder_blocks.append(block)
            current_channels = conv_features

        self.output_conv = nn.Conv2d(
            in_channels=current_channels,
            out_channels=self.config.output_channels,
            kernel_size=1,
            bias=True,
        )
        nn.init.kaiming_normal_(
            self.output_conv.weight,
            mode="fan_in",
            nonlinearity="linear",
        )
        nn.init.zeros_(self.output_conv.bias)

    @staticmethod
    def _validate_inputs(
        target_image: torch.Tensor,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> None:
        if target_image.ndim != 4:
            raise ValueError(
                "target_image must have shape [B,1,H,W], "
                f"received {tuple(target_image.shape)}."
            )
        if support_images.ndim != 5:
            raise ValueError(
                "support_images must have shape [B,S,1,H,W], "
                f"received {tuple(support_images.shape)}."
            )
        if support_labels.ndim != 5:
            raise ValueError(
                "support_labels must have shape [B,S,1,H,W], "
                f"received {tuple(support_labels.shape)}."
            )

        batch_size, target_channels, height, width = target_image.shape
        (
            support_batch,
            support_size,
            support_channels,
            support_height,
            support_width,
        ) = support_images.shape

        if target_channels != 1:
            raise ValueError(
                "target_image must have exactly one channel."
            )
        if support_channels != 1:
            raise ValueError(
                "support_images must have exactly one channel."
            )
        if support_labels.shape[2] != 1:
            raise ValueError(
                "support_labels must have exactly one channel."
            )
        if support_size < 1:
            raise ValueError(
                "The support set must contain at least one example."
            )
        if batch_size != support_batch:
            raise ValueError(
                "Target and support batch sizes differ."
            )
        if support_images.shape != support_labels.shape:
            raise ValueError(
                "support_images and support_labels must have identical "
                f"shapes; received {tuple(support_images.shape)} and "
                f"{tuple(support_labels.shape)}."
            )
        if (height, width) != (
            support_height,
            support_width,
        ):
            raise ValueError(
                "Target and support spatial dimensions differ."
            )
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                "For the four-encoder architecture, H and W must be "
                "divisible by 8."
            )
        if not (
            target_image.device
            == support_images.device
            == support_labels.device
        ):
            raise ValueError(
                "All inputs must be on the same device."
            )
        if not (
            target_image.dtype
            == support_images.dtype
            == support_labels.dtype
        ):
            raise ValueError(
                "All inputs must have the same dtype."
            )

    def forward(
        self,
        target_image: torch.Tensor,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, list[tuple[int, ...]]]]:
        self._validate_inputs(
            target_image,
            support_images,
            support_labels,
        )

        target = target_image.unsqueeze(dim=1)
        support = torch.cat(
            (support_images, support_labels),
            dim=2,
        )

        skip_connections: list[
            tuple[torch.Tensor, torch.Tensor]
        ] = []

        feature_trace: dict[str, list[tuple[int, ...]]] = {
            "encoder_target": [],
            "encoder_support": [],
            "decoder_target": [],
            "decoder_support": [],
        }

        for block_index, encoder_block in enumerate(
            self.encoder_blocks
        ):
            target, support = encoder_block(
                target,
                support,
            )

            feature_trace["encoder_target"].append(
                tuple(target.shape)
            )
            feature_trace["encoder_support"].append(
                tuple(support.shape)
            )

            if block_index == len(self.encoder_blocks) - 1:
                break

            skip_connections.append(
                (target, support)
            )
            target = _max_pool_set(target)
            support = _max_pool_set(support)

        for decoder_block in self.decoder_blocks:
            target_skip, support_skip = skip_connections.pop()

            target = _resize_set(
                target,
                size=target_skip.shape[-2:],
                mode="bilinear",
            )
            support = _resize_set(
                support,
                size=support_skip.shape[-2:],
                mode="bilinear",
            )

            target = torch.cat(
                (target, target_skip),
                dim=2,
            )
            support = torch.cat(
                (support, support_skip),
                dim=2,
            )

            target, support = decoder_block(
                target,
                support,
            )

            feature_trace["decoder_target"].append(
                tuple(target.shape)
            )
            feature_trace["decoder_support"].append(
                tuple(support.shape)
            )

        if skip_connections:
            raise RuntimeError(
                "Internal error: unused skip connections remain."
            )

        target_4d = target.squeeze(dim=1)
        logits = self.output_conv(target_4d)

        if return_features:
            return logits, feature_trace

        return logits


def build_reduced_universeg(
    feature_width: int = 16,
) -> UniverSegFromScratch:
    """
    CPU-friendly methodological reproduction.

    The block topology matches released UniverSeg v1, while feature_width
    may be reduced from the released model's 64 channels.
    """
    if feature_width < 1:
        raise ValueError("feature_width must be positive.")

    return UniverSegFromScratch(
        UniverSegConfig(
            encoder_blocks=(
                feature_width,
                feature_width,
                feature_width,
                feature_width,
            )
        )
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
