from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import nn


Size2D = Union[int, Tuple[int, int]]


def _pair(value: Size2D) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value

    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        return value

    raise TypeError(
        "Expected an integer or a tuple of two integers, "
        f"received {value!r}."
    )


def _make_activation(
    name: str | None,
    negative_slope: float = 0.01,
) -> nn.Module:
    if name is None:
        return nn.Identity()

    normalized = name.lower()

    if normalized == "leakyrelu":
        return nn.LeakyReLU(
            negative_slope=negative_slope,
            inplace=False,
        )
    if normalized == "relu":
        return nn.ReLU(inplace=False)
    if normalized == "gelu":
        return nn.GELU()

    raise ValueError(
        "Unsupported activation. Expected one of "
        "None, 'LeakyReLU', 'ReLU', or 'GELU'; "
        f"received {name!r}."
    )


def _reset_conv2d(
    conv: nn.Conv2d,
    activation: str | None,
    negative_slope: float = 0.01,
) -> None:
    if activation is None:
        nn.init.kaiming_normal_(
            conv.weight,
            mode="fan_in",
            nonlinearity="linear",
        )
    elif activation.lower() == "leakyrelu":
        nn.init.kaiming_normal_(
            conv.weight,
            a=negative_slope,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
    elif activation.lower() == "relu":
        nn.init.kaiming_normal_(
            conv.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
    else:
        nn.init.kaiming_normal_(
            conv.weight,
            mode="fan_in",
            nonlinearity="linear",
        )

    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class CrossConv2d(nn.Module):
    """
    Pairwise 2D convolution between query and support feature sets.

    query:
        [B, Q, Cq, H, W]
    support:
        [B, S, Cs, H, W]
    output:
        [B, Q, S, Cout, Hout, Wout]

    Every query element is concatenated channel-wise with every support
    element. One shared Conv2d is applied to all query-support pairs.

    This implementation uses only native PyTorch operations. It does not
    import the official UniverSeg package or pretrained weights.
    """

    def __init__(
        self,
        in_channels: int | tuple[int, int],
        out_channels: int,
        kernel_size: Size2D = 3,
        stride: Size2D = 1,
        padding: Size2D = 0,
        dilation: Size2D = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        if isinstance(in_channels, int):
            query_channels = in_channels
            support_channels = in_channels
        elif (
            isinstance(in_channels, tuple)
            and len(in_channels) == 2
            and all(isinstance(value, int) for value in in_channels)
        ):
            query_channels, support_channels = in_channels
        else:
            raise TypeError(
                "in_channels must be an integer or a tuple "
                "(query_channels, support_channels)."
            )

        if query_channels < 1 or support_channels < 1:
            raise ValueError("Input channel counts must be positive.")
        if out_channels < 1:
            raise ValueError("out_channels must be positive.")
        if groups < 1:
            raise ValueError("groups must be positive.")

        concatenated_channels = query_channels + support_channels
        if concatenated_channels % groups != 0:
            raise ValueError(
                "The sum of query and support channels must be divisible "
                f"by groups. Got {concatenated_channels} and groups={groups}."
            )
        if out_channels % groups != 0:
            raise ValueError(
                "out_channels must be divisible by groups. "
                f"Got out_channels={out_channels}, groups={groups}."
            )

        self.query_channels = query_channels
        self.support_channels = support_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels=concatenated_channels,
            out_channels=out_channels,
            kernel_size=_pair(kernel_size),
            stride=_pair(stride),
            padding=_pair(padding),
            dilation=_pair(dilation),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def _validate_inputs(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> None:
        if query.ndim != 5:
            raise ValueError(
                "query must have shape [B,Q,Cq,H,W], "
                f"received {tuple(query.shape)}."
            )
        if support.ndim != 5:
            raise ValueError(
                "support must have shape [B,S,Cs,H,W], "
                f"received {tuple(support.shape)}."
            )

        batch_query, query_count, query_channels, height_q, width_q = (
            query.shape
        )
        batch_support, support_count, support_channels, height_s, width_s = (
            support.shape
        )

        if batch_query != batch_support:
            raise ValueError(
                "query and support batch sizes differ: "
                f"{batch_query} versus {batch_support}."
            )
        if query_count < 1:
            raise ValueError("query set dimension Q must be at least 1.")
        if support_count < 1:
            raise ValueError("support set dimension S must be at least 1.")
        if query_channels != self.query_channels:
            raise ValueError(
                "Wrong query channel count: "
                f"expected {self.query_channels}, received {query_channels}."
            )
        if support_channels != self.support_channels:
            raise ValueError(
                "Wrong support channel count: "
                f"expected {self.support_channels}, "
                f"received {support_channels}."
            )
        if (height_q, width_q) != (height_s, width_s):
            raise ValueError(
                "query and support spatial sizes differ: "
                f"{(height_q, width_q)} versus {(height_s, width_s)}."
            )
        if query.device != support.device:
            raise ValueError(
                "query and support must be on the same device."
            )
        if query.dtype != support.dtype:
            raise ValueError(
                "query and support must have the same dtype."
            )

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(query, support)

        batch_size, query_count, _, height, width = query.shape
        _, support_count, _, _, _ = support.shape

        query_grid = query.unsqueeze(2).expand(
            batch_size,
            query_count,
            support_count,
            self.query_channels,
            height,
            width,
        )
        support_grid = support.unsqueeze(1).expand(
            batch_size,
            query_count,
            support_count,
            self.support_channels,
            height,
            width,
        )

        paired = torch.cat(
            (query_grid, support_grid),
            dim=3,
        )

        flat_pairs = paired.reshape(
            batch_size * query_count * support_count,
            self.query_channels + self.support_channels,
            height,
            width,
        )

        flat_output = self.conv(flat_pairs)
        output_height, output_width = flat_output.shape[-2:]

        return flat_output.reshape(
            batch_size,
            query_count,
            support_count,
            self.out_channels,
            output_height,
            output_width,
        )


def explicit_cross_conv_reference(
    layer: CrossConv2d,
    query: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Slow nested-loop reference used only by unit tests."""
    layer._validate_inputs(query, support)

    batch_size, query_count = query.shape[:2]
    support_count = support.shape[1]
    batch_outputs: list[torch.Tensor] = []

    for batch_index in range(batch_size):
        query_outputs: list[torch.Tensor] = []

        for query_index in range(query_count):
            support_outputs: list[torch.Tensor] = []

            for support_index in range(support_count):
                paired = torch.cat(
                    (
                        query[
                            batch_index : batch_index + 1,
                            query_index,
                        ],
                        support[
                            batch_index : batch_index + 1,
                            support_index,
                        ],
                    ),
                    dim=1,
                )
                support_outputs.append(layer.conv(paired))

            query_outputs.append(
                torch.cat(support_outputs, dim=0).unsqueeze(0)
            )

        batch_outputs.append(
            torch.cat(query_outputs, dim=0).unsqueeze(0)
        )

    return torch.cat(batch_outputs, dim=0)


class SetConv2d(nn.Module):
    """
    Apply one shared Conv2d independently to every element of a set tensor.

    Input:
        [B, N, Cin, H, W]
    Output:
        [B, N, Cout, Hout, Wout]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2D = 3,
        activation: str | None = "LeakyReLU",
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()

        if in_channels < 1 or out_channels < 1:
            raise ValueError("Channel counts must be positive.")

        kernel = _pair(kernel_size)

        if kernel[0] % 2 == 0 or kernel[1] % 2 == 0:
            raise ValueError(
                "SetConv2d requires odd kernel sizes so spatial dimensions "
                "can be preserved."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_name = activation

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            padding=(kernel[0] // 2, kernel[1] // 2),
            padding_mode="zeros",
            bias=True,
        )
        self.activation = _make_activation(
            activation,
            negative_slope=negative_slope,
        )

        _reset_conv2d(
            self.conv,
            activation=activation,
            negative_slope=negative_slope,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 5:
            raise ValueError(
                "SetConv2d input must have shape [B,N,C,H,W], "
                f"received {tuple(tensor.shape)}."
            )

        batch_size, set_size, channels, height, width = tensor.shape

        if set_size < 1:
            raise ValueError("Set dimension N must be at least 1.")
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, received {channels}."
            )

        flat = tensor.reshape(
            batch_size * set_size,
            channels,
            height,
            width,
        )
        flat_output = self.activation(self.conv(flat))
        output_height, output_width = flat_output.shape[-2:]

        return flat_output.reshape(
            batch_size,
            set_size,
            self.out_channels,
            output_height,
            output_width,
        )


class CrossOp(nn.Module):
    """
    Cross interaction and support-mean query aggregation.

    The query set is required to contain exactly one element, matching the
    UniverSeg target/query representation.

    Inputs:
        query   [B,1,Cq,H,W]
        support [B,S,Cs,H,W]

    Outputs:
        new_query   [B,1,Ccross,H,W]
        new_support [B,S,Ccross,H,W]
    """

    def __init__(
        self,
        in_channels: int | tuple[int, int],
        out_channels: int,
        kernel_size: Size2D = 3,
        activation: str | None = None,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()

        kernel = _pair(kernel_size)
        if kernel[0] % 2 == 0 or kernel[1] % 2 == 0:
            raise ValueError(
                "CrossOp requires odd kernel sizes so spatial dimensions "
                "can be preserved."
            )

        self.out_channels = out_channels
        self.activation_name = activation

        self.cross_conv = CrossConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            padding=(kernel[0] // 2, kernel[1] // 2),
            bias=True,
        )
        self.activation = _make_activation(
            activation,
            negative_slope=negative_slope,
        )

        _reset_conv2d(
            self.cross_conv.conv,
            activation=activation,
            negative_slope=negative_slope,
        )

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 5:
            raise ValueError(
                "query must have shape [B,1,C,H,W], "
                f"received {tuple(query.shape)}."
            )
        if query.shape[1] != 1:
            raise ValueError(
                "CrossOp requires exactly one query element (Q=1), "
                f"received Q={query.shape[1]}."
            )

        interaction = self.cross_conv(
            query,
            support,
        ).squeeze(dim=1)

        interaction = self.activation(interaction)

        new_query = interaction.mean(
            dim=1,
            keepdim=True,
        )
        new_support = interaction

        return new_query, new_support


class CrossBlock(nn.Module):
    """
    UniverSeg CrossBlock implemented from scratch.

    1. CrossOp creates pairwise query-support interaction features.
    2. The query is updated by the mean across support interactions.
    3. Separate SetConv2d modules update query and support features.

    The query and support convolution paths intentionally do not share
    parameters.
    """

    def __init__(
        self,
        in_channels: int | tuple[int, int],
        cross_features: int,
        conv_features: int | None = None,
        kernel_size: Size2D = 3,
        cross_activation: str | None = None,
        conv_activation: str | None = "LeakyReLU",
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()

        if cross_features < 1:
            raise ValueError("cross_features must be positive.")

        output_features = (
            cross_features
            if conv_features is None
            else conv_features
        )
        if output_features < 1:
            raise ValueError("conv_features must be positive.")

        self.cross_features = cross_features
        self.conv_features = output_features

        self.cross = CrossOp(
            in_channels=in_channels,
            out_channels=cross_features,
            kernel_size=kernel_size,
            activation=cross_activation,
            negative_slope=negative_slope,
        )

        self.query_conv = SetConv2d(
            in_channels=cross_features,
            out_channels=output_features,
            kernel_size=kernel_size,
            activation=conv_activation,
            negative_slope=negative_slope,
        )
        self.support_conv = SetConv2d(
            in_channels=cross_features,
            out_channels=output_features,
            kernel_size=kernel_size,
            activation=conv_activation,
            negative_slope=negative_slope,
        )

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_interaction, support_interaction = self.cross(
            query,
            support,
        )

        new_query = self.query_conv(query_interaction)
        new_support = self.support_conv(support_interaction)

        return new_query, new_support
