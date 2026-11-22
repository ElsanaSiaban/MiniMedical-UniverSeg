from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from universeg_core import (
    CrossConv2d,
    explicit_cross_conv_reference,
)


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unit-test the from-scratch CrossConv2d implementation before "
            "building CrossBlock."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Repository root. Defaults to this repository directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic test seed. Default: 42",
    )
    return parser.parse_args()


def assert_close(
    first: torch.Tensor,
    second: torch.Tensor,
    description: str,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    if not torch.allclose(first, second, atol=atol, rtol=rtol):
        maximum_error = float(
            torch.max(torch.abs(first - second)).item()
        )
        raise RuntimeError(
            f"{description} failed; maximum absolute error "
            f"was {maximum_error:.8g}."
        )


def expect_value_error(function, description: str) -> str:
    try:
        function()
    except ValueError as exc:
        return str(exc)

    raise RuntimeError(
        f"{description} did not raise the expected ValueError."
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    project_root = args.project_root.expanduser().resolve()

    report: dict[str, object] = {
        "status": "passed",
        "seed": args.seed,
        "device": "cpu",
        "tests": {},
    }

    layer = CrossConv2d(
        in_channels=(3, 4),
        out_channels=7,
        kernel_size=3,
        padding=1,
    )

    query = torch.randn(
        2,
        1,
        3,
        32,
        32,
        requires_grad=True,
    )
    support = torch.randn(
        2,
        5,
        4,
        32,
        32,
        requires_grad=True,
    )

    output = layer(query, support)
    expected_shape = (2, 1, 5, 7, 32, 32)

    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Wrong output shape: {tuple(output.shape)}, "
            f"expected {expected_shape}."
        )

    report["tests"]["shape"] = {
        "query": list(query.shape),
        "support": list(support.shape),
        "output": list(output.shape),
    }

    small_layer = CrossConv2d(
        in_channels=(2, 3),
        out_channels=4,
        kernel_size=3,
        padding=1,
    )
    small_query = torch.randn(2, 2, 2, 11, 13)
    small_support = torch.randn(2, 3, 3, 11, 13)

    vectorized = small_layer(
        small_query,
        small_support,
    )
    reference = explicit_cross_conv_reference(
        small_layer,
        small_query,
        small_support,
    )

    assert_close(
        vectorized,
        reference,
        "Vectorized-versus-loop equivalence",
    )

    report["tests"]["loop_equivalence"] = {
        "shape": list(vectorized.shape),
        "maximum_absolute_error": float(
            torch.max(torch.abs(vectorized - reference)).item()
        ),
    }

    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted_support = support[:, permutation]

    permuted_output = layer(query, permuted_support)
    expected_permuted_output = output[:, :, permutation]

    assert_close(
        permuted_output,
        expected_permuted_output,
        "Support permutation equivariance",
    )

    report["tests"]["support_permutation_equivariance"] = {
        "permutation": permutation.tolist(),
        "maximum_absolute_error": float(
            torch.max(
                torch.abs(
                    permuted_output
                    - expected_permuted_output
                )
            ).item()
        ),
    }

    scalar_loss = (
        output.square().mean()
        + output.abs().mean()
    )
    scalar_loss.backward()

    gradient_checks = {
        "query": query.grad,
        "support": support.grad,
        "weight": layer.conv.weight.grad,
        "bias": layer.conv.bias.grad,
    }

    gradient_norms: dict[str, float] = {}
    for name, gradient in gradient_checks.items():
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
                f"Zero gradient norm detected for {name}."
            )
        gradient_norms[name] = norm

    report["tests"]["gradients"] = gradient_norms

    first_layer = CrossConv2d(
        in_channels=(1, 2),
        out_channels=16,
        kernel_size=3,
        padding=1,
    )
    first_query = torch.randn(1, 1, 1, 128, 128)
    first_support = torch.randn(1, 8, 2, 128, 128)
    first_output = first_layer(
        first_query,
        first_support,
    )

    first_expected_shape = (1, 1, 8, 16, 128, 128)
    if tuple(first_output.shape) != first_expected_shape:
        raise RuntimeError(
            "First-stage CrossConv shape mismatch: "
            f"{tuple(first_output.shape)} versus "
            f"{first_expected_shape}."
        )

    report["tests"]["first_stage_smoke_test"] = {
        "query": list(first_query.shape),
        "support": list(first_support.shape),
        "output": list(first_output.shape),
        "finite": bool(
            torch.isfinite(first_output).all().item()
        ),
    }

    validation_messages = {
        "wrong_query_rank": expect_value_error(
            lambda: first_layer(
                torch.randn(1, 1, 128, 128),
                first_support,
            ),
            "Wrong query rank",
        ),
        "wrong_query_channels": expect_value_error(
            lambda: first_layer(
                torch.randn(1, 1, 2, 128, 128),
                first_support,
            ),
            "Wrong query channels",
        ),
        "wrong_support_channels": expect_value_error(
            lambda: first_layer(
                first_query,
                torch.randn(1, 8, 3, 128, 128),
            ),
            "Wrong support channels",
        ),
        "spatial_mismatch": expect_value_error(
            lambda: first_layer(
                first_query,
                torch.randn(1, 8, 2, 64, 128),
            ),
            "Spatial mismatch",
        ),
        "batch_mismatch": expect_value_error(
            lambda: first_layer(
                first_query,
                torch.randn(2, 8, 2, 128, 128),
            ),
            "Batch mismatch",
        ),
    }
    report["tests"]["input_validation"] = validation_messages

    parameter_count = sum(
        parameter.numel()
        for parameter in first_layer.parameters()
    )
    expected_parameter_count = (
        16 * (1 + 2) * 3 * 3 + 16
    )
    if parameter_count != expected_parameter_count:
        raise RuntimeError(
            f"Unexpected parameter count: {parameter_count}, "
            f"expected {expected_parameter_count}."
        )

    report["tests"]["parameter_count"] = {
        "actual": parameter_count,
        "expected": expected_parameter_count,
    }

    output_path = (
        project_root
        / "outputs"
        / "crossconv_validation.json"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n    CROSSCONV2D VALIDATION PASSED   ")
    print(
        "General output:",
        list(output.shape),
    )
    print(
        "First-stage output:",
        list(first_output.shape),
    )
    print(
        "Loop-equivalence max error:",
        report["tests"]["loop_equivalence"][
            "maximum_absolute_error"
        ],
    )
    print(
        "Permutation max error:",
        report["tests"][
            "support_permutation_equivariance"
        ]["maximum_absolute_error"],
    )
    print(
        "First-stage parameters:",
        parameter_count,
    )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
