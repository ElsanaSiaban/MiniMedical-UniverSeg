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
from universeg_core import CrossBlock, CrossOp


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 42
DEFAULT_SUPPORT_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the from-scratch CrossOp and CrossBlock "
            "implementations before constructing the full UniverSeg network."
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
    parser.add_argument(
        "--support-size",
        type=int,
        default=DEFAULT_SUPPORT_SIZE,
        help="Support size for the real-data smoke test. Default: 8",
    )
    return parser.parse_args()


def assert_close(
    first: torch.Tensor,
    second: torch.Tensor,
    description: str,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> float:
    error = float(torch.max(torch.abs(first - second)).item())

    if not torch.allclose(first, second, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"{description} failed; maximum absolute error={error:.8g}."
        )

    return error


def check_gradient(
    gradient: torch.Tensor | None,
    name: str,
) -> float:
    if gradient is None:
        raise RuntimeError(f"No gradient was produced for {name}.")
    if not torch.isfinite(gradient).all():
        raise RuntimeError(
            f"Non-finite gradient detected for {name}."
        )

    norm = float(gradient.norm().item())
    if norm <= 0:
        raise RuntimeError(
            f"Zero gradient norm detected for {name}."
        )

    return norm


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    project_root = args.project_root.expanduser().resolve()

    report: dict[str, Any] = {
        "status": "passed",
        "seed": args.seed,
        "device": "cpu",
        "support_size": args.support_size,
        "tests": {},
    }


# 1. CrossOp must return the raw interaction as support features and the support mean as the new query.
   
    cross_op = CrossOp(
        in_channels=(3, 4),
        out_channels=5,
        kernel_size=3,
        activation=None,
    )

    query = torch.randn(2, 1, 3, 17, 19)
    support = torch.randn(2, 6, 4, 17, 19)

    new_query, new_support = cross_op(query, support)
    raw_interaction = cross_op.cross_conv(
        query,
        support,
    ).squeeze(dim=1)

    expected_query = raw_interaction.mean(
        dim=1,
        keepdim=True,
    )

    support_error = assert_close(
        new_support,
        raw_interaction,
        "CrossOp support interaction identity",
    )
    mean_error = assert_close(
        new_query,
        expected_query,
        "CrossOp support-mean aggregation",
    )

    if tuple(new_query.shape) != (2, 1, 5, 17, 19):
        raise RuntimeError(
            f"Wrong CrossOp query shape: {tuple(new_query.shape)}"
        )
    if tuple(new_support.shape) != (2, 6, 5, 17, 19):
        raise RuntimeError(
            f"Wrong CrossOp support shape: {tuple(new_support.shape)}"
        )

    report["tests"]["crossop_mean_aggregation"] = {
        "query_output_shape": list(new_query.shape),
        "support_output_shape": list(new_support.shape),
        "support_identity_max_error": support_error,
        "mean_aggregation_max_error": mean_error,
    }

# 2. Permuting support elements must leave the aggregated query unchanged and must permute support outputs in the same way.

    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    permuted_query, permuted_support = cross_op(
        query,
        support[:, permutation],
    )

    query_permutation_error = assert_close(
        permuted_query,
        new_query,
        "CrossOp query permutation invariance",
    )
    support_permutation_error = assert_close(
        permuted_support,
        new_support[:, permutation],
        "CrossOp support permutation equivariance",
    )

    report["tests"]["crossop_permutation"] = {
        "permutation": permutation.tolist(),
        "query_invariance_max_error": query_permutation_error,
        "support_equivariance_max_error": (
            support_permutation_error
        ),
    }

    # 3. CrossBlock output shapes and separate query/support paths.
    
    block = CrossBlock(
        in_channels=(3, 4),
        cross_features=7,
        conv_features=9,
        kernel_size=3,
        cross_activation=None,
        conv_activation="LeakyReLU",
    )

    block_query, block_support = block(query, support)

    if tuple(block_query.shape) != (2, 1, 9, 17, 19):
        raise RuntimeError(
            f"Wrong CrossBlock query shape: {tuple(block_query.shape)}"
        )
    if tuple(block_support.shape) != (2, 6, 9, 17, 19):
        raise RuntimeError(
            f"Wrong CrossBlock support shape: "
            f"{tuple(block_support.shape)}"
        )

    if (
        block.query_conv.conv.weight.data_ptr()
        == block.support_conv.conv.weight.data_ptr()
    ):
        raise RuntimeError(
            "Query and support convolution paths share weights."
        )

    report["tests"]["crossblock_shapes_and_paths"] = {
        "query_output_shape": list(block_query.shape),
        "support_output_shape": list(block_support.shape),
        "query_support_weights_are_separate": True,
    }

# 4. CrossBlock must preserve the same permutation properties.
    
    permuted_block_query, permuted_block_support = block(
        query,
        support[:, permutation],
    )

    block_query_error = assert_close(
        permuted_block_query,
        block_query,
        "CrossBlock query permutation invariance",
        atol=2e-6,
        rtol=1e-5,
    )
    block_support_error = assert_close(
        permuted_block_support,
        block_support[:, permutation],
        "CrossBlock support permutation equivariance",
        atol=2e-6,
        rtol=1e-5,
    )

    report["tests"]["crossblock_permutation"] = {
        "query_invariance_max_error": block_query_error,
        "support_equivariance_max_error": block_support_error,
    }

    # 5. Gradient flow through query, support, cross convolution, query path, and support path.

    gradient_block = CrossBlock(
        in_channels=(2, 3),
        cross_features=5,
        conv_features=6,
    )

    gradient_query = torch.randn(
        1,
        1,
        2,
        23,
        21,
        requires_grad=True,
    )
    gradient_support = torch.randn(
        1,
        4,
        3,
        23,
        21,
        requires_grad=True,
    )

    gradient_query_output, gradient_support_output = gradient_block(
        gradient_query,
        gradient_support,
    )

    loss = (
        gradient_query_output.square().mean()
        + gradient_support_output.square().mean()
        + gradient_query_output.abs().mean()
        + gradient_support_output.abs().mean()
    )
    loss.backward()

    gradient_norms = {
        "query_input": check_gradient(
            gradient_query.grad,
            "query input",
        ),
        "support_input": check_gradient(
            gradient_support.grad,
            "support input",
        ),
        "cross_weight": check_gradient(
            gradient_block.cross.cross_conv.conv.weight.grad,
            "cross convolution weight",
        ),
        "query_path_weight": check_gradient(
            gradient_block.query_conv.conv.weight.grad,
            "query path weight",
        ),
        "support_path_weight": check_gradient(
            gradient_block.support_conv.conv.weight.grad,
            "support path weight",
        ),
    }

    report["tests"]["gradients"] = gradient_norms

# 6. Variable support sizes.

    support_size_shapes: dict[str, Any] = {}

    for support_size in (1, 3, 11):
        variable_support = torch.randn(
            1,
            support_size,
            4,
            15,
            13,
        )
        variable_query = torch.randn(
            1,
            1,
            3,
            15,
            13,
        )
        output_query, output_support = block(
            variable_query,
            variable_support,
        )

        expected_query_shape = (1, 1, 9, 15, 13)
        expected_support_shape = (
            1,
            support_size,
            9,
            15,
            13,
        )

        if tuple(output_query.shape) != expected_query_shape:
            raise RuntimeError(
                "Variable-S query shape mismatch for "
                f"S={support_size}: {tuple(output_query.shape)}"
            )
        if tuple(output_support.shape) != expected_support_shape:
            raise RuntimeError(
                "Variable-S support shape mismatch for "
                f"S={support_size}: {tuple(output_support.shape)}"
            )

        support_size_shapes[str(support_size)] = {
            "query": list(output_query.shape),
            "support": list(output_support.shape),
        }

    report["tests"]["variable_support_size"] = (
        support_size_shapes
    )

# 7. Q>1 must be rejected by CrossOp/CrossBlock.

    try:
        block(
            torch.randn(1, 2, 3, 17, 19),
            torch.randn(1, 6, 4, 17, 19),
        )
    except ValueError as exc:
        q_validation_message = str(exc)
    else:
        raise RuntimeError(
            "CrossBlock accepted Q>1 even though UniverSeg requires Q=1."
        )

    report["tests"]["query_count_validation"] = {
        "message": q_validation_message,
    }

# 8. First-stage parameter count.

    first_stage_block = CrossBlock(
        in_channels=(1, 2),
        cross_features=16,
        conv_features=16,
        cross_activation=None,
        conv_activation="LeakyReLU",
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in first_stage_block.parameters()
    )
    expected_parameter_count = 5088

    if parameter_count != expected_parameter_count:
        raise RuntimeError(
            f"First-stage parameter count={parameter_count}, "
            f"expected={expected_parameter_count}."
        )

    report["tests"]["first_stage_parameter_count"] = {
        "actual": parameter_count,
        "expected": expected_parameter_count,
    }


# 9. Real manifest-backed episode integration at 128x128.

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

    real_query = (
        episode.query_image
        .unsqueeze(0)
        .unsqueeze(1)
    )
    real_support = torch.cat(
        (
            episode.support_images,
            episode.support_masks,
        ),
        dim=1,
    ).unsqueeze(0)

    with torch.no_grad():
        real_query_output, real_support_output = (
            first_stage_block(
                real_query,
                real_support,
            )
        )

    expected_real_query_shape = (
        1,
        1,
        16,
        128,
        128,
    )
    expected_real_support_shape = (
        1,
        args.support_size,
        16,
        128,
        128,
    )

    if tuple(real_query_output.shape) != expected_real_query_shape:
        raise RuntimeError(
            "Real-episode query output shape mismatch: "
            f"{tuple(real_query_output.shape)}"
        )
    if tuple(real_support_output.shape) != expected_real_support_shape:
        raise RuntimeError(
            "Real-episode support output shape mismatch: "
            f"{tuple(real_support_output.shape)}"
        )
    if not torch.isfinite(real_query_output).all():
        raise RuntimeError(
            "Real-episode query output contains non-finite values."
        )
    if not torch.isfinite(real_support_output).all():
        raise RuntimeError(
            "Real-episode support output contains non-finite values."
        )

    report["tests"]["real_episode_first_stage"] = {
        "dataset": episode.metadata["dataset"],
        "task_id": episode.metadata["task_id"],
        "query_sample_id": (
            episode.metadata["query_sample_id"]
        ),
        "support_sample_ids": (
            episode.metadata["support_sample_ids"]
        ),
        "query_input_shape": list(real_query.shape),
        "support_input_shape": list(real_support.shape),
        "query_output_shape": list(real_query_output.shape),
        "support_output_shape": list(
            real_support_output.shape
        ),
        "finite": True,
        "distinct_supports": (
            len(set(episode.metadata["support_sample_ids"]))
            == args.support_size
        ),
    }

    output_path = (
        project_root
        / "outputs"
        / "crossblock_validation.json"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n  CROSSBLOCK VALIDATION PASSED  ")
    print(
        "CrossOp query/support:",
        list(new_query.shape),
        list(new_support.shape),
    )
    print(
        "CrossBlock query/support:",
        list(block_query.shape),
        list(block_support.shape),
    )
    print(
        "Query permutation max error:",
        block_query_error,
    )
    print(
        "Support permutation max error:",
        block_support_error,
    )
    print(
        "Real first-stage query/support:",
        list(real_query_output.shape),
        list(real_support_output.shape),
    )
    print(
        "First-stage parameters:",
        parameter_count,
    )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
