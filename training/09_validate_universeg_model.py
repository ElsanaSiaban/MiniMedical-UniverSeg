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
from universeg_model_v1 import (
    UniverSegConfig,
    UniverSegFromScratch,
    build_reduced_universeg,
    count_trainable_parameters,
)


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 42
DEFAULT_SUPPORT_SIZE = 4
DEFAULT_FEATURE_WIDTH = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the complete from-scratch UniverSeg encoder-decoder "
            "before implementing losses or training."
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
        help="CPU smoke-test support size. Default: 4",
    )
    parser.add_argument(
        "--feature-width",
        type=int,
        default=DEFAULT_FEATURE_WIDTH,
        help="CPU smoke-test width. Default: 16",
    )
    return parser.parse_args()


def assert_close(
    first: torch.Tensor,
    second: torch.Tensor,
    description: str,
    atol: float = 2e-5,
    rtol: float = 1e-5,
) -> float:
    error = float(
        torch.max(torch.abs(first - second)).item()
    )

    if not torch.allclose(
        first,
        second,
        atol=atol,
        rtol=rtol,
    ):
        raise RuntimeError(
            f"{description} failed; max error={error:.8g}."
        )

    return error


def gradient_norm(
    value: torch.Tensor | None,
    name: str,
) -> float:
    if value is None:
        raise RuntimeError(
            f"No gradient was produced for {name}."
        )
    if not torch.isfinite(value).all():
        raise RuntimeError(
            f"Non-finite gradient detected for {name}."
        )

    norm = float(value.norm().item())
    if norm <= 0:
        raise RuntimeError(
            f"Zero gradient detected for {name}."
        )

    return norm


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    project_root = args.project_root.expanduser().resolve()

    report: dict[str, Any] = {
        "status": "passed",
        "seed": args.seed,
        "device": "cpu",
        "support_size": args.support_size,
        "feature_width": args.feature_width,
        "tests": {},
    }

# 1. Complete forward pass and internal scale trace.

    model = build_reduced_universeg(
        feature_width=args.feature_width
    )
    model.eval()

    target = torch.randn(
        1,
        1,
        128,
        128,
    )
    support_images = torch.randn(
        1,
        args.support_size,
        1,
        128,
        128,
    )
    support_labels = torch.randint(
        low=0,
        high=2,
        size=(
            1,
            args.support_size,
            1,
            128,
            128,
        ),
    ).float()

    with torch.no_grad():
        logits, feature_trace = model(
            target,
            support_images,
            support_labels,
            return_features=True,
        )

    if tuple(logits.shape) != (1, 1, 128, 128):
        raise RuntimeError(
            f"Wrong model output shape: {tuple(logits.shape)}."
        )
    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "Model output contains non-finite values."
        )

    expected_encoder_spatial = [
        (128, 128),
        (64, 64),
        (32, 32),
        (16, 16),
    ]
    actual_encoder_spatial = [
        tuple(shape[-2:])
        for shape in feature_trace["encoder_target"]
    ]
    if actual_encoder_spatial != expected_encoder_spatial:
        raise RuntimeError(
            "Unexpected encoder spatial trace: "
            f"{actual_encoder_spatial}"
        )

    expected_decoder_spatial = [
        (32, 32),
        (64, 64),
        (128, 128),
    ]
    actual_decoder_spatial = [
        tuple(shape[-2:])
        for shape in feature_trace["decoder_target"]
    ]
    if actual_decoder_spatial != expected_decoder_spatial:
        raise RuntimeError(
            "Unexpected decoder spatial trace: "
            f"{actual_decoder_spatial}"
        )

    report["tests"]["forward_and_feature_trace"] = {
        "output_shape": list(logits.shape),
        "finite": True,
        "feature_trace": {
            key: [list(shape) for shape in shapes]
            for key, shapes in feature_trace.items()
        },
    }

# 2. Final prediction must be invariant to support ordering.

    permutation = torch.randperm(args.support_size)

    with torch.no_grad():
        original_output = model(
            target,
            support_images,
            support_labels,
        )
        permuted_output = model(
            target,
            support_images[:, permutation],
            support_labels[:, permutation],
        )

    permutation_error = assert_close(
        original_output,
        permuted_output,
        "Final prediction support-order invariance",
    )

    report["tests"]["support_order_invariance"] = {
        "permutation": permutation.tolist(),
        "maximum_absolute_error": permutation_error,
    }

# 3. Variable support sizes.

    variable_support_results: dict[str, Any] = {}

    with torch.no_grad():
        for support_size in (1, 3, 7):
            variable_images = torch.randn(
                1,
                support_size,
                1,
                64,
                64,
            )
            variable_labels = torch.randint(
                0,
                2,
                (
                    1,
                    support_size,
                    1,
                    64,
                    64,
                ),
            ).float()
            variable_target = torch.randn(
                1,
                1,
                64,
                64,
            )

            variable_output = model(
                variable_target,
                variable_images,
                variable_labels,
            )

            if tuple(variable_output.shape) != (
                1,
                1,
                64,
                64,
            ):
                raise RuntimeError(
                    f"Variable-support output error at S={support_size}: "
                    f"{tuple(variable_output.shape)}"
                )

            variable_support_results[str(support_size)] = {
                "output_shape": list(variable_output.shape),
                "finite": bool(
                    torch.isfinite(variable_output).all().item()
                ),
            }

    report["tests"]["variable_support_sizes"] = (
        variable_support_results
    )

# 4. Full backward pass through every encoder/decoder level.
# Use smaller 32x32 input to limit my CPU memory.

    gradient_model = build_reduced_universeg(
        feature_width=max(4, args.feature_width // 2)
    )
    gradient_model.train()

    gradient_target = torch.randn(
        1,
        1,
        32,
        32,
        requires_grad=True,
    )
    gradient_support_images = torch.randn(
        1,
        2,
        1,
        32,
        32,
        requires_grad=True,
    )
    gradient_support_labels = torch.randint(
        0,
        2,
        (1, 2, 1, 32, 32),
    ).float()

    gradient_logits = gradient_model(
        gradient_target,
        gradient_support_images,
        gradient_support_labels,
    )
    loss = (
        gradient_logits.square().mean()
        + gradient_logits.abs().mean()
    )
    loss.backward()

    gradient_report = {
        "target_input": gradient_norm(
            gradient_target.grad,
            "target input",
        ),
        "support_input": gradient_norm(
            gradient_support_images.grad,
            "support image input",
        ),
        "first_encoder_cross": gradient_norm(
            gradient_model.encoder_blocks[
                0
            ].cross.cross_conv.conv.weight.grad,
            "first encoder cross weight",
        ),
        "bottleneck_cross": gradient_norm(
            gradient_model.encoder_blocks[
                -1
            ].cross.cross_conv.conv.weight.grad,
            "bottleneck cross weight",
        ),
        "first_decoder_cross": gradient_norm(
            gradient_model.decoder_blocks[
                0
            ].cross.cross_conv.conv.weight.grad,
            "first decoder cross weight",
        ),
        "last_decoder_cross": gradient_norm(
            gradient_model.decoder_blocks[
                -1
            ].cross.cross_conv.conv.weight.grad,
            "last decoder cross weight",
        ),
        "output_conv": gradient_norm(
            gradient_model.output_conv.weight.grad,
            "output convolution weight",
        ),
    }
    report["tests"]["full_backward"] = gradient_report

# 5. Parameter counts for reduced and released-width topology.

    reduced_parameters = count_trainable_parameters(model)

    full_width_model = UniverSegFromScratch(
        UniverSegConfig(
            encoder_blocks=(64, 64, 64, 64)
        )
    )
    full_width_parameters = count_trainable_parameters(
        full_width_model
    )

    if full_width_parameters <= reduced_parameters:
        raise RuntimeError(
            "The 64-channel model should have more parameters than "
            "the reduced model."
        )

    report["tests"]["parameter_counts"] = {
        "reduced_width": args.feature_width,
        "reduced_trainable_parameters": reduced_parameters,
        "released_width": 64,
        "released_topology_trainable_parameters": (
            full_width_parameters
        ),
    }

    del full_width_model

# 6. Integration with a real manifest-backed episode.

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

    real_target = episode.query_image.unsqueeze(0)
    real_support_images = episode.support_images.unsqueeze(0)
    real_support_labels = episode.support_masks.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        real_logits = model(
            real_target,
            real_support_images,
            real_support_labels,
        )
        real_probabilities = torch.sigmoid(real_logits)

    if tuple(real_logits.shape) != (1, 1, 128, 128):
        raise RuntimeError(
            "Real-episode output shape mismatch: "
            f"{tuple(real_logits.shape)}"
        )
    if not torch.isfinite(real_logits).all():
        raise RuntimeError(
            "Real-episode logits contain non-finite values."
        )
    if (
        real_probabilities.min() < 0
        or real_probabilities.max() > 1
    ):
        raise RuntimeError(
            "Sigmoid probabilities are outside [0,1]."
        )

    report["tests"]["real_episode"] = {
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
        "target_shape": list(real_target.shape),
        "support_image_shape": list(
            real_support_images.shape
        ),
        "support_label_shape": list(
            real_support_labels.shape
        ),
        "logit_shape": list(real_logits.shape),
        "logit_minimum": float(real_logits.min().item()),
        "logit_maximum": float(real_logits.max().item()),
        "probability_minimum": float(
            real_probabilities.min().item()
        ),
        "probability_maximum": float(
            real_probabilities.max().item()
        ),
    }

    output_path = (
        project_root
        / "outputs"
        / "universeg_model_validation.json"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n FULL UNIVERSEG MODEL VALIDATION PASSED ")
    print("Output shape:", list(logits.shape))
    print(
        "Encoder scales:",
        actual_encoder_spatial,
    )
    print(
        "Decoder scales:",
        actual_decoder_spatial,
    )
    print(
        "Support-order max error:",
        permutation_error,
    )
    print(
        "Reduced parameters:",
        reduced_parameters,
    )
    print(
        "64-channel topology parameters:",
        full_width_parameters,
    )
    print(
        "Real episode output:",
        list(real_logits.shape),
    )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
