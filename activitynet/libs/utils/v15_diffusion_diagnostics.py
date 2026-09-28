"""Geometry and optimization diagnostics for V15 boundary diffusion.

The helpers in this module are deliberately model-agnostic.  They compare the
current endpoint-residual representation with a validity-preserving
center/log-duration representation, summarize train/inference distribution
shift, and measure cosine conflicts between timestep-specific gradients.
"""

import math

import torch


def endpoint_to_center_log_residual(coarse, target, minimum_duration=1e-6):
    """Encode a target interval relative to a coarse interval.

    The first coordinate is the normalized center displacement and the second
    is the log duration ratio.  Any finite second coordinate decodes to a
    strictly positive duration.
    """
    if coarse.shape != target.shape or coarse.ndim != 2 \
            or coarse.shape[-1] != 2:
        raise ValueError("coarse and target must have matching shape [N, 2]")
    if minimum_duration <= 0.0:
        raise ValueError("minimum_duration must be positive")
    coarse_duration = (
        coarse[:, 1] - coarse[:, 0]).clamp_min(minimum_duration)
    target_duration = (
        target[:, 1] - target[:, 0]).clamp_min(minimum_duration)
    coarse_center = 0.5 * (coarse[:, 0] + coarse[:, 1])
    target_center = 0.5 * (target[:, 0] + target[:, 1])
    return torch.stack((
        (target_center - coarse_center) / coarse_duration,
        torch.log(target_duration / coarse_duration),
    ), dim=-1)


def center_log_residual_to_endpoint(
        coarse, residual, minimum_duration=1e-6):
    """Decode normalized center displacement and log duration ratio."""
    if coarse.shape != residual.shape or coarse.ndim != 2 \
            or coarse.shape[-1] != 2:
        raise ValueError("coarse and residual must have matching shape [N, 2]")
    if minimum_duration <= 0.0:
        raise ValueError("minimum_duration must be positive")
    coarse_duration = (
        coarse[:, 1] - coarse[:, 0]).clamp_min(minimum_duration)
    coarse_center = 0.5 * (coarse[:, 0] + coarse[:, 1])
    center = coarse_center + residual[:, 0] * coarse_duration
    duration = coarse_duration * torch.exp(residual[:, 1])
    return torch.stack((
        center - 0.5 * duration,
        center + 0.5 * duration,
    ), dim=-1)


def endpoint_residual(coarse, target, minimum_duration=1e-6):
    """Return the current DyFADet normalized start/end residual."""
    if coarse.shape != target.shape or coarse.ndim != 2 \
            or coarse.shape[-1] != 2:
        raise ValueError("coarse and target must have matching shape [N, 2]")
    duration = (coarse[:, 1] - coarse[:, 0]).clamp_min(
        minimum_duration)
    return (target - coarse) / duration[:, None]


def temporal_iou(candidate, target):
    """Vectorized temporal IoU for matching interval tensors."""
    if candidate.shape != target.shape or candidate.ndim != 2 \
            or candidate.shape[-1] != 2:
        raise ValueError(
            "candidate and target must have matching shape [N, 2]")
    intersection = (
        torch.minimum(candidate[:, 1], target[:, 1])
        - torch.maximum(candidate[:, 0], target[:, 0])
    ).clamp_min(0.0)
    union = (
        (candidate[:, 1] - candidate[:, 0]).clamp_min(0.0)
        + (target[:, 1] - target[:, 0]).clamp_min(0.0)
        - intersection
    ).clamp_min(1e-6)
    return intersection / union


def _safe_moment(values, order):
    centered = values - values.mean()
    std = values.std(unbiased=False)
    if float(std) <= 1e-12:
        return 0.0
    return float((centered / std).pow(order).mean())


def vector_distribution_summary(values, names=None):
    """Return robust per-coordinate statistics for a finite 2-D tensor."""
    if values.ndim != 2:
        raise ValueError("values must have shape [N, D]")
    if values.shape[0] == 0:
        raise ValueError("values must not be empty")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("values must be finite")
    dimension = values.shape[1]
    if names is None:
        names = ["dimension_{}".format(index) for index in range(dimension)]
    if len(names) != dimension:
        raise ValueError("names must match the value dimension")
    values = values.detach().float().cpu()
    probabilities = values.new_tensor(
        [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    coordinates = {}
    for index, name in enumerate(names):
        column = values[:, index]
        quantiles = torch.quantile(column, probabilities)
        coordinates[str(name)] = {
            "mean": float(column.mean()),
            "std": float(column.std(unbiased=False)),
            "mean_abs": float(column.abs().mean()),
            "maximum_abs": float(column.abs().max()),
            "skewness": _safe_moment(column, 3),
            "excess_kurtosis": _safe_moment(column, 4) - 3.0,
            "quantiles": {
                "{:.2f}".format(float(probability)): float(value)
                for probability, value in zip(
                    probabilities.tolist(), quantiles.tolist())
            },
        }
    if dimension == 1:
        correlation = [[1.0]]
    else:
        centered = values - values.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(values.shape[0], 1)
        scale = covariance.diag().clamp_min(1e-12).sqrt()
        correlation_tensor = (
            covariance / (scale[:, None] * scale[None, :])
        ).clamp(-1.0, 1.0)
        correlation = correlation_tensor.tolist()
    return {
        "count": int(values.shape[0]),
        "dimension": int(dimension),
        "coordinates": coordinates,
        "correlation": correlation,
    }


def compare_vector_distributions(reference, observed, quantile_points=101):
    """Measure a transparent train-to-inference residual distribution gap."""
    if reference.ndim != 2 or observed.ndim != 2 \
            or reference.shape[1] != observed.shape[1]:
        raise ValueError(
            "reference and observed must have matching feature dimensions")
    if reference.shape[0] == 0 or observed.shape[0] == 0:
        raise ValueError("both distributions must contain samples")
    if quantile_points < 3:
        raise ValueError("quantile_points must be at least three")
    reference = reference.detach().float().cpu()
    observed = observed.detach().float().cpu()
    probabilities = torch.linspace(0.0, 1.0, quantile_points)
    dimensions = []
    for index in range(reference.shape[1]):
        left = reference[:, index]
        right = observed[:, index]
        left_std = left.std(unbiased=False).clamp_min(1e-6)
        wasserstein = torch.mean(torch.abs(
            torch.quantile(left, probabilities)
            - torch.quantile(right, probabilities)))
        dimensions.append({
            "dimension": index,
            "standardized_mean_shift": float(
                (right.mean() - left.mean()) / left_std),
            "std_ratio": float(
                right.std(unbiased=False) / left_std),
            "quantile_wasserstein": float(wasserstein),
            "normalized_quantile_wasserstein": float(
                wasserstein / left_std),
        })
    return {
        "reference_count": int(reference.shape[0]),
        "observed_count": int(observed.shape[0]),
        "dimensions": dimensions,
        "mean_normalized_quantile_wasserstein": float(
            sum(item["normalized_quantile_wasserstein"]
                for item in dimensions) / len(dimensions)),
    }


def gradient_cosine_report(gradient_groups, labels=None):
    """Compute exact cosine similarities for timestep gradient groups.

    ``gradient_groups`` is a list of equally structured lists/tuples of
    parameter gradients.  ``None`` entries are treated as zero gradients.
    """
    if len(gradient_groups) < 2:
        raise ValueError("at least two gradient groups are required")
    parameter_count = len(gradient_groups[0])
    if any(len(group) != parameter_count for group in gradient_groups):
        raise ValueError("gradient groups must have matching structure")
    if labels is None:
        labels = [str(index) for index in range(len(gradient_groups))]
    if len(labels) != len(gradient_groups):
        raise ValueError("labels must match gradient groups")

    def dot(left, right):
        total = 0.0
        for first, second in zip(left, right):
            if first is None or second is None:
                continue
            total += float(torch.sum(
                first.detach().double().cpu()
                * second.detach().double().cpu()))
        return total

    norms = [math.sqrt(max(dot(group, group), 0.0))
             for group in gradient_groups]
    matrix = []
    off_diagonal = []
    for row, left in enumerate(gradient_groups):
        values = []
        for column, right in enumerate(gradient_groups):
            denominator = max(norms[row] * norms[column], 1e-12)
            cosine = dot(left, right) / denominator
            values.append(float(max(-1.0, min(1.0, cosine))))
            if row < column:
                off_diagonal.append(values[-1])
        matrix.append(values)
    return {
        "labels": [str(label) for label in labels],
        "cosine_matrix": matrix,
        "gradient_norms": norms,
        "off_diagonal_mean": float(
            sum(off_diagonal) / len(off_diagonal)),
        "negative_pair_fraction": float(
            sum(value < 0.0 for value in off_diagonal)
            / len(off_diagonal)),
        "minimum_pair_cosine": float(min(off_diagonal)),
        "maximum_pair_cosine": float(max(off_diagonal)),
    }
