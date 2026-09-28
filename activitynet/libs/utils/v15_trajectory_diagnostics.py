"""Pure helpers for the V15-B high-quality boundary trajectory diagnostic."""

from typing import Dict, Iterable, List, Sequence, Tuple

import torch


DEFAULT_TIOU_GROUPS = (
    ("0.5-0.6", 0.5, 0.6),
    ("0.6-0.7", 0.6, 0.7),
    ("0.7-0.8", 0.7, 0.8),
    ("0.8+", 0.8, float("inf")),
)


def temporal_iou(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Vectorized temporal IoU for ``[N, 2]`` boundaries."""
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


def tiou_group_masks(
    coarse_tiou: torch.Tensor,
    groups: Sequence[Tuple[str, float, float]] = DEFAULT_TIOU_GROUPS,
) -> Dict[str, torch.Tensor]:
    """Return non-overlapping masks with left-closed, right-open intervals."""
    masks = {}
    for name, lower, upper in groups:
        masks[name] = (
            (coarse_tiou >= float(lower))
            & (coarse_tiou < float(upper))
        )
    return masks


def stratified_indices(
    coarse_tiou: torch.Tensor,
    samples_per_group: int,
    seed: int,
    groups: Sequence[Tuple[str, float, float]] = DEFAULT_TIOU_GROUPS,
) -> Tuple[torch.Tensor, Dict[str, Dict[str, int]]]:
    """Deterministically sample up to ``samples_per_group`` from each group."""
    if samples_per_group <= 0:
        raise ValueError("samples_per_group must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    selected: List[torch.Tensor] = []
    counts = {}
    for name, _, _ in groups:
        population = torch.nonzero(
            tiou_group_masks(coarse_tiou, groups)[name], as_tuple=True)[0]
        take = min(int(samples_per_group), int(population.numel()))
        if take:
            order = torch.randperm(
                population.numel(), generator=generator)[:take]
            selected.append(population[order])
        counts[name] = {
            "population": int(population.numel()),
            "sampled": int(take),
        }
    if not selected:
        raise RuntimeError("No samples fall within the requested tIoU groups")
    return torch.cat(selected), counts


def boundary_metric_row(
    coarse: torch.Tensor,
    target: torch.Tensor,
    predicted: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """Summarize boundary bias and quality for one coarse-tIoU group."""
    count = int(mask.sum().item())
    if count == 0:
        return {"count": 0}
    coarse = coarse[mask].float()
    target = target[mask].float()
    predicted = predicted[mask].float()
    target_duration = (
        target[:, 1] - target[:, 0]).clamp_min(1e-6)
    coarse_duration = (
        coarse[:, 1] - coarse[:, 0]).clamp_min(1e-6)
    predicted_duration = (
        predicted[:, 1] - predicted[:, 0]).clamp_min(1e-6)
    normalized_center_error = (
        predicted.mean(dim=-1) - target.mean(dim=-1)
    ) / target_duration
    log_duration_bias = torch.log(
        predicted_duration / target_duration)
    log_duration_change = torch.log(
        predicted_duration / coarse_duration)
    coarse_quality = temporal_iou(coarse, target)
    predicted_quality = temporal_iou(predicted, target)
    delta = predicted_quality - coarse_quality
    return {
        "count": count,
        "coarse_mean_tiou": float(coarse_quality.mean()),
        "predicted_mean_tiou": float(predicted_quality.mean()),
        "mean_delta_tiou": float(delta.mean()),
        "center_bias_target_duration": float(
            normalized_center_error.mean()),
        "mean_abs_center_error_target_duration": float(
            normalized_center_error.abs().mean()),
        "log_duration_bias_to_target": float(
            log_duration_bias.mean()),
        "mean_abs_log_duration_error": float(
            log_duration_bias.abs().mean()),
        "log_duration_change_from_coarse": float(
            log_duration_change.mean()),
        "expansion_rate_over_target": float(
            (predicted_duration > target_duration).float().mean()),
        "expansion_rate_over_coarse": float(
            (predicted_duration > coarse_duration).float().mean()),
        "tiou_improved_rate": float((delta > 1e-6).float().mean()),
        "tiou_harmful_rate": float((delta < -1e-6).float().mean()),
        "tiou_drop_over_0p05_rate": float(
            (delta < -0.05).float().mean()),
    }


def grouped_boundary_metrics(
    coarse: torch.Tensor,
    target: torch.Tensor,
    predicted: torch.Tensor,
    groups: Sequence[Tuple[str, float, float]] = DEFAULT_TIOU_GROUPS,
) -> Dict[str, Dict[str, float]]:
    coarse_tiou = temporal_iou(coarse, target)
    masks = tiou_group_masks(coarse_tiou, groups)
    return {
        name: boundary_metric_row(coarse, target, predicted, masks[name])
        for name, _, _ in groups
    }


def combine_group_rows(
    rows: Dict[str, Dict[str, float]],
    group_names: Iterable[str],
) -> Dict[str, float]:
    """Count-weight numeric metrics over selected group rows."""
    chosen = [
        rows[name] for name in group_names
        if name in rows and rows[name].get("count", 0) > 0
    ]
    total = sum(int(row["count"]) for row in chosen)
    if total == 0:
        return {"count": 0}
    keys = set.intersection(*(
        set(row.keys()) for row in chosen))
    result = {"count": total}
    for key in sorted(keys - {"count"}):
        result[key] = sum(
            float(row[key]) * int(row["count"]) for row in chosen
        ) / total
    return result


def trajectory_diagnosis(
    single_step_rows: Dict[str, Dict[str, float]],
    final_ddim_rows: Dict[str, Dict[str, float]],
    minimum_bias_delta: float = 0.02,
    minimum_rate_delta: float = 0.05,
) -> Dict:
    """Classify whether expansion begins in x0 or accumulates along DDIM."""
    high_groups = ("0.7-0.8", "0.8+")
    single = combine_group_rows(single_step_rows, high_groups)
    final = combine_group_rows(final_ddim_rows, high_groups)
    if single.get("count", 0) == 0 or final.get("count", 0) == 0:
        return {
            "primary_finding": "insufficient_high_quality_samples",
            "recommendation": (
                "Increase samples_per_group before choosing V15-C or a "
                "trajectory modification."),
            "high_quality_single_step": single,
            "high_quality_final_ddim": final,
        }
    single_expansion = (
        single["log_duration_bias_to_target"] > minimum_bias_delta
        and single["expansion_rate_over_target"] > 0.55
    )
    trajectory_accumulation = (
        final["log_duration_bias_to_target"]
        > single["log_duration_bias_to_target"] + minimum_bias_delta
        or final["expansion_rate_over_target"]
        > single["expansion_rate_over_target"] + minimum_rate_delta
        or final["tiou_harmful_rate"]
        > single["tiou_harmful_rate"] + minimum_rate_delta
    )
    if trajectory_accumulation and not single_expansion:
        finding = "ddim_trajectory_accumulation"
        recommendation = (
            "Modify the DDIM trajectory or add trajectory-consistency "
            "supervision before adding a keep/refine mixture.")
    elif single_expansion and not trajectory_accumulation:
        finding = "single_step_x0_expansion"
        recommendation = (
            "Revise the state target and add an identity/no-change prior "
            "before changing the sampler.")
    elif single_expansion and trajectory_accumulation:
        finding = "single_step_and_trajectory_expansion"
        recommendation = (
            "The x0 target already expands and DDIM compounds it; correct "
            "the identity/state prior first, then re-check the trajectory.")
    else:
        finding = "no_clear_expansion_mechanism"
        recommendation = (
            "Do not enter V15-C yet; inspect timestep-band and per-step "
            "tables because expansion is not the dominant high-quality error.")
    return {
        "primary_finding": finding,
        "recommendation": recommendation,
        "single_step_expansion_present": bool(single_expansion),
        "ddim_trajectory_accumulation_present": bool(
            trajectory_accumulation),
        "decision_thresholds": {
            "minimum_log_duration_bias_or_delta": minimum_bias_delta,
            "minimum_rate_delta": minimum_rate_delta,
            "single_step_expansion_rate_threshold": 0.55,
        },
        "high_quality_single_step": single,
        "high_quality_final_ddim": final,
    }
