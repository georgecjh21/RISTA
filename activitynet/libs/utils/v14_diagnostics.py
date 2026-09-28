"""Cross-fitted, learnable direction probes for V14 Phase A.

Unlike the V13 ground-truth oracle, these helpers evaluate predictions made
by small models on videos excluded from their training fold.  Ground truth is
used only to construct training targets and to score held-out predictions.
"""

import torch
import torch.nn.functional as F

from .v13_diagnostics import class_matched_quality, proposal_recall


def assign_group_folds(groups, num_folds=5, seed=0):
    """Assign every video group to exactly one deterministic fold."""
    unique = torch.unique(groups, sorted=True)
    if not 2 <= int(num_folds) <= unique.numel():
        raise ValueError("num_folds must be between 2 and the video count")
    generator = torch.Generator().manual_seed(int(seed))
    shuffled = unique[torch.randperm(unique.numel(), generator=generator)]
    mapping = {
        int(group): index % int(num_folds)
        for index, group in enumerate(shuffled.tolist())
    }
    return torch.tensor(
        [mapping[int(group)] for group in groups.tolist()],
        dtype=torch.long)


def proposal_supervision(segments, labels, gt_segments, gt_labels):
    """Return class-matched quality, target boundary and validity."""
    quality, assignment = class_matched_quality(
        segments, labels, gt_segments, gt_labels)
    valid = assignment >= 0
    matched = segments.new_zeros(segments.shape)
    if bool(valid.any()):
        matched[valid] = gt_segments[assignment[valid]]
    duration = (segments[:, 1] - segments[:, 0]).clamp_min(1e-6)
    residual = torch.stack((
        (matched[:, 0] - segments[:, 0]) / duration,
        (matched[:, 1] - segments[:, 1]) / duration,
    ), dim=1)
    residual[~valid] = 0.0
    return quality, residual, matched, valid


def build_probe_inputs(
        regression_features, score_features, segments, scores, labels, levels,
        durations, num_classes, num_levels):
    """Build compact proposal descriptors without using ground truth."""
    if regression_features.ndim != 2:
        raise ValueError("regression_features must have shape [N, D]")
    count = regression_features.shape[0]
    expected_vectors = (scores, labels, levels, durations)
    if any(value.shape != (count,) for value in expected_vectors):
        raise ValueError("proposal metadata must have shape [N]")
    if score_features.shape != (count, 4):
        raise ValueError("score_features must have shape [N, 4]")
    if segments.shape != (count, 2):
        raise ValueError("segments must have shape [N, 2]")
    if num_classes <= 0 or num_levels <= 0:
        raise ValueError("class and level counts must be positive")
    safe_duration = durations.clamp_min(1e-6)
    segment_duration = (
        segments[:, 1] - segments[:, 0]).clamp_min(1e-6)
    geometry = torch.stack((
        segments[:, 0] / safe_duration,
        segments[:, 1] / safe_duration,
        segment_duration / safe_duration,
        segment_duration.log(),
        scores.clamp(1e-6, 1.0 - 1e-6).logit(),
    ), dim=1)
    class_encoding = F.one_hot(
        labels.long(), num_classes=int(num_classes)).to(geometry)
    level_encoding = F.one_hot(
        levels.long(), num_classes=int(num_levels)).to(geometry)
    return torch.cat((
        regression_features.float(),
        score_features.float(),
        geometry.float(),
        class_encoding.float(),
        level_encoding.float(),
    ), dim=1)


def standardize_from_training(inputs, train_mask):
    """Standardize with statistics computed only from training folds."""
    if train_mask.shape != (inputs.shape[0],):
        raise ValueError("train_mask has the wrong shape")
    if not bool(train_mask.any()):
        raise ValueError("training fold is empty")
    mean = inputs[train_mask].mean(dim=0)
    std = inputs[train_mask].std(dim=0, unbiased=False).clamp_min(1e-5)
    return (inputs - mean) / std, mean, std


def apply_boundary_residuals(
        segments, normalized_residuals, durations,
        maximum_relative_shift=0.25):
    """Apply bounded residuals and keep valid segments inside each video."""
    if segments.shape != normalized_residuals.shape:
        raise ValueError("segments and residuals must have matching [N, 2]")
    if durations.shape != (segments.shape[0],):
        raise ValueError("durations must have shape [N]")
    if maximum_relative_shift <= 0.0:
        raise ValueError("maximum_relative_shift must be positive")
    segment_duration = (
        segments[:, 1] - segments[:, 0]).clamp_min(1e-6)
    shift = normalized_residuals.clamp(
        -float(maximum_relative_shift),
        float(maximum_relative_shift)) * segment_duration[:, None]
    corrected = segments + shift
    corrected[:, 0].clamp_(min=0.0)
    corrected[:, 1] = torch.minimum(
        corrected[:, 1].clamp_min(0.0), durations)
    center = corrected.mean(dim=1)
    half = (
        corrected[:, 1] - corrected[:, 0]).clamp_min(1e-4) * 0.5
    corrected = torch.stack((center - half, center + half), dim=1)
    corrected[:, 0].clamp_(min=0.0)
    corrected[:, 1] = torch.minimum(corrected[:, 1], durations)
    return corrected


def blend_quality_scores(original_scores, predicted_quality, power=0.5):
    """Geometrically blend class confidence and predicted localization."""
    if original_scores.shape != predicted_quality.shape:
        raise ValueError("score tensors must have matching shapes")
    if not 0.0 <= float(power) <= 1.0:
        raise ValueError("power must be within [0, 1]")
    original = original_scores.clamp(1e-8, 1.0)
    quality = predicted_quality.clamp(1e-8, 1.0)
    return (
        original.pow(1.0 - float(power))
        * quality.pow(float(power))).clamp(max=1.0)


def merge_recall_by_budget(records, budgets, thresholds=(0.5, 0.6, 0.7)):
    """Aggregate class-matched recall for score prefixes of each video."""
    reports = {}
    for budget in budgets:
        items = []
        for record in records:
            count = min(int(budget), record["segments"].shape[0])
            items.append(proposal_recall(
                record["segments"][:count],
                record["labels"][:count],
                record["gt_segments"],
                record["gt_labels"],
                thresholds=thresholds))
        total = sum(item["ground_truth_instances"] for item in items)
        matched = {
            str(value): sum(
                item["matched_instances"][str(value)] for item in items)
            for value in thresholds
        }
        reports[str(int(budget))] = {
            "ground_truth_instances": total,
            "matched_instances": matched,
            "recall": {
                str(value): matched[str(value)] / total if total else 0.0
                for value in thresholds
            },
            "mean_best_tiou": (
                sum(
                    item["mean_best_tiou"]
                    * item["ground_truth_instances"] for item in items)
                / total if total else 0.0),
        }
    return reports


def recommend_direction(score_gain, boundary_gain, joint_gain):
    """Return a conservative route from held-out, learnable gains."""
    values = {
        "score_oof_gain_percent": float(score_gain),
        "boundary_oof_gain_percent": float(boundary_gain),
        "joint_oof_gain_percent": float(joint_gain),
    }
    if joint_gain >= 1.5 and score_gain >= 0.5 and boundary_gain >= 0.5:
        route = "V14-C_joint_classification_localization"
    elif score_gain >= 0.8 and score_gain >= boundary_gain + 0.2:
        route = "V14-A_quality_aware_classification"
    elif boundary_gain >= 0.8 and boundary_gain >= score_gain + 0.2:
        route = "V14-B_boundary_regression"
    elif max(score_gain, boundary_gain, joint_gain) < 0.3:
        route = "feature_bottleneck_test_stronger_video_features"
    elif joint_gain >= max(score_gain, boundary_gain) + 0.2:
        route = "V14-C_joint_classification_localization"
    else:
        route = "insufficient_margin_repeat_probe_or_feature_test"
    return {
        **values,
        "recommended_route": route,
        "thresholds_are_heuristic": True,
        "uses_test_set_for_selection": False,
    }
