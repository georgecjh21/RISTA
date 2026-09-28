"""Video-level contextual descriptors for V14.1 OOF score probes."""

import torch

from .nms_diagnostics import pairwise_segment_iou


def _compact_features(features, groups=16):
    """Deterministically compress channels without fitting on held-out data."""
    if features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    groups = min(int(groups), int(features.shape[1]))
    if groups <= 0:
        raise ValueError("groups must be positive")
    chunks = torch.tensor_split(features.float(), groups, dim=1)
    return torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)


def build_video_context_features(
        segments, scores, labels, levels, regression_features,
        nms_iou_threshold=0.1, neighbor_topk=8, feature_groups=16):
    """Describe same-class proposal competition without using ground truth.

    Each proposal receives score-rank, density, overlap/NMS-risk, neighbouring
    score/scale statistics, and compact self/neighbour regression features.
    """
    count = int(segments.shape[0])
    if segments.shape != (count, 2):
        raise ValueError("segments must have shape [N, 2]")
    if any(value.shape != (count,) for value in (scores, labels, levels)):
        raise ValueError("scores, labels and levels must have shape [N]")
    if regression_features.ndim != 2 \
            or regression_features.shape[0] != count:
        raise ValueError("regression_features must have shape [N, D]")
    if not 0.0 <= float(nms_iou_threshold) <= 1.0:
        raise ValueError("nms_iou_threshold must be within [0, 1]")
    if int(neighbor_topk) <= 0:
        raise ValueError("neighbor_topk must be positive")

    compact = _compact_features(regression_features, feature_groups)
    scalar_dim = 19
    output = torch.zeros(
        count, scalar_dim + compact.shape[1] * 3, dtype=torch.float32)
    durations = (segments[:, 1] - segments[:, 0]).clamp_min(1e-6)
    for class_id in torch.unique(labels, sorted=True):
        indices = torch.nonzero(labels == class_id, as_tuple=True)[0]
        class_count = int(indices.numel())
        if class_count == 0:
            continue
        class_segments = segments[indices].float()
        class_scores = scores[indices].float()
        class_levels = levels[indices].float()
        class_durations = durations[indices].float()
        class_compact = compact[indices]
        overlaps = pairwise_segment_iou(class_segments)
        overlaps.fill_diagonal_(0.0)

        order = class_scores.argsort(descending=True)
        rank = torch.empty(class_count, dtype=torch.float32)
        rank[order] = torch.arange(class_count, dtype=torch.float32)
        rank /= max(class_count - 1, 1)
        higher = class_scores[None, :] > class_scores[:, None]
        higher_overlap = overlaps.masked_fill(~higher, 0.0)
        maximum_higher_overlap = higher_overlap.max(dim=1).values
        suppression_risk = (
            higher_overlap > float(nms_iou_threshold)).float().mean(dim=1)

        k = min(int(neighbor_topk), max(class_count - 1, 1))
        top_values, top_positions = overlaps.topk(k, dim=1)
        neighbour_scores = class_scores[top_positions]
        neighbour_durations = class_durations[top_positions]
        neighbour_levels = class_levels[top_positions]
        neighbour_compact = class_compact[top_positions]
        valid_neighbour = top_values > 0.0
        weights = top_values * valid_neighbour.float()
        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled_compact = (
            neighbour_compact * weights[..., None]).sum(dim=1) / weight_sum
        weighted_score = (
            neighbour_scores * weights).sum(dim=1) / weight_sum[:, 0]
        score_variance = (
            (neighbour_scores - weighted_score[:, None]).square()
            * weights).sum(dim=1) / weight_sum[:, 0]
        duration_ratio = (
            neighbour_durations
            / class_durations[:, None].clamp_min(1e-6))
        weighted_duration_ratio = (
            duration_ratio * weights).sum(dim=1) / weight_sum[:, 0]
        same_level_ratio = (
            (neighbour_levels == class_levels[:, None]).float()
            * valid_neighbour.float()).sum(dim=1) \
            / valid_neighbour.float().sum(dim=1).clamp_min(1.0)

        scalar = torch.stack((
            rank,
            torch.full_like(rank, float(class_count)).log1p(),
            (class_scores - class_scores.mean())
            / class_scores.std(unbiased=False).clamp_min(1e-6),
            torch.full_like(rank, float(class_scores.mean())),
            torch.full_like(rank, float(class_scores.std(
                unbiased=False))),
            torch.full_like(rank, float(class_scores.max())),
            overlaps.max(dim=1).values,
            overlaps.mean(dim=1),
            (overlaps > 0.1).float().mean(dim=1),
            (overlaps > 0.3).float().mean(dim=1),
            (overlaps > 0.5).float().mean(dim=1),
            (overlaps > 0.7).float().mean(dim=1),
            maximum_higher_overlap,
            suppression_risk,
            weighted_score,
            score_variance.sqrt(),
            weighted_duration_ratio,
            same_level_ratio,
            class_durations / class_durations.mean().clamp_min(1e-6),
        ), dim=1)
        output[indices] = torch.cat((
            scalar,
            class_compact,
            pooled_compact,
            class_compact - pooled_compact,
        ), dim=1)
    return output


def build_context_ranking_pairs(
        segments, scores, labels, quality, video_slices,
        maximum_candidates_per_class=64, maximum_pairs_per_class=16,
        minimum_overlap=0.1, minimum_quality_gap=0.05):
    """Mine fixed same-video/class high-quality -> low-quality pairs."""
    pairs = []
    for start, end in video_slices:
        start, end = int(start), int(end)
        local_labels = labels[start:end]
        for class_id in torch.unique(local_labels, sorted=True):
            local = torch.nonzero(
                local_labels == class_id, as_tuple=True)[0] + start
            count = min(int(maximum_candidates_per_class), local.numel())
            if count < 2:
                continue
            selected = local[
                scores[local].argsort(descending=True)[:count]]
            overlaps = pairwise_segment_iou(segments[selected].float())
            difference = quality[selected, None] - quality[selected][None]
            eligible = (
                (difference >= float(minimum_quality_gap))
                & (overlaps >= float(minimum_overlap)))
            eligible.fill_diagonal_(False)
            high, low = torch.nonzero(eligible, as_tuple=True)
            if high.numel() == 0:
                continue
            priority = difference[high, low]
            keep = priority.argsort(descending=True)[
                :int(maximum_pairs_per_class)]
            pairs.append(torch.stack((
                selected[high[keep]], selected[low[keep]]), dim=1))
    if not pairs:
        return torch.empty(0, 2, dtype=torch.long)
    return torch.cat(pairs, dim=0)


def context_probe_decision(gain_percent):
    """Apply the predeclared V14.1 validation-only decision rule."""
    gain = float(gain_percent)
    if gain >= 0.5:
        route = "implement_training_time_context_ranking"
    elif gain >= 0.1:
        route = "weak_context_signal_prioritize_stronger_video_features"
    else:
        route = "stop_i3d_context_route_test_stronger_video_features"
    return {
        "context_oof_gain_percent": gain,
        "required_gain_percent": 0.5,
        "passes_context_gate": gain >= 0.5,
        "recommended_route": route,
        "uses_test_set_for_selection": False,
    }
