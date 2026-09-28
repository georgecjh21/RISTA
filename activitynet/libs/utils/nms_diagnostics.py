"""Read-only diagnostics for proposal interactions under temporal NMS."""

import math

import torch


def pairwise_segment_iou(segments):
    """Return the pairwise temporal IoU matrix for ``[N, 2]`` segments."""
    if segments.ndim != 2 or segments.shape[1] != 2:
        raise ValueError("segments must have shape [N, 2]")
    left = torch.maximum(segments[:, None, 0], segments[None, :, 0])
    right = torch.minimum(segments[:, None, 1], segments[None, :, 1])
    intersection = (right - left).clamp_min(0.0)
    duration = (segments[:, 1] - segments[:, 0]).clamp_min(0.0)
    union = duration[:, None] + duration[None, :] - intersection
    return intersection / union.clamp_min(1e-8)


def same_class_nms_adjacency(
        segments, labels, scores, iou_threshold):
    """Build directed higher-score -> lower-score same-class NMS edges."""
    count = segments.shape[0]
    if labels.shape != (count,) or scores.shape != (count,):
        raise ValueError("labels and scores must have shape [N]")
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be within [0, 1]")
    overlaps = pairwise_segment_iou(segments)
    same_class = labels[:, None] == labels[None, :]
    indices = torch.arange(count, device=scores.device)
    higher_priority = (
        (scores[:, None] > scores[None, :])
        | ((scores[:, None] == scores[None, :])
           & (indices[:, None] < indices[None, :])))
    adjacency = (
        same_class
        & higher_priority
        & (overlaps > float(iou_threshold)))
    adjacency.fill_diagonal_(False)
    return adjacency


def simulate_multiclass_nms(
        segments, labels, scores, iou_threshold, min_score,
        max_seg_num, method="soft", sigma=0.5):
    """Return retained proposal ids and final scores using repository NMS rules.

    The returned score tensor is aligned to original proposal ids. Filtered
    proposals have ``-inf`` scores.
    """
    count = segments.shape[0]
    if labels.shape != (count,) or scores.shape != (count,):
        raise ValueError("labels and scores must have shape [N]")
    if method not in ("soft", "hard", "none"):
        raise ValueError("method must be soft, hard, or none")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    final_scores = scores.new_full((count,), float("-inf"))
    retained = []
    for class_id in torch.unique(labels, sorted=True):
        class_indices = torch.nonzero(
            labels == class_id, as_tuple=True)[0]
        class_segments = segments[class_indices].clone()
        class_scores = scores[class_indices].clone()
        class_ids = class_indices.clone()
        while class_scores.numel():
            best_local = int(class_scores.argmax().item())
            best_id = int(class_ids[best_local].item())
            best_score = class_scores[best_local]
            if float(best_score) <= float(min_score):
                break
            retained.append(best_id)
            final_scores[best_id] = best_score
            keep = torch.arange(
                class_scores.numel(), device=class_scores.device
            ) != best_local
            if not bool(keep.any()):
                break
            remaining_segments = class_segments[keep]
            remaining_scores = class_scores[keep]
            remaining_ids = class_ids[keep]
            if method != "none":
                selected = class_segments[best_local]
                left = torch.maximum(selected[0], remaining_segments[:, 0])
                right = torch.minimum(selected[1], remaining_segments[:, 1])
                intersection = (right - left).clamp_min(0.0)
                selected_duration = (
                    selected[1] - selected[0]).clamp_min(0.0)
                remaining_duration = (
                    remaining_segments[:, 1]
                    - remaining_segments[:, 0]).clamp_min(0.0)
                overlap = intersection / (
                    selected_duration + remaining_duration
                    - intersection).clamp_min(1e-8)
                if method == "soft":
                    remaining_scores = remaining_scores * torch.exp(
                        -(overlap * overlap) / float(sigma))
                else:
                    remaining_scores = remaining_scores * (
                        overlap <= float(iou_threshold))
            valid = remaining_scores > float(min_score)
            class_segments = remaining_segments[valid]
            class_scores = remaining_scores[valid]
            class_ids = remaining_ids[valid]

    if retained:
        retained_tensor = torch.tensor(
            retained, dtype=torch.long, device=scores.device)
        order = final_scores[retained_tensor].argsort(descending=True)
        retained_tensor = retained_tensor[order]
    else:
        retained_tensor = torch.empty(
            0, dtype=torch.long, device=scores.device)
    if int(max_seg_num) > 0:
        retained_tensor = retained_tensor[:int(max_seg_num)]
        allowed = torch.zeros(count, dtype=torch.bool, device=scores.device)
        allowed[retained_tensor] = True
        final_scores[~allowed] = float("-inf")
    return retained_tensor, final_scores


def compare_single_replacement_nms(
        coarse_segments, replacement, proposal_index, labels, scores,
        iou_threshold, min_score, max_seg_num, method="soft", sigma=0.5,
        score_tolerance=1e-8):
    """Compare graph, retention, and score decay after one boundary change."""
    if not 0 <= int(proposal_index) < coarse_segments.shape[0]:
        raise IndexError("proposal_index is out of range")
    refined_segments = coarse_segments.clone()
    refined_segments[int(proposal_index)] = replacement
    coarse_graph = same_class_nms_adjacency(
        coarse_segments, labels, scores, iou_threshold)
    refined_graph = same_class_nms_adjacency(
        refined_segments, labels, scores, iou_threshold)
    coarse_keep, coarse_scores = simulate_multiclass_nms(
        coarse_segments, labels, scores, iou_threshold, min_score,
        max_seg_num, method=method, sigma=sigma)
    refined_keep, refined_scores = simulate_multiclass_nms(
        refined_segments, labels, scores, iou_threshold, min_score,
        max_seg_num, method=method, sigma=sigma)

    changed_edges = coarse_graph ^ refined_graph
    other_mask = torch.ones(
        coarse_segments.shape[0], dtype=torch.bool,
        device=coarse_segments.device)
    other_mask[int(proposal_index)] = False
    coarse_retained = torch.zeros_like(other_mask)
    refined_retained = torch.zeros_like(other_mask)
    coarse_retained[coarse_keep] = True
    refined_retained[refined_keep] = True
    other_retention_changed = bool(
        ((coarse_retained ^ refined_retained) & other_mask).any())

    finite_both = (
        torch.isfinite(coarse_scores) & torch.isfinite(refined_scores)
        & other_mask)
    score_difference = torch.zeros_like(scores)
    score_difference[finite_both] = (
        refined_scores[finite_both] - coarse_scores[finite_both]).abs()
    finite_changed = bool(
        (score_difference > float(score_tolerance)).any())
    finite_mismatch = bool(
        ((torch.isfinite(coarse_scores) ^ torch.isfinite(refined_scores))
         & other_mask).any())
    other_score_changed = finite_changed or finite_mismatch
    graph_changed = bool(changed_edges.any())
    return {
        "graph_changed": graph_changed,
        "changed_edge_count": int(changed_edges.sum().item()),
        "retention_changed": other_retention_changed,
        "other_score_changed": other_score_changed,
        "nms_changed": (
            graph_changed or other_retention_changed or other_score_changed),
        "coarse_retained": int(coarse_keep.numel()),
        "refined_retained": int(refined_keep.numel()),
        "maximum_other_score_change": float(score_difference.max().item())
        if score_difference.numel() else 0.0,
    }


def summarize_nms_stratum(records):
    """Aggregate true utility and harm statistics for one NMS stratum."""
    count = len(records)
    if not count:
        return {
            "count": 0,
            "fraction": 0.0,
            "improvement_count": 0,
            "harmful_count": 0,
            "low_tiou_harm_count": 0,
            "mean_true_utility": 0.0,
            "mean_delta_tiou": 0.0,
            "improvement_rate": 0.0,
            "harmful_rate": 0.0,
            "low_tiou_harm_rate": 0.0,
        }
    improvement_count = sum(
        item["true_utility"] > 0.0 for item in records)
    harmful_count = sum(
        item["true_utility"] < 0.0 for item in records)
    low_tiou_harm_count = sum(
        item["low_tiou_harmful"] for item in records)
    return {
        "count": count,
        "fraction": 0.0,
        "improvement_count": improvement_count,
        "harmful_count": harmful_count,
        "low_tiou_harm_count": low_tiou_harm_count,
        "mean_true_utility": sum(
            item["true_utility"] for item in records) / count,
        "mean_delta_tiou": sum(
            item["delta_tiou"] for item in records) / count,
        "improvement_rate": improvement_count / count,
        "harmful_rate": harmful_count / count,
        "low_tiou_harm_rate": low_tiou_harm_count / count,
    }


def decide_post_nms_gate(
        stable_summary, changed_summary, minimum_harm_events=5,
        minimum_harm_concentration=0.7, minimum_risk_ratio=2.0):
    """Apply a predeclared validation-only rule for entering Phase B."""
    stable_harms = int(stable_summary.get(
        "low_tiou_harm_count",
        round(stable_summary["count"]
              * stable_summary["low_tiou_harm_rate"])))
    changed_harms = int(changed_summary.get(
        "low_tiou_harm_count",
        round(changed_summary["count"]
              * changed_summary["low_tiou_harm_rate"])))
    total_harms = stable_harms + changed_harms
    concentration = (
        changed_harms / total_harms if total_harms else 0.0)
    stable_rate = stable_summary["low_tiou_harm_rate"]
    changed_rate = changed_summary["low_tiou_harm_rate"]
    if stable_rate > 0.0:
        risk_ratio = changed_rate / stable_rate
    elif changed_rate > 0.0:
        risk_ratio = math.inf
    else:
        risk_ratio = 0.0
    enough_events = total_harms >= int(minimum_harm_events)
    concentrated = concentration >= float(minimum_harm_concentration)
    elevated_risk = risk_ratio >= float(minimum_risk_ratio)
    passes = enough_events and concentrated and elevated_risk
    if not enough_events:
        reason = "inconclusive: too few held-out low-tIoU harm events"
    elif not concentrated:
        reason = "do not implement: low-tIoU harm is not concentrated in NMS-changed cases"
    elif not elevated_risk:
        reason = "do not implement: NMS-changed cases do not have elevated harm risk"
    else:
        reason = "proceed: held-out evidence supports a post-NMS safety gate"
    return {
        "passes_post_nms_gate_criterion": passes,
        "reason": reason,
        "low_tiou_harm_events": total_harms,
        "nms_changed_harm_share": concentration,
        "nms_changed_to_stable_risk_ratio": (
            risk_ratio if math.isfinite(risk_ratio) else "infinity"),
        "minimum_harm_events": int(minimum_harm_events),
        "minimum_harm_concentration": float(minimum_harm_concentration),
        "minimum_risk_ratio": float(minimum_risk_ratio),
    }
