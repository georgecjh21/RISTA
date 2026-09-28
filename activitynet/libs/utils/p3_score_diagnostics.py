"""Bounded ground-truth score oracle for ActivityNet P3-0.

This module is diagnostic-only. Ground truth defines a proposal's target
localization quality and selects a globally limited set of score corrections.
It never changes proposal boundaries or labels and never participates in
normal model inference or training.
"""

import math

import torch

from .v13_diagnostics import (
    _postprocess_variant,
    class_matched_quality,
    pairwise_segment_iou,
)


def _validate_inputs(segments, scores, labels, gt_segments, gt_labels):
    if segments.ndim != 2 or segments.shape[1] != 2:
        raise ValueError("segments must have shape [N, 2]")
    if scores.shape != (segments.shape[0],):
        raise ValueError("scores must have shape [N]")
    if labels.shape != (segments.shape[0],):
        raise ValueError("labels must have shape [N]")
    if gt_segments.ndim != 2 or gt_segments.shape[1] != 2:
        raise ValueError("gt_segments must have shape [M, 2]")
    if gt_labels.shape != (gt_segments.shape[0],):
        raise ValueError("gt_labels must have shape [M]")
    if not torch.isfinite(segments).all() or not torch.isfinite(scores).all():
        raise ValueError("segments and scores must be finite")
    if bool(((scores < 0.0) | (scores > 1.0)).any()):
        raise ValueError("scores must be probabilities within [0, 1]")


def _best_gt_quality(segments, labels, gt_segments, gt_labels):
    best = gt_segments.new_zeros(gt_segments.shape[0])
    if not segments.shape[0] or not gt_segments.shape[0]:
        return best
    overlaps = pairwise_segment_iou(segments, gt_segments)
    same_class = labels[:, None] == gt_labels[None, :]
    return overlaps.masked_fill(~same_class, 0.0).max(dim=0).values


def _render(record, scores):
    config = record["nms_config"]
    return _postprocess_variant(
        record["segments"], scores, record["labels"],
        config["method"], config["iou_threshold"], config["min_score"],
        config["max_seg_num"], config["multiclass"], config["sigma"],
        config["voting_thresh"],
    )


def build_bounded_score_record(
        video_id, segments, scores, labels, gt_segments, gt_labels,
        nms_config, epsilon=1e-6):
    """Build one video's fixed-boundary score-calibration record."""
    _validate_inputs(segments, scores, labels, gt_segments, gt_labels)
    required = {
        "method", "iou_threshold", "min_score", "max_seg_num",
        "multiclass", "sigma", "voting_thresh",
    }
    missing = required.difference(nms_config)
    if missing:
        raise ValueError(
            "nms_config is missing: {}".format(
                ", ".join(sorted(missing))))
    epsilon = float(epsilon)
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be within (0, 0.5)")

    segments = segments.detach().cpu()
    scores = scores.detach().cpu()
    labels = labels.detach().cpu()
    gt_segments = gt_segments.detach().cpu()
    gt_labels = gt_labels.detach().cpu()
    quality, _ = class_matched_quality(
        segments, labels, gt_segments, gt_labels)
    safe_scores = scores.clamp(epsilon, 1.0 - epsilon)
    safe_quality = quality.clamp(epsilon, 1.0 - epsilon)
    score_logits = torch.logit(safe_scores)
    target_logits = torch.logit(safe_quality)
    desired_delta = target_logits - score_logits

    record = {
        "video_id": video_id,
        "segments": segments,
        "scores": scores,
        "labels": labels,
        "quality": quality,
        "score_logits": score_logits,
        "desired_logit_delta": desired_delta,
        "gt_segments": gt_segments,
        "gt_labels": gt_labels,
        "nms_config": dict(nms_config),
    }
    baseline_segments, _, baseline_labels = _render(record, scores)
    record["baseline_gt_best"] = _best_gt_quality(
        baseline_segments, baseline_labels, gt_segments, gt_labels)
    return record


def global_score_adjustment_mask_sweep(
        records, budget_fractions, tolerance=1e-8):
    """Select the globally largest GT score mismatches under each budget."""
    budgets = tuple(float(value) for value in budget_fractions)
    if any(not 0.0 <= budget <= 1.0 for budget in budgets):
        raise ValueError("budget fractions must be within [0, 1]")
    if len(budgets) != len(set(budgets)):
        raise ValueError("budget fractions must be unique")
    lengths = [int(record["scores"].numel()) for record in records]
    total = sum(lengths)
    mismatch = torch.cat([
        record["desired_logit_delta"].abs() for record in records
    ]) if total else torch.empty(0)
    eligible = torch.nonzero(
        mismatch > float(tolerance), as_tuple=True)[0]
    if eligible.numel():
        order = torch.argsort(
            mismatch[eligible], descending=True, stable=True)
        ranked = eligible[order]
    else:
        ranked = eligible

    outputs = {}
    for budget in budgets:
        requested = int(math.ceil(total * budget))
        used = min(requested, int(ranked.numel()))
        flat_mask = torch.zeros(total, dtype=torch.bool)
        flat_mask[ranked[:used]] = True
        outputs[budget] = (list(torch.split(flat_mask, lengths)), {
            "budget_fraction": budget,
            "total_candidate_proposals": total,
            "requested_adjustment_limit": requested,
            "eligible_mismatch_proposals": int(ranked.numel()),
            "used_adjustments": used,
        })
    return outputs


def render_bounded_score_predictions(record, adjustment_mask, logit_cap):
    """Apply bounded score corrections and the detector's original NMS."""
    count = int(record["scores"].numel())
    if adjustment_mask.shape != (count,):
        raise ValueError("adjustment_mask must have shape [num_proposals]")
    cap = float(logit_cap)
    if cap < 0.0:
        raise ValueError("logit_cap must be non-negative")
    scores = record["scores"].clone()
    if bool(adjustment_mask.any()) and cap > 0.0:
        delta = record["desired_logit_delta"].clamp(-cap, cap)
        scores[adjustment_mask] = torch.sigmoid(
            record["score_logits"][adjustment_mask]
            + delta[adjustment_mask])
    return _render(record, scores)


def summarize_score_adjustments(
        records, masks, logit_cap, rendered_predictions=None):
    """Report score movement and high-tIoU coverage crossings."""
    if len(records) != len(masks):
        raise ValueError("records and masks must have the same length")
    if (rendered_predictions is not None
            and len(rendered_predictions) != len(records)):
        raise ValueError(
            "rendered_predictions and records must have the same length")
    cap = float(logit_cap)
    if cap < 0.0:
        raise ValueError("logit_cap must be non-negative")
    selected_deltas = []
    selected_quality = []
    selected_before = []
    selected_after = []
    crossings = {
        "{:.2f}".format(value): {
            "improved_count": 0, "harmed_count": 0}
        for value in (0.85, 0.90, 0.95)
    }
    for index, (record, mask) in enumerate(zip(records, masks)):
        if mask.shape != record["scores"].shape:
            raise ValueError("adjustment mask has an invalid shape")
        delta = record["desired_logit_delta"].clamp(-cap, cap)
        if bool(mask.any()):
            before = record["scores"][mask]
            after = torch.sigmoid(record["score_logits"][mask] + delta[mask])
            selected_deltas.append(delta[mask])
            selected_quality.append(record["quality"][mask])
            selected_before.append(before)
            selected_after.append(after)

        if rendered_predictions is None:
            variant_segments, _, variant_labels = \
                render_bounded_score_predictions(record, mask, cap)
        else:
            variant_segments, _, variant_labels = \
                rendered_predictions[index]
        variant_best = _best_gt_quality(
            variant_segments, variant_labels,
            record["gt_segments"], record["gt_labels"])
        baseline_best = record["baseline_gt_best"]
        for threshold in (0.85, 0.90, 0.95):
            key = "{:.2f}".format(threshold)
            before_recall = baseline_best >= threshold
            after_recall = variant_best >= threshold
            crossings[key]["improved_count"] += int(
                ((~before_recall) & after_recall).sum().item())
            crossings[key]["harmed_count"] += int(
                (before_recall & (~after_recall)).sum().item())

    def concatenate(items):
        return torch.cat(items) if items else torch.empty(0)

    deltas = concatenate(selected_deltas)
    quality = concatenate(selected_quality)
    before = concatenate(selected_before)
    after = concatenate(selected_after)
    upward = deltas > 0.0
    downward = deltas < 0.0
    count = int(deltas.numel())
    return {
        "adjustment_count": count,
        "upward_adjustment_count": int(upward.sum().item()),
        "downward_adjustment_count": int(downward.sum().item()),
        "mean_signed_logit_adjustment": (
            float(deltas.mean().item()) if count else 0.0),
        "mean_absolute_logit_adjustment": (
            float(deltas.abs().mean().item()) if count else 0.0),
        "mean_selected_true_tiou": (
            float(quality.mean().item()) if count else 0.0),
        "mean_score_before": (
            float(before.mean().item()) if count else 0.0),
        "mean_score_after": (
            float(after.mean().item()) if count else 0.0),
        "high_tiou_gt_coverage": crossings,
    }
