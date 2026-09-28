"""Ground-truth-only oracle decomposition for V13 Phase A.

The functions in this module never participate in training or normal
inference.  They operate on the detector's existing pre-NMS proposal pool and
are intentionally separated from the model so the diagnostic semantics can be
unit tested.
"""

import torch

from .nms import batched_nms


def _threshold_key(value):
    """Return a stable JSON key for a temporal-IoU threshold."""
    return "{:.10g}".format(float(value))


def pairwise_segment_iou(segments, targets):
    """Return temporal IoU for every segment/target pair."""
    if segments.ndim != 2 or segments.shape[1] != 2:
        raise ValueError("segments must have shape [N, 2]")
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("targets must have shape [M, 2]")
    if not segments.shape[0] or not targets.shape[0]:
        return segments.new_zeros((segments.shape[0], targets.shape[0]))
    left = torch.maximum(segments[:, None, 0], targets[None, :, 0])
    right = torch.minimum(segments[:, None, 1], targets[None, :, 1])
    intersection = (right - left).clamp_min(0.0)
    segment_duration = (
        segments[:, 1] - segments[:, 0]).clamp_min(0.0)
    target_duration = (
        targets[:, 1] - targets[:, 0]).clamp_min(0.0)
    union = (
        segment_duration[:, None] + target_duration[None, :]
        - intersection
    ).clamp_min(1e-8)
    return intersection / union


def class_matched_quality(segments, labels, gt_segments, gt_labels):
    """Return each proposal's best tIoU to ground truth of the same class."""
    count = segments.shape[0]
    if labels.shape != (count,):
        raise ValueError("labels must have shape [N]")
    if gt_labels.shape != (gt_segments.shape[0],):
        raise ValueError("gt_labels must have shape [M]")
    quality = segments.new_zeros(count)
    assignment = torch.full(
        (count,), -1, dtype=torch.long, device=segments.device)
    if not count or not gt_segments.shape[0]:
        return quality, assignment
    overlaps = pairwise_segment_iou(segments, gt_segments)
    same_class = labels[:, None] == gt_labels[None, :]
    overlaps = overlaps.masked_fill(~same_class, -1.0)
    best_quality, best_assignment = overlaps.max(dim=1)
    valid = best_quality >= 0.0
    quality[valid] = best_quality[valid]
    assignment[valid] = best_assignment[valid]
    return quality, assignment


def best_available_boundaries(
        segments, labels, gt_segments, gt_labels, tolerance=1e-8):
    """Replace each matched proposal with the best existing boundary for its GT.

    Scores and labels are deliberately untouched.  No ground-truth boundary is
    copied into the prediction: the replacement must already exist in the
    detector's pre-NMS candidate pool.
    """
    quality, assignment = class_matched_quality(
        segments, labels, gt_segments, gt_labels)
    replaced = segments.clone()
    replacement_count = 0
    for target_index in range(gt_segments.shape[0]):
        assigned = torch.nonzero(
            (assignment == target_index) & (quality > 0.0),
            as_tuple=True,
        )[0]
        if not assigned.numel():
            continue
        best_local = quality[assigned].argmax()
        best_candidate = assigned[best_local]
        original = replaced[assigned]
        best_boundary = segments[best_candidate]
        changed = ~torch.isclose(
            original, best_boundary[None], rtol=0.0,
            atol=float(tolerance)).all(dim=1)
        replacement_count += int(changed.sum().item())
        replaced[assigned] = best_boundary
    return replaced, replacement_count


def proposal_recall(
        segments, labels, gt_segments, gt_labels,
        thresholds=(0.5, 0.6, 0.7)):
    """Compute class-matched proposal recall over ground-truth instances."""
    thresholds = tuple(float(value) for value in thresholds)
    threshold_keys = tuple(_threshold_key(value) for value in thresholds)
    total = int(gt_segments.shape[0])
    if total == 0:
        return {
            "ground_truth_instances": 0,
            "matched_instances": {
                key: 0 for key in threshold_keys},
            "recall": {key: 0.0 for key in threshold_keys},
            "mean_best_tiou": 0.0,
        }
    best = gt_segments.new_zeros(total)
    if segments.shape[0]:
        overlaps = pairwise_segment_iou(segments, gt_segments)
        same_class = labels[:, None] == gt_labels[None, :]
        overlaps = overlaps.masked_fill(~same_class, 0.0)
        best = overlaps.max(dim=0).values
    matched = {
        key: int((best >= value).sum().item())
        for key, value in zip(threshold_keys, thresholds)
    }
    return {
        "ground_truth_instances": total,
        "matched_instances": matched,
        "recall": {
            key: float(matched[key] / total)
            for key in threshold_keys
        },
        "mean_best_tiou": float(best.mean().item()),
    }


def _oracle_scores(quality, original_scores):
    """Use tIoU as score with a deterministic original-score tie breaker."""
    return (
        quality + original_scores.clamp(0.0, 1.0) * 1e-7
    ).clamp(max=1.0)


def _postprocess_variant(
        segments, scores, labels, nms_method, iou_threshold, min_score,
        max_seg_num, multiclass, sigma, voting_thresh):
    if nms_method == "none":
        return segments.clone(), scores.clone(), labels.clone()
    return batched_nms(
        segments.clone(), scores.clone(), labels.clone(),
        float(iou_threshold), float(min_score), int(max_seg_num),
        use_soft_nms=(nms_method == "soft"),
        multiclass=bool(multiclass), sigma=float(sigma),
        voting_thresh=float(voting_thresh),
    )


def build_v13_oracle_variants(
        segments, scores, labels, gt_segments, gt_labels, nms_config,
        recall_thresholds=(0.5, 0.6, 0.7)):
    """Build Score, Boundary/Proposal and Joint oracle predictions.

    All three variants use only boundaries already present in ``segments``.
    Ground truth is used for ranking, assignment and selection, so every output
    is diagnostic-only and must never be reported as model performance.
    """
    if scores.shape != (segments.shape[0],):
        raise ValueError("scores must have shape [N]")
    if labels.shape != (segments.shape[0],):
        raise ValueError("labels must have shape [N]")
    required = {
        "method", "iou_threshold", "min_score", "max_seg_num",
        "multiclass", "sigma", "voting_thresh",
    }
    missing = required.difference(nms_config)
    if missing:
        raise ValueError(
            "nms_config is missing: {}".format(", ".join(sorted(missing))))

    original_quality, _ = class_matched_quality(
        segments, labels, gt_segments, gt_labels)
    boundary_segments, replacement_count = best_available_boundaries(
        segments, labels, gt_segments, gt_labels)
    boundary_quality, _ = class_matched_quality(
        boundary_segments, labels, gt_segments, gt_labels)

    common = (
        nms_config["method"], nms_config["iou_threshold"],
        nms_config["min_score"], nms_config["max_seg_num"],
        nms_config["multiclass"], nms_config["sigma"],
        nms_config["voting_thresh"],
    )
    score_variant = _postprocess_variant(
        segments, _oracle_scores(original_quality, scores), labels, *common)
    boundary_variant = _postprocess_variant(
        boundary_segments, scores, labels, *common)
    joint_variant = _postprocess_variant(
        boundary_segments, _oracle_scores(boundary_quality, scores),
        labels, *common)
    return {
        "score": score_variant,
        "boundary": boundary_variant,
        "joint": joint_variant,
        "pre_nms_recall": proposal_recall(
            segments, labels, gt_segments, gt_labels,
            thresholds=recall_thresholds),
        "replacement_count": replacement_count,
        "candidate_count": int(segments.shape[0]),
    }


def merge_recall_statistics(items):
    """Merge per-video proposal-recall dictionaries."""
    if not items:
        return {
            "ground_truth_instances": 0,
            "matched_instances": {},
            "recall": {},
            "mean_best_tiou": 0.0,
        }
    thresholds = tuple(sorted(
        items[0]["matched_instances"],
        key=lambda value: float(value),
    ))
    for item in items[1:]:
        if set(item["matched_instances"]) != set(thresholds):
            raise ValueError(
                "proposal-recall threshold sets must match before merging")
    total = sum(int(item["ground_truth_instances"]) for item in items)
    matched = {
        threshold: sum(
            int(item["matched_instances"][threshold]) for item in items)
        for threshold in thresholds
    }
    weighted_tiou = sum(
        float(item["mean_best_tiou"])
        * int(item["ground_truth_instances"])
        for item in items
    )
    return {
        "ground_truth_instances": total,
        "matched_instances": matched,
        "recall": {
            threshold: float(matched[threshold] / total) if total else 0.0
            for threshold in thresholds
        },
        "mean_best_tiou": float(weighted_tiou / total) if total else 0.0,
    }
