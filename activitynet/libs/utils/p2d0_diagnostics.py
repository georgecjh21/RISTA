"""Deployability-constrained proposal-cluster oracle for ActivityNet P2-D0.

This module is diagnostic-only.  Ground truth may select an existing proposal
inside a local cluster, but it never creates a boundary, changes a score, or
participates in normal model inference.
"""

import math

import torch

from .v13_diagnostics import pairwise_segment_iou, _postprocess_variant


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


def greedy_local_clusters(segments, scores, labels, overlap_threshold):
    """Form deterministic NMS-style local clusters around top-score anchors.

    The highest-score unassigned proposal becomes the next anchor.  Every
    unassigned same-class proposal whose tIoU with that anchor reaches the
    threshold joins its cluster.  This deliberately avoids transitive graph
    components, which can merge distant proposals through overlap chains.
    """
    threshold = float(overlap_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("overlap_threshold must be within [0, 1]")
    if scores.shape != (segments.shape[0],):
        raise ValueError("scores must have shape [N]")
    if labels.shape != (segments.shape[0],):
        raise ValueError("labels must have shape [N]")
    if not segments.shape[0]:
        return []

    clusters = []
    for class_id in torch.unique(labels):
        class_indices = torch.nonzero(
            labels == class_id, as_tuple=True)[0]
        score_order = class_indices[
            torch.argsort(scores[class_indices], descending=True)]
        start_order = class_indices[
            torch.argsort(segments[class_indices, 0])]
        sorted_starts = segments[start_order, 0]
        remaining = torch.zeros(
            segments.shape[0], dtype=torch.bool, device=segments.device)
        remaining[class_indices] = True
        durations = (segments[:, 1] - segments[:, 0]).clamp_min(0.0)

        for anchor_index in score_order:
            anchor = int(anchor_index.item())
            if not bool(remaining[anchor]):
                continue
            # tIoU >= threshold requires temporal overlap and, for t > 0,
            # proposal-length ratio within [t, 1/t].  The start-sorted prefix
            # avoids scanning every proposal for every anchor.
            prefix_end = int(torch.searchsorted(
                sorted_starts, segments[anchor, 1], right=True).item())
            candidates = start_order[:prefix_end]
            candidates = candidates[
                remaining[candidates]
                & (segments[candidates, 1] >= segments[anchor, 0])]
            if threshold > 0.0:
                anchor_duration = durations[anchor]
                candidates = candidates[
                    (durations[candidates] >= threshold * anchor_duration)
                    & (durations[candidates]
                       <= anchor_duration / threshold)]
            overlaps = pairwise_segment_iou(
                segments[anchor:anchor + 1], segments[candidates])[0]
            members = candidates[overlaps >= threshold]
            if not members.numel():
                members = anchor_index.reshape(1)
            remaining[members] = False
            # Anchor is explicitly first; remaining members keep score order.
            members_by_score = members[
                torch.argsort(scores[members], descending=True)]
            if int(members_by_score[0].item()) != anchor:
                raise RuntimeError(
                    "cluster anchor is not its highest-score member")
            clusters.append(members_by_score.detach().cpu())
    clusters.sort(
        key=lambda members: float(scores[int(members[0])]), reverse=True)
    return clusters


def _best_gt_quality(segments, labels, gt_segments, gt_labels):
    """Return the best class-matched proposal tIoU for every GT instance."""
    best = gt_segments.new_zeros(gt_segments.shape[0])
    if not segments.shape[0] or not gt_segments.shape[0]:
        return best
    overlaps = pairwise_segment_iou(segments, gt_segments)
    same_class = labels[:, None] == gt_labels[None, :]
    return overlaps.masked_fill(~same_class, 0.0).max(dim=0).values


def build_cluster_oracle_record(
        video_id, segments, scores, labels, gt_segments, gt_labels,
        nms_config, overlap_threshold, tolerance=1e-8):
    """Build one video's cluster anchors and GT-selected existing candidates.

    A cluster is assigned to the same-class GT best matched by its anchor.  The
    oracle may replace the anchor only with a member that improves tIoU to that
    same GT.  Scores and labels always remain those of the A0 anchor.
    """
    _validate_inputs(segments, scores, labels, gt_segments, gt_labels)
    required = {
        "method", "iou_threshold", "min_score", "max_seg_num",
        "multiclass", "sigma", "voting_thresh",
    }
    missing = required.difference(nms_config)
    if missing:
        raise ValueError(
            "nms_config is missing: {}".format(", ".join(sorted(missing))))

    segments = segments.detach().cpu()
    scores = scores.detach().cpu()
    labels = labels.detach().cpu()
    gt_segments = gt_segments.detach().cpu()
    gt_labels = gt_labels.detach().cpu()
    clusters = greedy_local_clusters(
        segments, scores, labels, overlap_threshold)

    anchor_segments = []
    oracle_segments = []
    anchor_scores = []
    anchor_labels = []
    gains = []
    anchor_quality = []
    oracle_quality = []
    target_gt_indices = []
    cluster_sizes = []
    multi_gt_any = []
    multi_gt_05 = []

    for members in clusters:
        anchor_index = int(members[0].item())
        anchor = segments[anchor_index]
        label = labels[anchor_index]
        member_segments = segments[members]
        same_gt = torch.nonzero(gt_labels == label, as_tuple=True)[0]

        selected = anchor_index
        selected_gt = -1
        before = 0.0
        after = 0.0
        ambiguity_any = 0
        ambiguity_05 = 0
        if same_gt.numel():
            member_overlaps = pairwise_segment_iou(
                member_segments, gt_segments[same_gt])
            per_gt_best = member_overlaps.max(dim=0).values
            ambiguity_any = int((per_gt_best > tolerance).sum().item())
            ambiguity_05 = int((per_gt_best >= 0.5).sum().item())

            anchor_overlaps = pairwise_segment_iou(
                anchor[None], gt_segments[same_gt])[0]
            local_gt = int(anchor_overlaps.argmax().item())
            before = float(anchor_overlaps[local_gt].item())
            if before > tolerance:
                selected_gt = int(same_gt[local_gt].item())
                candidate_quality = member_overlaps[:, local_gt]
                local_best = int(candidate_quality.argmax().item())
                candidate_index = int(members[local_best].item())
                candidate_value = float(candidate_quality[local_best].item())
                if candidate_value > before + tolerance:
                    selected = candidate_index
                    after = candidate_value
                else:
                    after = before

        anchor_segments.append(anchor)
        oracle_segments.append(segments[selected])
        anchor_scores.append(scores[anchor_index])
        anchor_labels.append(label)
        gains.append(after - before)
        anchor_quality.append(before)
        oracle_quality.append(after)
        target_gt_indices.append(selected_gt)
        cluster_sizes.append(int(members.numel()))
        multi_gt_any.append(ambiguity_any > 1)
        multi_gt_05.append(ambiguity_05 > 1)

    def stacked(items, shape, dtype):
        if items:
            return torch.stack(items).to(dtype=dtype)
        return torch.empty((0,) + shape, dtype=dtype)

    anchor_segments = stacked(anchor_segments, (2,), segments.dtype)
    oracle_segments = stacked(oracle_segments, (2,), segments.dtype)
    anchor_scores = stacked(anchor_scores, (), scores.dtype)
    anchor_labels = stacked(anchor_labels, (), labels.dtype)
    gains = torch.tensor(gains, dtype=segments.dtype)
    target_gt_indices = torch.tensor(target_gt_indices, dtype=torch.long)
    record = {
        "video_id": video_id,
        "overlap_threshold": float(overlap_threshold),
        "candidate_count": int(segments.shape[0]),
        "anchor_segments": anchor_segments,
        "oracle_segments": oracle_segments,
        "scores": anchor_scores,
        "labels": anchor_labels,
        "improvements": gains,
        "anchor_quality": torch.tensor(anchor_quality, dtype=segments.dtype),
        "oracle_quality": torch.tensor(oracle_quality, dtype=segments.dtype),
        "target_gt_indices": target_gt_indices,
        "cluster_sizes": torch.tensor(cluster_sizes, dtype=torch.long),
        "multi_gt_any": torch.tensor(multi_gt_any, dtype=torch.bool),
        "multi_gt_05": torch.tensor(multi_gt_05, dtype=torch.bool),
        "gt_segments": gt_segments,
        "gt_labels": gt_labels,
        "nms_config": dict(nms_config),
    }
    record["anchor_gt_best"] = _best_gt_quality(
        anchor_segments, anchor_labels, gt_segments, gt_labels)
    record["oracle_gt_best"] = _best_gt_quality(
        oracle_segments, anchor_labels, gt_segments, gt_labels)
    return record


def global_replacement_masks(records, budget_fraction, tolerance=1e-8):
    """Select the globally best positive replacements within a cluster budget."""
    return global_replacement_mask_sweep(
        records, (budget_fraction,), tolerance=tolerance)[
            float(budget_fraction)]


def global_replacement_mask_sweep(
        records, budget_fractions, tolerance=1e-8):
    """Rank positive gains once and materialize masks for several budgets."""
    budgets = tuple(float(value) for value in budget_fractions)
    if any(not 0.0 <= budget <= 1.0 for budget in budgets):
        raise ValueError("budget fractions must be within [0, 1]")
    if len(budgets) != len(set(budgets)):
        raise ValueError("budget fractions must be unique")
    lengths = [
        int(record["improvements"].numel()) for record in records]
    all_gains = torch.cat([
        record["improvements"] for record in records
    ]) if sum(lengths) else torch.empty(0)
    total_clusters = sum(
        int(record["improvements"].numel()) for record in records)
    positive_indices = torch.nonzero(
        all_gains > float(tolerance), as_tuple=True)[0]
    if positive_indices.numel():
        rank_order = torch.argsort(
            all_gains[positive_indices], descending=True)
        ranked_indices = positive_indices[rank_order]
    else:
        ranked_indices = positive_indices

    outputs = {}
    for budget in budgets:
        requested = int(math.ceil(total_clusters * budget))
        used = min(requested, int(ranked_indices.numel()))
        flat_mask = torch.zeros(total_clusters, dtype=torch.bool)
        flat_mask[ranked_indices[:used]] = True
        masks = list(torch.split(flat_mask, lengths))
        outputs[budget] = (masks, {
            "budget_fraction": budget,
            "total_clusters": total_clusters,
            "requested_replacement_limit": requested,
            "positive_gain_clusters": int(ranked_indices.numel()),
            "used_replacements": used,
        })
    return outputs


def full_oracle_masks(records, tolerance=1e-8):
    """Select every strictly positive cluster replacement."""
    return [
        record["improvements"] > float(tolerance) for record in records]


def render_cluster_predictions(record, replacement_mask=None):
    """Apply selected boundaries and the unchanged detector NMS settings."""
    count = int(record["anchor_segments"].shape[0])
    if replacement_mask is None:
        replacement_mask = torch.zeros(count, dtype=torch.bool)
    if replacement_mask.shape != (count,):
        raise ValueError("replacement_mask must have shape [num_clusters]")
    segments = record["anchor_segments"].clone()
    segments[replacement_mask] = record["oracle_segments"][replacement_mask]
    config = record["nms_config"]
    return _postprocess_variant(
        segments, record["scores"], record["labels"],
        config["method"], config["iou_threshold"], config["min_score"],
        config["max_seg_num"], config["multiclass"], config["sigma"],
        config["voting_thresh"],
    )


def summarize_cluster_records(records, high_tiou=(0.85, 0.90, 0.95)):
    """Aggregate cluster ambiguity, gains, and GT coverage crossings."""
    total_candidates = sum(record["candidate_count"] for record in records)
    total_clusters = sum(
        int(record["improvements"].numel()) for record in records)
    positive = torch.cat([
        record["improvements"] for record in records
    ]) if total_clusters else torch.empty(0)
    cluster_sizes = torch.cat([
        record["cluster_sizes"] for record in records
    ]) if total_clusters else torch.empty(0, dtype=torch.long)
    multi_any = sum(
        int(record["multi_gt_any"].sum().item()) for record in records)
    multi_05 = sum(
        int(record["multi_gt_05"].sum().item()) for record in records)
    crossings = high_tiou_crossings(
        records, full_oracle_masks(records), high_tiou=high_tiou)
    positive_mask = positive > 1e-8
    positive_values = positive[positive_mask]
    gain_quantiles = {}
    for name, quantile in (
            ("p50", 0.50), ("p75", 0.75), ("p90", 0.90),
            ("p95", 0.95), ("p99", 0.99)):
        gain_quantiles[name] = (
            float(torch.quantile(positive_values, quantile).item())
            if positive_values.numel() else 0.0)
    return {
        "video_count": len(records),
        "candidate_count": int(total_candidates),
        "cluster_count": int(total_clusters),
        "mean_cluster_size": (
            float(cluster_sizes.float().mean().item())
            if total_clusters else 0.0),
        "multi_gt_cluster_count_any_overlap": int(multi_any),
        "multi_gt_cluster_ratio_any_overlap": (
            float(multi_any / total_clusters) if total_clusters else 0.0),
        "multi_gt_cluster_count_tiou_0_5": int(multi_05),
        "multi_gt_cluster_ratio_tiou_0_5": (
            float(multi_05 / total_clusters) if total_clusters else 0.0),
        "positive_gain_cluster_count": int(positive_mask.sum().item()),
        "positive_gain_cluster_ratio": (
            float(positive_mask.float().mean().item())
            if total_clusters else 0.0),
        "mean_true_gain_all_clusters": (
            float(positive.mean().item()) if total_clusters else 0.0),
        "mean_true_gain_positive_clusters": (
            float(positive_values.mean().item())
            if bool(positive_mask.any()) else 0.0),
        "max_true_gain": (
            float(positive.max().item()) if total_clusters else 0.0),
        "positive_true_gain_quantiles": gain_quantiles,
        "high_tiou_gt_coverage": crossings,
    }


def high_tiou_crossings(
        records, replacement_masks, high_tiou=(0.85, 0.90, 0.95)):
    """Count per-GT threshold improvements and harms for selected replacements."""
    if len(records) != len(replacement_masks):
        raise ValueError("records and replacement_masks must have equal length")
    anchor_values = []
    selected_values = []
    for record, mask in zip(records, replacement_masks):
        count = int(record["anchor_segments"].shape[0])
        if mask.shape != (count,):
            raise ValueError(
                "replacement mask must have shape [num_clusters]")
        selected = record["anchor_segments"].clone()
        selected[mask] = record["oracle_segments"][mask]
        anchor_values.append(record["anchor_gt_best"])
        selected_values.append(_best_gt_quality(
            selected, record["labels"],
            record["gt_segments"], record["gt_labels"]))
    anchor_best = torch.cat(anchor_values) if anchor_values else torch.empty(0)
    selected_best = (
        torch.cat(selected_values) if selected_values else torch.empty(0))
    crossings = {}
    for threshold in high_tiou:
        key = "{:.2f}".format(float(threshold))
        before = anchor_best >= float(threshold)
        after = selected_best >= float(threshold)
        crossings[key] = {
            "improved_count": int((~before & after).sum().item()),
            "harmed_count": int((before & ~after).sum().item()),
            "anchor_reached_count": int(before.sum().item()),
            "selected_reached_count": int(after.sum().item()),
        }
    return crossings
