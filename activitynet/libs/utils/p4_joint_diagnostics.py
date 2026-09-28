"""Joint boundary-and-score oracle helpers for ActivityNet P4-0.

P4-0 remains diagnostic-only. Ground truth may select an existing boundary
inside a P2-D0 cluster and may select a globally budgeted score correction.
The model, labels, NMS settings, and external classification protocol are not
changed.
"""

import torch

from .p2d0_diagnostics import render_cluster_predictions
from .p3_score_diagnostics import build_bounded_score_record
from .v13_diagnostics import pairwise_segment_iou


def _best_gt_quality(segments, labels, gt_segments, gt_labels):
    best = gt_segments.new_zeros(gt_segments.shape[0])
    if not segments.shape[0] or not gt_segments.shape[0]:
        return best
    overlaps = pairwise_segment_iou(segments, gt_segments)
    same_class = labels[:, None] == gt_labels[None, :]
    return overlaps.masked_fill(~same_class, 0.0).max(dim=0).values


def build_joint_score_records(cluster_records, boundary_masks):
    """Apply existing-boundary masks and prepare bounded score records.

    ``baseline_gt_best`` is deliberately reset to the cluster anchor-only
    prediction, so combined high-tIoU crossings are always measured against
    the deployable P4 identity rather than against each boundary-only variant.
    """
    if len(cluster_records) != len(boundary_masks):
        raise ValueError(
            "cluster_records and boundary_masks must have the same length")
    outputs = []
    for cluster, boundary_mask in zip(cluster_records, boundary_masks):
        count = int(cluster["anchor_segments"].shape[0])
        if boundary_mask.shape != (count,):
            raise ValueError("boundary mask has an invalid shape")
        selected_segments = cluster["anchor_segments"].clone()
        selected_segments[boundary_mask] = cluster[
            "oracle_segments"][boundary_mask]
        score_record = build_bounded_score_record(
            cluster["video_id"], selected_segments,
            cluster["scores"], cluster["labels"],
            cluster["gt_segments"], cluster["gt_labels"],
            cluster["nms_config"],
        )
        empty = torch.zeros(count, dtype=torch.bool)
        anchor_segments, _, anchor_labels = render_cluster_predictions(
            cluster, empty)
        score_record["baseline_gt_best"] = _best_gt_quality(
            anchor_segments, anchor_labels,
            cluster["gt_segments"], cluster["gt_labels"])
        score_record["boundary_replacement_mask"] = boundary_mask.clone()
        score_record["boundary_true_gains"] = cluster[
            "improvements"][boundary_mask].clone()
        outputs.append(score_record)
    return outputs


def summarize_boundary_replacements(score_records):
    """Aggregate GT-selected existing-boundary replacement statistics."""
    replacement_count = sum(
        int(record["boundary_replacement_mask"].sum().item())
        for record in score_records)
    gains = [
        record["boundary_true_gains"] for record in score_records
        if record["boundary_true_gains"].numel()
    ]
    gains = torch.cat(gains) if gains else torch.empty(0)
    return {
        "boundary_replacement_count": replacement_count,
        "mean_true_boundary_gain": (
            float(gains.mean().item()) if gains.numel() else 0.0),
        "minimum_true_boundary_gain": (
            float(gains.min().item()) if gains.numel() else 0.0),
        "maximum_true_boundary_gain": (
            float(gains.max().item()) if gains.numel() else 0.0),
    }
