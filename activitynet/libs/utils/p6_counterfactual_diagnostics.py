"""P6 full-video counterfactual cluster-policy diagnostic helpers.

Ground truth is permitted only while constructing offline training targets.
The learnability probe consumes proposal features, scores, geometry, levels,
and cluster membership; none of its inputs contains a ground-truth quantity.
"""

import math

import torch

from .p2d0_diagnostics import greedy_local_clusters
from .v13_diagnostics import pairwise_segment_iou


ACTIVITYNET_TIOU_THRESHOLDS = (
    0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95,
)
HIGH_TIOU_THRESHOLDS = (0.85, 0.90, 0.95)


def validate_p6_cache(cache):
    """Validate P6 cache provenance, tensor alignment, and target safety."""
    metadata = cache.get("metadata") or {}
    required_watermarks = (
        "activitynet_p6_full_video_cache",
        "full_video_eval_mode",
        "random_crop_disabled",
        "training_noise_disabled",
        "contains_ground_truth_training_targets",
        "ground_truth_is_absent_from_probe_inputs",
        "forbidden_for_activitynet_validation_inference",
        "existing_candidate_boundaries_only",
        "fixed_anchor_gt_assignment",
        "a0_anchor_is_action_zero",
        "a0_detector_frozen_and_in_eval_mode",
    )
    missing_watermarks = [
        key for key in required_watermarks if not metadata.get(key, False)]
    if missing_watermarks:
        raise ValueError(
            "P6 cache lacks safety watermarks: {}".format(
                sorted(missing_watermarks)))
    if metadata.get("dataset_split") != "training":
        raise ValueError("P6 OOF may use ActivityNet training split only")
    if float(metadata.get("cluster_threshold", -1.0)) != 0.9:
        raise ValueError("P6-0 cache must use cluster threshold 0.9")
    if int(metadata.get("maximum_proposals_per_video", -1)) != 2000:
        raise ValueError("P6-0 cache must preserve pre_nms_topk=2000")

    inputs = set(cache.get("input_keys") or ())
    targets = set(cache.get("training_target_keys") or ())
    if inputs & targets:
        raise ValueError("P6 model inputs contain training target fields")
    forbidden_fragments = (
        "gt", "ground_truth", "utility", "gain", "harm", "improve",
        "target_action", "target_gt")
    leaked = sorted(
        key for key in inputs
        if any(fragment in key.lower() for fragment in forbidden_fragments))
    if leaked:
        raise ValueError(
            "P6 model inputs contain forbidden target/GT names: {}".format(
                leaked))

    required = {
        "member_reg_features", "member_score_features", "member_segments",
        "member_scores", "member_labels", "member_levels", "member_mask",
        "source_member_indices", "durations", "groups", "cluster_sizes",
        "target_actions", "target_gt_indices", "ap_utility", "tiou_gain",
        "high_tiou_harm", "high_tiou_improve", "video_ids",
        "video_cluster_counts", "per_video_statistics",
    }
    missing = required.difference(cache)
    if missing:
        raise ValueError("P6 cache is missing: {}".format(sorted(missing)))
    rows, members = cache["member_mask"].shape
    row_tensors = (
        "member_reg_features", "member_score_features", "member_segments",
        "member_scores", "member_labels", "member_levels", "member_mask",
        "source_member_indices", "durations", "groups", "cluster_sizes",
        "target_actions", "target_gt_indices", "ap_utility", "tiou_gain",
        "high_tiou_harm", "high_tiou_improve")
    for key in row_tensors:
        value = cache[key]
        if not torch.is_tensor(value) or value.shape[0] != rows:
            raise ValueError("P6 cache row alignment failed for {}".format(key))
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("P6 cache contains NaN/Inf in {}".format(key))
    member_tensors = (
        "member_reg_features", "member_score_features", "member_segments",
        "member_scores", "member_labels", "member_levels", "member_mask",
        "source_member_indices", "ap_utility", "tiou_gain",
        "high_tiou_harm", "high_tiou_improve")
    for key in member_tensors:
        if cache[key].shape[1] != members:
            raise ValueError(
                "P6 cache member alignment failed for {}".format(key))
    mask = cache["member_mask"].bool()
    if rows == 0 or members < 2 or not bool(mask[:, 0].all()):
        raise ValueError("P6 cache must contain non-singleton anchor clusters")
    if not bool((mask.sum(dim=1) >= 2).all()):
        raise ValueError("P6 cache contains a singleton policy example")
    sources = cache["source_member_indices"]
    if not bool((sources[mask] >= 0).all()) \
            or not bool((sources[~mask] == -1).all()):
        raise ValueError("P6 cache source-member indices are invalid")
    scores = cache["member_scores"].masked_fill(~mask, float("-inf"))
    if not bool((scores[:, 0] >= scores[:, 1:].max(dim=1).values).all()):
        raise ValueError("P6 action zero is not the highest-score A0 anchor")
    actions = cache["target_actions"].long()
    if not bool(((actions >= 0) & (actions < members)).all()):
        raise ValueError("P6 target action is out of bounds")
    action_valid = mask.gather(1, actions[:, None]).squeeze(1)
    if not bool(action_valid.all()):
        raise ValueError("P6 target action points to padded data")
    positive = actions > 0
    if bool(positive.any()):
        rows_index = torch.nonzero(positive, as_tuple=True)[0]
        utility = cache["ap_utility"][rows_index, actions[rows_index]]
        harmful = cache["high_tiou_harm"][
            rows_index, actions[rows_index]].any(dim=1)
        threshold = float(metadata["positive_ap_utility_threshold"])
        if not bool((utility >= threshold).all()) or bool(harmful.any()):
            raise ValueError("P6 positive target violates AP/safety labels")
    if len(cache["video_ids"]) != len(cache["video_cluster_counts"]) \
            or len(cache["video_ids"]) != len(cache["per_video_statistics"]):
        raise ValueError("P6 per-video metadata is misaligned")
    if int(metadata.get("video_count", -1)) != len(cache["video_ids"]):
        raise ValueError("P6 video-count watermark is inconsistent")
    if len(set(cache["video_ids"])) != len(cache["video_ids"]):
        raise ValueError("P6 cache contains duplicate video IDs")
    if len(set(cache["groups"].tolist())) < 5:
        raise ValueError("P6 cache needs at least five videos for OOF")


def ap_threshold_credit(quality, thresholds=ACTIVITYNET_TIOU_THRESHOLDS):
    """Return mean threshold credit for scalar or vector tIoU quality."""
    values = torch.as_tensor(quality)
    threshold_tensor = values.new_tensor(tuple(float(x) for x in thresholds))
    if not threshold_tensor.numel():
        raise ValueError("AP thresholds must not be empty")
    return (values[..., None] >= threshold_tensor).to(
        values.dtype).mean(dim=-1)


def fixed_gt_member_targets(
        segments, labels, members, gt_segments, gt_labels,
        positive_utility=0.10, tolerance=1e-8):
    """Build fixed-anchor-GT counterfactual targets for one cluster.

    The anchor chooses one same-class GT exactly once. Every member is then
    compared against that GT. A candidate cannot become positive by switching
    to a nearby instance. Candidate zero is always the score anchor/fallback.
    """
    if members.ndim != 1 or not members.numel():
        raise ValueError("members must be a non-empty one-dimensional tensor")
    member_segments = segments[members]
    count = int(members.numel())
    quality = segments.new_zeros(count)
    target_gt = -1
    anchor_index = int(members[0].item())
    same_gt = torch.nonzero(
        gt_labels == labels[anchor_index], as_tuple=True)[0]
    if same_gt.numel():
        anchor_overlap = pairwise_segment_iou(
            segments[anchor_index:anchor_index + 1],
            gt_segments[same_gt])[0]
        local_gt = int(anchor_overlap.argmax().item())
        anchor_quality = float(anchor_overlap[local_gt].item())
        if anchor_quality > float(tolerance):
            target_gt = int(same_gt[local_gt].item())
            quality = pairwise_segment_iou(
                member_segments,
                gt_segments[target_gt:target_gt + 1])[:, 0]
    anchor_quality = quality[0]
    utility = ap_threshold_credit(quality) - ap_threshold_credit(
        anchor_quality)
    high_thresholds = quality.new_tensor(HIGH_TIOU_THRESHOLDS)
    anchor_crossing = anchor_quality >= high_thresholds
    member_crossing = quality[:, None] >= high_thresholds[None]
    high_harm = anchor_crossing[None] & ~member_crossing
    high_improve = ~anchor_crossing[None] & member_crossing
    safe_positive = (
        utility >= float(positive_utility))
    safe_positive &= ~high_harm.any(dim=1)
    safe_positive[0] = False
    target_action = 0
    if bool(safe_positive.any()):
        eligible = torch.nonzero(safe_positive, as_tuple=True)[0]
        target_action = int(eligible[utility[eligible].argmax()].item())
    return {
        "quality": quality,
        "tiou_gain": quality - anchor_quality,
        "ap_utility": utility,
        "high_tiou_harm": high_harm,
        "high_tiou_improve": high_improve,
        "safe_positive": safe_positive,
        "target_action": target_action,
        "target_gt_index": target_gt,
    }


def pad_cluster_record(
        segments, scores, labels, reg_features, score_features, levels,
        members, targets, maximum_members):
    """Pad one non-singleton cluster into a cache-ready tensor record."""
    maximum_members = int(maximum_members)
    if maximum_members < 2:
        raise ValueError("maximum_members must be at least two")
    members = members[:maximum_members]
    count = int(members.numel())
    if count < 2:
        raise ValueError("P6 policy cache accepts non-singleton clusters only")
    feature_dim = int(reg_features.shape[1])
    score_dim = int(score_features.shape[1])
    output = {
        "member_segments": segments.new_zeros(maximum_members, 2),
        "member_scores": scores.new_zeros(maximum_members),
        "member_labels": labels.new_zeros(maximum_members),
        "member_reg_features": reg_features.new_zeros(
            maximum_members, feature_dim),
        "member_score_features": score_features.new_zeros(
            maximum_members, score_dim),
        "member_levels": levels.new_zeros(maximum_members),
        "member_mask": torch.zeros(maximum_members, dtype=torch.bool),
        "ap_utility": scores.new_zeros(maximum_members),
        "tiou_gain": scores.new_zeros(maximum_members),
        "high_tiou_harm": torch.zeros(
            maximum_members, len(HIGH_TIOU_THRESHOLDS), dtype=torch.bool),
        "high_tiou_improve": torch.zeros(
            maximum_members, len(HIGH_TIOU_THRESHOLDS), dtype=torch.bool),
        "target_action": int(targets["target_action"]),
        "target_gt_index": int(targets["target_gt_index"]),
        "cluster_size": count,
        "source_member_indices": torch.full(
            (maximum_members,), -1, dtype=torch.long),
    }
    output["member_segments"][:count] = segments[members]
    output["member_scores"][:count] = scores[members]
    output["member_labels"][:count] = labels[members]
    output["member_reg_features"][:count] = reg_features[members]
    output["member_score_features"][:count] = score_features[members]
    output["member_levels"][:count] = levels[members]
    output["member_mask"][:count] = True
    output["ap_utility"][:count] = targets["ap_utility"][:count]
    output["tiou_gain"][:count] = targets["tiou_gain"][:count]
    output["high_tiou_harm"][:count] = targets[
        "high_tiou_harm"][:count]
    output["high_tiou_improve"][:count] = targets[
        "high_tiou_improve"][:count]
    output["source_member_indices"][:count] = members
    return output


def cluster_video_examples(
        segments, scores, labels, reg_features, score_features, levels,
        gt_segments, gt_labels, overlap_threshold=0.9,
        maximum_members=16, positive_utility=0.10):
    """Cluster one full-video proposal pool and build P6 cache records."""
    if reg_features.ndim != 2 or reg_features.shape[0] != segments.shape[0]:
        raise ValueError("reg_features must align with segments")
    if score_features.ndim != 2 \
            or score_features.shape[0] != segments.shape[0]:
        raise ValueError("score_features must align with segments")
    clusters = greedy_local_clusters(
        segments, scores, labels, float(overlap_threshold))
    examples = []
    anchor_false_positives = 0
    positive_clusters = 0
    for members in clusters:
        anchor = int(members[0].item())
        same_gt = torch.nonzero(
            gt_labels == labels[anchor], as_tuple=True)[0]
        anchor_quality = 0.0
        if same_gt.numel():
            anchor_quality = float(pairwise_segment_iou(
                segments[anchor:anchor + 1],
                gt_segments[same_gt]).max().item())
        anchor_false_positives += int(anchor_quality < 0.5)
        if members.numel() < 2:
            continue
        members = members[:int(maximum_members)]
        targets = fixed_gt_member_targets(
            segments, labels, members, gt_segments, gt_labels,
            positive_utility=positive_utility)
        record = pad_cluster_record(
            segments, scores, labels, reg_features, score_features,
            levels, members, targets, maximum_members)
        positive_clusters += int(record["target_action"] > 0)
        examples.append(record)
    return examples, {
        "candidate_count": int(segments.shape[0]),
        "cluster_count": len(clusters),
        "non_singleton_cluster_count": len(examples),
        "positive_cluster_count": positive_clusters,
        "anchor_false_positive_count": anchor_false_positives,
        "ground_truth_count": int(gt_segments.shape[0]),
    }


def pairwise_policy_metadata(
        member_segments, member_scores, member_levels, member_mask,
        durations, number_of_levels):
    """Build inference-safe anchor-relative scalar descriptors."""
    if member_segments.ndim != 3 or member_segments.shape[-1] != 2:
        raise ValueError("member_segments must have shape [B, K, 2]")
    batch, members = member_scores.shape
    if member_segments.shape[:2] != (batch, members):
        raise ValueError("member segments and scores must align")
    if member_levels.shape != (batch, members) \
            or member_mask.shape != (batch, members):
        raise ValueError("member level/mask tensors must align")
    if durations.shape != (batch,):
        raise ValueError("durations must have shape [B]")
    anchor = member_segments[:, :1]
    anchor_duration = (
        anchor[..., 1] - anchor[..., 0]).clamp_min(1e-6)
    candidate_duration = (
        member_segments[..., 1] - member_segments[..., 0]).clamp_min(1e-6)
    intersection = (
        torch.minimum(anchor[..., 1], member_segments[..., 1])
        - torch.maximum(anchor[..., 0], member_segments[..., 0])
    ).clamp_min(0.0)
    union = (
        anchor_duration + candidate_duration - intersection).clamp_min(1e-6)
    overlap = intersection / union
    anchor_center = anchor.mean(dim=-1)
    candidate_center = member_segments.mean(dim=-1)
    safe_video = durations[:, None].clamp_min(1e-6)
    safe_score = member_scores.clamp(1e-6, 1.0 - 1e-6)
    anchor_score = member_scores[:, :1]
    level_denominator = max(int(number_of_levels) - 1, 1)
    metadata = torch.stack((
        (member_segments[..., 0] - anchor[..., 0]) / anchor_duration,
        (member_segments[..., 1] - anchor[..., 1]) / anchor_duration,
        torch.log(candidate_duration / anchor_duration),
        overlap,
        anchor_score.expand(-1, members),
        member_scores,
        member_scores - anchor_score,
        (candidate_center - anchor_center).abs() / anchor_duration,
        member_segments[..., 0] / safe_video,
        member_segments[..., 1] / safe_video,
        candidate_duration / safe_video,
        torch.logit(safe_score),
        member_levels.to(member_scores.dtype) / level_denominator,
        member_mask.to(member_scores.dtype),
    ), dim=-1)
    return metadata


def deterministic_video_folds(groups, number_of_folds=5, seed=0):
    """Assign whole cached videos to deterministic OOF folds."""
    unique = torch.unique(groups, sorted=True)
    if not 2 <= int(number_of_folds) <= int(unique.numel()):
        raise ValueError("number_of_folds must fit the video count")
    generator = torch.Generator().manual_seed(int(seed))
    shuffled = unique[torch.randperm(unique.numel(), generator=generator)]
    mapping = {
        int(group): index % int(number_of_folds)
        for index, group in enumerate(shuffled.tolist())
    }
    return torch.tensor(
        [mapping[int(group)] for group in groups.tolist()],
        dtype=torch.long)


def expected_calibration_error(confidence, outcome, bins=10):
    """Compute binary ECE for replacement-improvement confidence."""
    if confidence.shape != outcome.shape:
        raise ValueError("confidence and outcome must have equal shapes")
    if confidence.numel() == 0:
        return 0.0
    confidence = confidence.float().clamp(0.0, 1.0)
    outcome = outcome.float()
    edges = torch.linspace(0.0, 1.0, int(bins) + 1)
    error = confidence.new_zeros(())
    for index in range(int(bins)):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == bins - 1
            else confidence < upper)
        if bool(selected.any()):
            fraction = selected.float().mean()
            error += fraction * (
                confidence[selected].mean()
                - outcome[selected].mean()).abs()
    return float(error.item())


def select_budgeted_actions(
        logits, member_mask, groups, video_cluster_counts,
        replacement_fraction=0.01, minimum_margin=0.0):
    """Select OOF actions using only predictions and fixed per-video budget."""
    if logits.shape != member_mask.shape:
        raise ValueError("logits and member_mask must have equal shapes")
    masked = logits.masked_fill(~member_mask, float("-inf"))
    candidate_logits = masked[:, 1:]
    best_candidate_logit, local_action = candidate_logits.max(dim=1)
    action = local_action + 1
    anchor_logit = masked[:, 0]
    margin = best_candidate_logit - anchor_logit
    probability = torch.softmax(masked, dim=1).gather(
        1, action[:, None]).squeeze(1)
    priority = margin * probability
    selected = torch.zeros(logits.shape[0], dtype=torch.bool)
    for group in torch.unique(groups, sorted=True):
        group_index = int(group.item())
        cluster_indices = torch.nonzero(
            groups == group, as_tuple=True)[0]
        budget = int(math.floor(
            int(video_cluster_counts[group_index])
            * float(replacement_fraction)))
        eligible = cluster_indices[margin[cluster_indices] > minimum_margin]
        if budget <= 0 or not eligible.numel():
            continue
        ranked = eligible[torch.argsort(
            priority[eligible], descending=True, stable=True)]
        selected[ranked[:budget]] = True
    return {
        "action": action,
        "margin": margin,
        "confidence": probability,
        "priority": priority,
        "selected": selected,
    }


def summarize_selected_actions(
        selection, cache, high_thresholds=HIGH_TIOU_THRESHOLDS):
    """Summarize held-out counterfactual actions using cached targets."""
    selected = selection["selected"]
    rows = torch.nonzero(selected, as_tuple=True)[0]
    if rows.numel():
        actions = selection["action"][rows]
        utility = cache["ap_utility"][rows, actions]
        tiou_gain = cache["tiou_gain"][rows, actions]
        harm = cache["high_tiou_harm"][rows, actions]
        improve = cache["high_tiou_improve"][rows, actions]
        confidence = selection["confidence"][rows]
    else:
        utility = torch.empty(0)
        tiou_gain = torch.empty(0)
        harm = torch.empty(0, len(tuple(high_thresholds)), dtype=torch.bool)
        improve = torch.empty_like(harm)
        confidence = torch.empty(0)
    positive = utility > 1e-8
    total_clusters = int(sum(cache["video_cluster_counts"]))
    total_gt = int(cache["metadata"]["ground_truth_count"])
    return {
        "replacement_count": int(rows.numel()),
        "replacement_ratio": (
            rows.numel() / total_clusters if total_clusters else 0.0),
        "true_improvement_count": int(positive.sum().item()),
        "true_harm_count": int((utility < -1e-8).sum().item()),
        "true_neutral_count": int((utility.abs() <= 1e-8).sum().item()),
        "true_improvement_precision": (
            float(positive.float().mean().item())
            if utility.numel() else 0.0),
        "mean_true_ap_utility": (
            float(utility.mean().item()) if utility.numel() else 0.0),
        "mean_true_tiou_gain": (
            float(tiou_gain.mean().item()) if tiou_gain.numel() else 0.0),
        "threshold_coverage_proxy_gain_percent": (
            float(utility.sum().item()) / total_gt * 100.0
            if total_gt else 0.0),
        "calibration_error": expected_calibration_error(
            confidence, positive),
        "high_tiou": {
            "{:.2f}".format(float(threshold)): {
                "improved_count": int(improve[:, index].sum().item()),
                "harmed_count": int(harm[:, index].sum().item()),
            }
            for index, threshold in enumerate(high_thresholds)
        },
    }


def p6_route_decision(summary):
    """Apply fixed P6-0 go/no-go gates without validation-set tuning."""
    precision = float(summary["true_improvement_precision"])
    mean_utility = float(summary["mean_true_ap_utility"])
    proxy = float(summary["threshold_coverage_proxy_gain_percent"])
    safety = all(
        int(item["improved_count"]) >= int(item["harmed_count"])
        for key, item in summary["high_tiou"].items()
        if key in ("0.90", "0.95"))
    core = precision >= 0.70 and mean_utility > 0.02 and safety
    strong = core and proxy >= 1.0
    if strong:
        route = "ENTER_P6A_STRONG_GATE"
    elif core and proxy >= 0.5:
        route = "ENTER_P6A_ONE_CONSERVATIVE_ITERATION"
    else:
        route = "STOP_SELECTION_ROUTE"
    return {
        "recommended_route": route,
        "replacement_precision_gate": 0.70,
        "replacement_precision_passed": precision >= 0.70,
        "mean_ap_utility_gate": 0.02,
        "mean_ap_utility_passed": mean_utility > 0.02,
        "high_tiou_safety_passed": safety,
        "minimum_proxy_gain_percent": 0.5,
        "strong_proxy_gain_percent": 1.0,
        "proxy_gain_passed": proxy >= 0.5,
        "strong_proxy_gain_passed": proxy >= 1.0,
        "enter_p6a": route.startswith("ENTER_P6A"),
    }
