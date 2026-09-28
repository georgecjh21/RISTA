import math
import time
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import DTFAM, MaskedConv1D, Scale, LayerNorm, DynamicScale_chk
from .selective_dynamic_scale import SelectiveDynamicScale


class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits

class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )

    def forward(self, fpn_feats, fpn_masks, return_features=False):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        reg_features = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            reg_features += (cur_out, )
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), )

        # fpn_masks remains the same
        return (out_offsets, reg_features) if return_features else out_offsets


class StartEndDistributionHead(nn.Module):
    """Predict a bounded residual distribution for both temporal boundaries.

    The baseline regression distances remain the main prediction.  A
    zero-initialized per-level/per-side gate blends the expected residual into
    those distances, so enabling this head is an exact identity before any
    optimization:

        refined = coarse + tanh(gate) * E[residual]

    Distribution logits are still supervised while the gate is zero, avoiding
    the dead-branch problem of a purely zero-gated residual network.
    """

    def __init__(
            self,
            feature_dim,
            fpn_levels,
            hidden_dim=128,
            num_bins=17,
            max_residual=2.0,
            dropout=0.0,
            gate_init=0.0):
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if fpn_levels <= 0:
            raise ValueError("fpn_levels must be positive")
        if num_bins < 3 or num_bins % 2 == 0:
            raise ValueError("num_bins must be odd and at least 3")
        if max_residual <= 0.0:
            raise ValueError("max_residual must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.feature_dim = int(feature_dim)
        self.fpn_levels = int(fpn_levels)
        self.hidden_dim = int(hidden_dim)
        self.num_bins = int(num_bins)
        self.max_residual = float(max_residual)

        self.input_projection = MaskedConv1D(
            self.feature_dim + 2, self.hidden_dim, 3, padding=1,
            bias=False)
        self.norm = LayerNorm(self.hidden_dim)
        self.activation = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(float(dropout))
        self.logit_projection = MaskedConv1D(
            self.hidden_dim, 2 * self.num_bins, 1)

        self.gates = nn.Parameter(torch.full(
            (self.fpn_levels, 2, 1), float(gate_init)))
        self.register_buffer(
            "residual_bins",
            torch.linspace(
                -self.max_residual, self.max_residual, self.num_bins),
            persistent=True)

    def forward(self, reg_features, fpn_masks, coarse_offsets):
        if not (
                len(reg_features) == len(fpn_masks) == len(coarse_offsets)
                == self.fpn_levels):
            raise ValueError("unexpected number of start/end pyramid levels")

        refined_offsets = []
        distribution_logits = []
        for level, (features, mask, coarse) in enumerate(zip(
                reg_features, fpn_masks, coarse_offsets)):
            if features.shape[1] != self.feature_dim:
                raise ValueError("unexpected regression feature dimension")
            if coarse.shape[1] != 2:
                raise ValueError("coarse offsets must have two boundary sides")

            # Coarse geometry is conditioning evidence, not a target path for
            # the auxiliary distribution loss.
            geometry = torch.log1p(coarse.detach())
            hidden, _ = self.input_projection(
                torch.cat((features, geometry), dim=1), mask)
            hidden = self.dropout(self.activation(self.norm(hidden)))
            logits, _ = self.logit_projection(hidden, mask)
            logits = logits.reshape(
                logits.shape[0], 2, self.num_bins, logits.shape[-1])
            probabilities = logits.softmax(dim=2)
            expected = (
                probabilities
                * self.residual_bins.to(probabilities)[None, None, :, None]
            ).sum(dim=2)
            gate = torch.tanh(self.gates[level])[None]
            refined = F.relu(coarse + gate * expected)
            refined_offsets.append(refined)
            distribution_logits.append(logits)

        return tuple(refined_offsets), tuple(distribution_logits)


class BoundaryAwareProposalGenerator(nn.Module):
    """Generate several bounded start/end hypotheses without replacing coarse.

    The head learns residual distributions around the original regression
    distances.  A zero-initialized gate makes every generated hypothesis
    identical to the coarse boundary at initialization.  At inference the
    detector keeps the coarse proposal and may append only hypotheses whose
    learned improvement probability passes a conservative threshold.
    """

    def __init__(
            self,
            feature_dim,
            fpn_levels,
            hidden_dim=128,
            num_bins=17,
            max_residual=4.0,
            num_candidates=4,
            dropout=0.0,
            gate_init=0.0):
        super().__init__()
        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        side_topk = int(math.ceil(math.sqrt(num_candidates)))
        if side_topk > num_bins:
            raise ValueError("num_candidates requires more bins than available")
        self.feature_dim = int(feature_dim)
        self.fpn_levels = int(fpn_levels)
        self.hidden_dim = int(hidden_dim)
        self.num_bins = int(num_bins)
        self.max_residual = float(max_residual)
        self.num_candidates = int(num_candidates)
        self.side_topk = side_topk
        if self.feature_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if self.fpn_levels <= 0:
            raise ValueError("fpn_levels must be positive")
        if self.num_bins < 3 or self.num_bins % 2 == 0:
            raise ValueError("num_bins must be odd and at least three")
        if self.max_residual <= 0.0:
            raise ValueError("max_residual must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_projection = MaskedConv1D(
            self.feature_dim + 2, self.hidden_dim, 3, padding=1,
            bias=False)
        self.norm = LayerNorm(self.hidden_dim)
        self.activation = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(float(dropout))
        self.distribution_projection = MaskedConv1D(
            self.hidden_dim, 2 * self.num_bins, 1)
        self.improvement_projection = MaskedConv1D(
            self.hidden_dim, self.num_candidates, 1)
        # Start conservatively: no proposal is accepted before learning.
        nn.init.zeros_(self.improvement_projection.conv.weight)
        nn.init.constant_(self.improvement_projection.conv.bias, -2.0)

        self.gates = nn.Parameter(torch.full(
            (self.fpn_levels, 2, 1), float(gate_init)))
        self.register_buffer(
            "residual_bins",
            torch.linspace(
                -self.max_residual, self.max_residual, self.num_bins),
            persistent=True)

    def _decode_level(self, coarse, logits, improvement_logits, level):
        """Decode top joint start/end modes into K offset hypotheses."""
        probabilities = logits.softmax(dim=2)
        _, top_indices = probabilities.topk(self.side_topk, dim=2)
        bins = self.residual_bins.to(coarse)
        start_residuals = bins[top_indices[:, 0]]
        end_residuals = bins[top_indices[:, 1]]

        combinations = []
        combination_probabilities = []
        for start_index in range(self.side_topk):
            for end_index in range(self.side_topk):
                combinations.append(torch.stack((
                    start_residuals[:, start_index],
                    end_residuals[:, end_index]), dim=-1))
                combination_probabilities.append(
                    probabilities[:, 0].gather(
                        1, top_indices[:, 0, start_index:start_index + 1]
                    ).squeeze(1)
                    * probabilities[:, 1].gather(
                        1, top_indices[:, 1, end_index:end_index + 1]
                    ).squeeze(1))
        residuals = torch.stack(combinations, dim=2)
        joint_probability = torch.stack(
            combination_probabilities, dim=2)
        order = joint_probability.argsort(dim=2, descending=True)[
            :, :, :self.num_candidates]
        residuals = residuals.gather(
            2, order[..., None].expand(-1, -1, -1, 2))
        # Candidate zero is the differentiable distribution expectation.
        # This guarantees a useful gradient for both the logits and the
        # zero-initialized residual gate; the remaining candidates retain
        # distinct high-probability modes for proposal diversity.
        expected = (
            probabilities
            * bins.reshape(1, 1, -1, 1)
        ).sum(dim=2).permute(0, 2, 1)
        residuals = torch.cat(
            (expected[:, :, None], residuals[:, :, 1:]), dim=2)

        gate = torch.tanh(self.gates[level]).reshape(1, 1, 1, 2)
        candidates = F.relu(coarse.permute(0, 2, 1)[:, :, None] + (
            gate * residuals))
        quality = improvement_logits.permute(0, 2, 1)
        return candidates, quality

    def forward(self, reg_features, fpn_masks, coarse_offsets):
        if not (
                len(reg_features) == len(fpn_masks) == len(coarse_offsets)
                == self.fpn_levels):
            raise ValueError("unexpected number of proposal pyramid levels")
        candidates, distributions, improvements = [], [], []
        for level, (features, mask, coarse) in enumerate(zip(
                reg_features, fpn_masks, coarse_offsets)):
            if features.shape[1] != self.feature_dim:
                raise ValueError("unexpected regression feature dimension")
            if coarse.shape[1] != 2:
                raise ValueError("coarse offsets must have two sides")
            geometry = torch.log1p(coarse.detach())
            hidden, _ = self.input_projection(
                torch.cat((features, geometry), dim=1), mask)
            hidden = self.dropout(self.activation(self.norm(hidden)))
            logits, _ = self.distribution_projection(hidden, mask)
            logits = logits.reshape(
                logits.shape[0], 2, self.num_bins, logits.shape[-1])
            improvement, _ = self.improvement_projection(hidden, mask)
            decoded, decoded_quality = self._decode_level(
                coarse, logits, improvement, level)
            candidates.append(decoded)
            distributions.append(logits)
            improvements.append(decoded_quality)
        return tuple(candidates), tuple(distributions), tuple(improvements)


class TridentRelativeBoundaryHead(nn.Module):
    """Class-aware relative boundary modeling with a safe coarse fallback.

    This is a DyFADet-native implementation of the relative-boundary idea
    introduced by TriDet.  A center-distance distribution is combined with
    class-specific start/end responses sampled from neighbouring temporal
    locations.  Unlike replacing the baseline regression head, a
    zero-initialized residual gate blends the decoded Trident candidate with
    DyFADet's original coarse offsets:

        output = coarse + tanh(gate) * (trident - coarse)

    Consequently, enabling this module before optimization is an exact
    element-wise identity with respect to the T0 coarse detector.
    """

    def __init__(
            self,
            boundary_feature_dim,
            regression_feature_dim,
            num_classes,
            fpn_levels,
            hidden_dim=128,
            num_bins=16,
            num_layers=2,
            kernel_size=3,
            dropout=0.0,
            prior_prob=0.01,
            detach_boundary_features=True,
            gate_init=0.0):
        super().__init__()
        if boundary_feature_dim <= 0 or regression_feature_dim <= 0:
            raise ValueError("Trident feature dimensions must be positive")
        if num_classes <= 0 or fpn_levels <= 0:
            raise ValueError("Trident classes and FPN levels must be positive")
        if num_bins <= 0:
            raise ValueError("Trident num_bins must be positive")
        if num_layers <= 0:
            raise ValueError("Trident num_layers must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Trident kernel_size must be positive and odd")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Trident dropout must be in [0, 1)")

        self.num_classes = int(num_classes)
        self.fpn_levels = int(fpn_levels)
        self.num_bins = int(num_bins)
        self.detach_boundary_features = bool(detach_boundary_features)
        self.activation = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(float(dropout))

        self.boundary_tower = nn.ModuleList()
        self.boundary_norm = nn.ModuleList()
        for layer in range(int(num_layers)):
            in_dim = boundary_feature_dim if layer == 0 else hidden_dim
            self.boundary_tower.append(MaskedConv1D(
                in_dim, hidden_dim, kernel_size,
                padding=kernel_size // 2, bias=False))
            self.boundary_norm.append(LayerNorm(hidden_dim))
        self.start_projection = MaskedConv1D(
            hidden_dim, self.num_classes, kernel_size,
            padding=kernel_size // 2)
        self.end_projection = MaskedConv1D(
            hidden_dim, self.num_classes, kernel_size,
            padding=kernel_size // 2)

        self.center_tower = MaskedConv1D(
            regression_feature_dim, hidden_dim, kernel_size,
            padding=kernel_size // 2, bias=False)
        self.center_norm = LayerNorm(hidden_dim)
        self.center_projection = MaskedConv1D(
            hidden_dim, 2 * (self.num_bins + 1), kernel_size,
            padding=kernel_size // 2)

        if prior_prob > 0.0:
            prior_bias = -math.log((1.0 - prior_prob) / prior_prob)
            nn.init.constant_(self.start_projection.conv.bias, prior_bias)
            nn.init.constant_(self.end_projection.conv.bias, prior_bias)

        self.gates = nn.Parameter(torch.full(
            (self.fpn_levels, 2, 1), float(gate_init)))
        self.register_buffer(
            "left_indices",
            torch.arange(self.num_bins, -1, -1, dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            "right_indices",
            torch.arange(self.num_bins + 1, dtype=torch.float32),
            persistent=True)

    def _boundary_logits(self, feature, mask):
        hidden = feature.detach() if self.detach_boundary_features else feature
        for convolution, normalization in zip(
                self.boundary_tower, self.boundary_norm):
            hidden, _ = convolution(hidden, mask)
            hidden = self.dropout(
                self.activation(normalization(hidden)))
        start, _ = self.start_projection(hidden, mask)
        end, _ = self.end_projection(hidden, mask)
        return start, end

    def _neighbour_windows(self, logits, side):
        if side == "start":
            padded = F.pad(logits, (self.num_bins, 0))
        elif side == "end":
            padded = F.pad(logits, (0, self.num_bins))
        else:
            raise ValueError("Trident boundary side must be start or end")
        # [B, C, T, bins+1] -> [B, T, C, bins+1]
        return padded.unfold(-1, self.num_bins + 1, 1).permute(
            0, 2, 1, 3)

    def forward(
            self, boundary_features, regression_features, fpn_masks,
            coarse_offsets):
        if not (
                len(boundary_features) == len(regression_features)
                == len(fpn_masks) == len(coarse_offsets)
                == self.fpn_levels):
            raise ValueError("unexpected number of Trident pyramid levels")

        outputs, candidates = [], []
        center_logits_all, start_logits_all, end_logits_all = [], [], []
        for level, (boundary_feature, regression_feature, mask, coarse) in \
                enumerate(zip(
                    boundary_features, regression_features, fpn_masks,
                    coarse_offsets)):
            if coarse.ndim != 3 or coarse.shape[1] != 2:
                raise ValueError(
                    "Trident coarse offsets must have shape [B, 2, T]")
            start_logits, end_logits = self._boundary_logits(
                boundary_feature, mask)
            center_hidden, _ = self.center_tower(
                regression_feature, mask)
            center_hidden = self.dropout(self.activation(
                self.center_norm(center_hidden)))
            center_logits, _ = self.center_projection(center_hidden, mask)
            center_logits = center_logits.reshape(
                center_logits.shape[0], 2, self.num_bins + 1,
                center_logits.shape[-1]).permute(0, 3, 1, 2)

            start_neighbours = self._neighbour_windows(
                start_logits, "start")
            end_neighbours = self._neighbour_windows(end_logits, "end")
            left_probabilities = F.softmax(
                start_neighbours + center_logits[:, :, :1, :], dim=-1)
            right_probabilities = F.softmax(
                end_neighbours + center_logits[:, :, 1:, :], dim=-1)
            left = (
                left_probabilities
                * self.left_indices.to(left_probabilities)[None, None, None]
            ).sum(dim=-1)
            right = (
                right_probabilities
                * self.right_indices.to(right_probabilities)[None, None, None]
            ).sum(dim=-1)
            candidate = torch.stack((left, right), dim=-1)

            coarse_classwise = coarse.permute(0, 2, 1).unsqueeze(2)
            gate = torch.tanh(self.gates[level]).reshape(1, 1, 1, 2)
            output = F.relu(
                coarse_classwise + gate * (candidate - coarse_classwise))
            valid = mask.permute(0, 2, 1).to(output.dtype)
            output = output * valid.unsqueeze(-1)
            candidate = candidate * valid.unsqueeze(-1)

            outputs.append(output)
            candidates.append(candidate)
            center_logits_all.append(center_logits)
            start_logits_all.append(start_logits.permute(0, 2, 1))
            end_logits_all.append(end_logits.permute(0, 2, 1))

        return (
            tuple(outputs),
            tuple(candidates),
            tuple(center_logits_all),
            tuple(start_logits_all),
            tuple(end_logits_all),
        )


class DenseIoUQualityHead(nn.Module):
    """Predict dense class-aware localization quality and calibrate scores.

    The head observes the frozen regression feature together with the decoded
    class-aware boundary geometry.  Its per-level calibration gates are
    zero-initialized, so enabling the module initially leaves every detector
    classification logit bit-for-bit unchanged.
    """

    def __init__(
            self,
            feature_dim,
            num_classes,
            fpn_levels,
            hidden_dim=128,
            num_layers=2,
            kernel_size=3,
            dropout=0.0,
            prior_prob=0.01,
            gate_init=0.0,
            max_logit_shift=0.5,
            score_fusion="bounded_logit_residual",
            score_fusion_beta=0.0):
        super().__init__()
        if min(feature_dim, num_classes, fpn_levels, hidden_dim) <= 0:
            raise ValueError("dense quality dimensions must be positive")
        if num_layers <= 0:
            raise ValueError("dense quality num_layers must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "dense quality kernel_size must be positive and odd")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dense quality dropout must be in [0, 1)")
        if not 0.0 < prior_prob < 1.0:
            raise ValueError("dense quality prior_prob must be in (0, 1)")
        if max_logit_shift <= 0.0:
            raise ValueError(
                "dense quality max_logit_shift must be positive")
        if score_fusion not in (
                "bounded_logit_residual", "multiplicative_quality"):
            raise ValueError("unsupported dense quality score fusion")
        if not 0.0 <= float(score_fusion_beta) <= 1.0:
            raise ValueError("score_fusion_beta must be in [0, 1]")

        self.num_classes = int(num_classes)
        self.fpn_levels = int(fpn_levels)
        self.max_logit_shift = float(max_logit_shift)
        self.score_fusion = str(score_fusion)
        self.score_fusion_beta = float(score_fusion_beta)
        self.activation = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(float(dropout))

        input_dim = int(feature_dim) + 2 * self.num_classes
        self.tower = nn.ModuleList()
        self.norm = nn.ModuleList()
        for layer in range(int(num_layers)):
            layer_input_dim = input_dim if layer == 0 else int(hidden_dim)
            self.tower.append(MaskedConv1D(
                layer_input_dim, int(hidden_dim), kernel_size,
                padding=kernel_size // 2, bias=False))
            self.norm.append(LayerNorm(int(hidden_dim)))
        self.output = MaskedConv1D(
            int(hidden_dim), self.num_classes, 1, padding=0)
        nn.init.zeros_(self.output.conv.weight)
        nn.init.constant_(
            self.output.conv.bias,
            -math.log((1.0 - float(prior_prob)) / float(prior_prob)))
        self.score_gates = nn.Parameter(torch.full(
            (self.fpn_levels, 1, 1), float(gate_init)))

    def _classwise_geometry(self, offsets, temporal_length):
        if offsets.ndim == 4:
            if offsets.shape[2:] != (self.num_classes, 2):
                raise ValueError(
                    "class-aware offsets must have shape [B, T, C, 2]")
            if offsets.shape[1] != temporal_length:
                raise ValueError("quality geometry length mismatch")
            geometry = offsets.detach().permute(0, 2, 3, 1)
        elif offsets.ndim == 3:
            if offsets.shape[1] == 2:
                if offsets.shape[2] != temporal_length:
                    raise ValueError("quality geometry length mismatch")
                geometry = offsets.detach().unsqueeze(1).expand(
                    -1, self.num_classes, -1, -1)
            elif offsets.shape[2] == 2:
                if offsets.shape[1] != temporal_length:
                    raise ValueError("quality geometry length mismatch")
                geometry = offsets.detach().permute(0, 2, 1).unsqueeze(
                    1).expand(-1, self.num_classes, -1, -1)
            else:
                raise ValueError(
                    "shared offsets must have one boundary dimension of 2")
        else:
            raise ValueError("unexpected quality offset rank")
        return torch.log1p(geometry.clamp_min(0.0)).reshape(
            geometry.shape[0], 2 * self.num_classes, temporal_length)

    def forward(self, regression_features, fpn_masks, offsets):
        if not (
                len(regression_features) == len(fpn_masks) == len(offsets)
                == self.fpn_levels):
            raise ValueError(
                "unexpected number of dense quality pyramid levels")
        outputs = []
        for feature, mask, boundary in zip(
                regression_features, fpn_masks, offsets):
            geometry = self._classwise_geometry(
                boundary, feature.shape[-1]).to(feature)
            hidden = torch.cat((feature.detach(), geometry), dim=1)
            for convolution, normalization in zip(self.tower, self.norm):
                hidden, _ = convolution(hidden, mask)
                hidden = self.dropout(
                    self.activation(normalization(hidden)))
            logits, _ = self.output(hidden, mask)
            outputs.append(logits)
        return tuple(outputs)

    def calibrate_logits(self, class_logits, quality_logits):
        if not (
                len(class_logits) == len(quality_logits)
                == self.fpn_levels):
            raise ValueError(
                "unexpected number of score calibration levels")
        outputs = []
        for level, (class_logit, quality_logit) in enumerate(zip(
                class_logits, quality_logits)):
            if class_logit.shape != quality_logit.shape:
                raise ValueError(
                    "class and quality logits must have matching shapes")
            if self.score_fusion == "multiplicative_quality":
                if self.score_fusion_beta == 0.0:
                    # P2A-I0 identity: preserve the original tensor and every
                    # downstream threshold/top-k decision exactly.
                    outputs.append(class_logit)
                    continue
                log_probability = (
                    F.logsigmoid(class_logit.float())
                    + self.score_fusion_beta
                    * F.logsigmoid(quality_logit.float()))
                # Convert p * q**beta back to logits without producing
                # infinities for saturated mixed-precision probabilities.
                maximum = math.log1p(-1e-6)
                log_probability = log_probability.clamp(max=maximum)
                fused_logit = (
                    log_probability
                    - torch.log1p(-log_probability.exp()))
                outputs.append(fused_logit.to(class_logit.dtype))
                continue
            gate = torch.tanh(self.score_gates[level]).to(class_logit)
            quality_direction = (
                2.0 * quality_logit.sigmoid() - 1.0)
            outputs.append(
                class_logit
                + gate * self.max_logit_shift * quality_direction)
        return tuple(outputs)


class A0PreservingResidualAlignmentHead(nn.Module):
    """Learn bounded detector corrections while retaining the A0 fallback.

    The original classification and regression heads remain frozen.  This
    branch observes their detached regression features and predicts bounded
    logit/offset residuals plus separate confidence gates.  Both residual
    projections are zero initialized, so enabling the branch is an exact
    identity before optimization.  During inference a correction is applied
    only when its confidence passes the fixed threshold; otherwise the exact
    A0 tensor is returned for that point.
    """

    def __init__(
            self, feature_dim, num_classes, fpn_levels, hidden_dim=128,
            num_layers=2, kernel_size=3, dropout=0.1,
            maximum_class_logit_residual=0.5,
            maximum_regression_residual=0.5,
            residual_weight=1.0, minimum_confidence=0.5,
            confidence_prior=0.1):
        super().__init__()
        if min(
                feature_dim, num_classes, fpn_levels,
                hidden_dim, num_layers) <= 0:
            raise ValueError(
                "residual alignment dimensions must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "residual alignment kernel_size must be positive and odd")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "residual alignment dropout must be in [0, 1)")
        if maximum_class_logit_residual <= 0.0 \
                or maximum_regression_residual <= 0.0:
            raise ValueError(
                "residual alignment bounds must be positive")
        if not 0.0 <= residual_weight <= 1.0:
            raise ValueError(
                "residual alignment weight must be in [0, 1]")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError(
                "residual alignment confidence must be in [0, 1]")
        if not 0.0 < confidence_prior < 1.0:
            raise ValueError(
                "residual alignment confidence_prior must be in (0, 1)")

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.fpn_levels = int(fpn_levels)
        self.maximum_class_logit_residual = float(
            maximum_class_logit_residual)
        self.maximum_regression_residual = float(
            maximum_regression_residual)
        self.residual_weight = float(residual_weight)
        self.minimum_confidence = float(minimum_confidence)
        self.activation = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(float(dropout))

        self.tower = nn.ModuleList()
        self.norm = nn.ModuleList()
        for layer in range(int(num_layers)):
            input_dim = self.feature_dim if layer == 0 else int(hidden_dim)
            self.tower.append(MaskedConv1D(
                input_dim, int(hidden_dim), kernel_size,
                padding=kernel_size // 2, bias=False))
            self.norm.append(LayerNorm(int(hidden_dim)))

        self.class_residual = MaskedConv1D(
            int(hidden_dim), self.num_classes, 1, padding=0)
        self.regression_residual = MaskedConv1D(
            int(hidden_dim), 2, 1, padding=0)
        self.class_confidence = MaskedConv1D(
            int(hidden_dim), self.num_classes, 1, padding=0)
        self.regression_confidence = MaskedConv1D(
            int(hidden_dim), 1, 1, padding=0)

        for projection in (
                self.class_residual, self.regression_residual):
            nn.init.zeros_(projection.conv.weight)
            nn.init.zeros_(projection.conv.bias)
        prior_bias = math.log(
            float(confidence_prior) / (1.0 - float(confidence_prior)))
        for projection in (
                self.class_confidence, self.regression_confidence):
            nn.init.zeros_(projection.conv.weight)
            nn.init.constant_(projection.conv.bias, prior_bias)

    def forward(
            self, regression_features, fpn_masks,
            baseline_class_logits, baseline_offsets,
            apply_confidence=True):
        if not (
                len(regression_features) == len(fpn_masks)
                == len(baseline_class_logits) == len(baseline_offsets)
                == self.fpn_levels):
            raise ValueError(
                "unexpected number of residual alignment pyramid levels")

        aligned_logits, aligned_offsets = [], []
        class_residuals, regression_residuals = [], []
        class_confidences, regression_confidences = [], []
        for feature, mask, base_logit, base_offset in zip(
                regression_features, fpn_masks,
                baseline_class_logits, baseline_offsets):
            if feature.ndim != 3 or feature.shape[1] != self.feature_dim:
                raise ValueError(
                    "residual alignment features have invalid shape")
            if base_logit.shape[:2] != (
                    feature.shape[0], self.num_classes) \
                    or base_logit.shape[-1] != feature.shape[-1]:
                raise ValueError(
                    "residual alignment class logits have invalid shape")
            if base_offset.shape != (
                    feature.shape[0], 2, feature.shape[-1]):
                raise ValueError(
                    "residual alignment offsets have invalid shape")

            hidden = feature.detach()
            for convolution, normalization in zip(
                    self.tower, self.norm):
                hidden, _ = convolution(hidden, mask)
                hidden = self.dropout(
                    self.activation(normalization(hidden)))
            class_delta, _ = self.class_residual(hidden, mask)
            regression_delta, _ = self.regression_residual(hidden, mask)
            class_confidence, _ = self.class_confidence(hidden, mask)
            regression_confidence, _ = self.regression_confidence(
                hidden, mask)
            class_delta = torch.tanh(class_delta) * \
                self.maximum_class_logit_residual
            regression_delta = torch.tanh(regression_delta) * \
                self.maximum_regression_residual

            if self.residual_weight == 0.0:
                # Preserve tensor identity, not merely numerical closeness.
                class_output = base_logit
                regression_output = base_offset
            else:
                if apply_confidence:
                    class_gate = (
                        class_confidence.sigmoid()
                        >= self.minimum_confidence).to(class_delta.dtype)
                    regression_gate = (
                        regression_confidence.sigmoid()
                        >= self.minimum_confidence).to(
                            regression_delta.dtype)
                else:
                    class_gate = torch.ones_like(class_delta)
                    regression_gate = torch.ones_like(regression_delta)
                class_output = (
                    base_logit
                    + self.residual_weight * class_gate * class_delta)
                regression_output = (
                    base_offset
                    + self.residual_weight
                    * regression_gate * regression_delta).clamp_min(0.0)

            aligned_logits.append(class_output)
            aligned_offsets.append(regression_output)
            class_residuals.append(class_delta)
            regression_residuals.append(regression_delta)
            class_confidences.append(class_confidence)
            regression_confidences.append(regression_confidence)

        return (
            tuple(aligned_logits), tuple(aligned_offsets),
            tuple(class_residuals), tuple(regression_residuals),
            tuple(class_confidences), tuple(regression_confidences),
        )


class ClusterProposalSelectorHead(nn.Module):
    """Predict relative gain for an existing proposal against its anchor."""

    def __init__(
            self, feature_dim, fpn_levels, hidden_dim=128,
            level_embedding_dim=16, geometry_dim=8, dropout=0.1,
            maximum_gain=1.0, replacement_prior=0.01,
            ap_listwise=False):
        super().__init__()
        if min(
                feature_dim, fpn_levels, hidden_dim,
                level_embedding_dim, geometry_dim) <= 0:
            raise ValueError("cluster selector dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("cluster selector dropout must be in [0, 1)")
        if maximum_gain <= 0.0:
            raise ValueError("cluster selector maximum_gain must be positive")
        if not 0.0 < replacement_prior < 1.0:
            raise ValueError(
                "cluster selector replacement_prior must be in (0, 1)")
        self.feature_dim = int(feature_dim)
        self.fpn_levels = int(fpn_levels)
        self.geometry_dim = int(geometry_dim)
        self.maximum_gain = float(maximum_gain)
        self.ap_listwise = bool(ap_listwise)
        self.level_embedding = nn.Embedding(
            self.fpn_levels, int(level_embedding_dim))
        self.feature_encoder = nn.Sequential(
            nn.Linear(3 * self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(float(dropout)),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(6 * self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(float(dropout)),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(
                self.geometry_dim + 2 * int(level_embedding_dim),
                hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(float(dropout)),
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(float(dropout)),
        )
        self.gain_output = nn.Linear(hidden_dim, 1)
        self.confidence_output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gain_output.weight)
        nn.init.zeros_(self.gain_output.bias)
        nn.init.zeros_(self.confidence_output.weight)
        prior_bias = math.log(
            float(replacement_prior) / (1.0 - float(replacement_prior)))
        nn.init.constant_(self.confidence_output.bias, prior_bias)

    def forward(
            self, anchor_features, candidate_features,
            anchor_context, candidate_context, geometry,
            anchor_levels, candidate_levels):
        count = anchor_features.shape[0]
        expected_feature = (count, self.feature_dim)
        expected_context = (count, 2 * self.feature_dim)
        if anchor_features.shape != expected_feature \
                or candidate_features.shape != expected_feature:
            raise ValueError(
                "cluster selector proposal features have invalid shape")
        if anchor_context.shape != expected_context \
                or candidate_context.shape != expected_context:
            raise ValueError(
                "cluster selector boundary context has invalid shape")
        if geometry.shape != (count, self.geometry_dim):
            raise ValueError(
                "cluster selector geometry has invalid shape")
        if anchor_levels.shape != (count,) \
                or candidate_levels.shape != (count,):
            raise ValueError(
                "cluster selector levels must have shape [N]")
        feature_input = torch.cat((
            anchor_features, candidate_features,
            candidate_features - anchor_features), dim=-1)
        context_input = torch.cat((
            anchor_context, candidate_context,
            candidate_context - anchor_context), dim=-1)
        geometry_input = torch.cat((
            geometry,
            self.level_embedding(anchor_levels),
            self.level_embedding(candidate_levels)), dim=-1)
        hidden = self.fusion(torch.cat((
            self.feature_encoder(feature_input),
            self.context_encoder(context_input),
            self.geometry_encoder(geometry_input)), dim=-1))
        raw_gain = self.gain_output(hidden).squeeze(-1)
        # P2-D1 regresses a bounded tIoU delta.  P5 instead treats this output
        # as an unbounded ranking logit and only consumes differences inside a
        # cluster.  Both modes retain the same zero initialization, so the
        # anchor is the exact fallback before the new head learns anything.
        gain = (
            raw_gain if self.ap_listwise
            else torch.tanh(raw_gain) * self.maximum_gain)
        confidence_logit = self.confidence_output(hidden).squeeze(-1)
        return gain, confidence_logit


class ClusterScoreSuppressionHead(nn.Module):
    """Predict a bounded, downward-only score-logit adjustment.

    The head operates on the score-anchored representative of a local
    proposal cluster.  It never increases a proposal score; the caller keeps
    the unmodified anchor as the fallback whenever the confidence gate or the
    fixed suppression budget rejects the prediction.
    """

    def __init__(
            self, feature_dim, fpn_levels, hidden_dim=128,
            level_embedding_dim=16, dropout=0.1,
            maximum_logit_suppression=1.0, suppression_prior=0.05):
        super().__init__()
        if min(
                feature_dim, fpn_levels, hidden_dim,
                level_embedding_dim) <= 0:
            raise ValueError(
                "score suppression dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "score suppression dropout must be in [0, 1)")
        if maximum_logit_suppression <= 0.0:
            raise ValueError(
                "maximum_logit_suppression must be positive")
        if not 0.0 < suppression_prior < 1.0:
            raise ValueError(
                "suppression_prior must be in (0, 1)")
        self.feature_dim = int(feature_dim)
        self.maximum_logit_suppression = float(
            maximum_logit_suppression)
        self.level_embedding = nn.Embedding(
            int(fpn_levels), int(level_embedding_dim))
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(float(dropout)),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(2 * self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(float(dropout)),
        )
        # score, score logit, and log duration are the only scalar geometry
        # inputs.  No ground-truth quantity can enter the inference head.
        self.geometry_encoder = nn.Sequential(
            nn.Linear(3 + int(level_embedding_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(float(dropout)),
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(float(dropout)),
        )
        self.amount_output = nn.Linear(hidden_dim, 1)
        self.confidence_output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.amount_output.weight)
        nn.init.zeros_(self.amount_output.bias)
        nn.init.zeros_(self.confidence_output.weight)
        prior_bias = math.log(
            float(suppression_prior) / (1.0 - float(suppression_prior)))
        nn.init.constant_(self.confidence_output.bias, prior_bias)

    def forward(self, features, context, scores, durations, levels):
        count = features.shape[0]
        if features.shape != (count, self.feature_dim):
            raise ValueError(
                "score suppression features have invalid shape")
        if context.shape != (count, 2 * self.feature_dim):
            raise ValueError(
                "score suppression context has invalid shape")
        if scores.shape != (count,) or durations.shape != (count,):
            raise ValueError(
                "score suppression scalars must have shape [N]")
        if levels.shape != (count,):
            raise ValueError(
                "score suppression levels must have shape [N]")
        safe_scores = scores.clamp(min=1e-6, max=1.0 - 1e-6)
        geometry = torch.cat((
            safe_scores[:, None],
            torch.logit(safe_scores)[:, None],
            durations.clamp_min(1e-6).log()[:, None],
            self.level_embedding(levels),
        ), dim=-1)
        hidden = self.fusion(torch.cat((
            self.feature_encoder(features),
            self.context_encoder(context),
            self.geometry_encoder(geometry),
        ), dim=-1))
        amount = (
            self.amount_output(hidden).squeeze(-1).sigmoid()
            * self.maximum_logit_suppression)
        confidence_logit = self.confidence_output(hidden).squeeze(-1)
        return amount, confidence_logit


class RegressionSGPAdapter(nn.Module):
    """Lightweight SGP-style residual adapter for the regression path only.

    The adapter retains DyFADet's DynE encoder and classification features.
    It uses local/expanded depthwise temporal aggregation plus a global
    channel gate, followed by a compact point-wise MLP.  Per-level
    zero-initialized gates make its initial output exactly equal to its input.
    """

    def __init__(
            self,
            feature_dim,
            fpn_levels,
            hidden_dim=256,
            kernel_size=3,
            expansion=1.5,
            dropout=0.0,
            gate_init=0.0):
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or fpn_levels <= 0:
            raise ValueError("SGP adapter dimensions must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("SGP adapter kernel_size must be positive and odd")
        if expansion <= 0.0:
            raise ValueError("SGP adapter expansion must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("SGP adapter dropout must be in [0, 1)")

        expanded_kernel = int(round((kernel_size + 1) * expansion))
        if expanded_kernel % 2 == 0:
            expanded_kernel += 1
        self.feature_dim = int(feature_dim)
        self.fpn_levels = int(fpn_levels)
        self.norm = LayerNorm(self.feature_dim)
        self.local_conv = nn.Conv1d(
            self.feature_dim, self.feature_dim, kernel_size,
            padding=kernel_size // 2, groups=self.feature_dim)
        self.expanded_conv = nn.Conv1d(
            self.feature_dim, self.feature_dim, expanded_kernel,
            padding=expanded_kernel // 2, groups=self.feature_dim)
        self.local_gate = nn.Conv1d(
            self.feature_dim, self.feature_dim, kernel_size,
            padding=kernel_size // 2, groups=self.feature_dim)
        self.global_gate = nn.Conv1d(
            self.feature_dim, self.feature_dim, 1,
            groups=self.feature_dim)
        group_count = 16 if self.feature_dim % 16 == 0 else 1
        self.mlp_norm = nn.GroupNorm(group_count, self.feature_dim)
        self.mlp = nn.Sequential(
            nn.Conv1d(self.feature_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(hidden_dim, self.feature_dim, 1),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.gates = nn.Parameter(torch.full(
            (self.fpn_levels, self.feature_dim, 1), float(gate_init)))

    def forward(self, features, masks):
        if len(features) != self.fpn_levels or len(masks) != self.fpn_levels:
            raise ValueError("unexpected number of SGP adapter levels")
        outputs = []
        for level, (feature, mask) in enumerate(zip(features, masks)):
            if feature.shape[1] != self.feature_dim:
                raise ValueError("unexpected SGP adapter feature dimension")
            valid = mask.to(feature.dtype)
            normalized = self.norm(feature)
            local = self.local_conv(normalized)
            expanded = self.expanded_conv(normalized)
            local_weight = torch.sigmoid(self.local_gate(normalized))
            global_weight = torch.sigmoid(self.global_gate(
                normalized.mean(dim=-1, keepdim=True)))
            aggregated = (
                local_weight * (local + expanded)
                + global_weight * normalized)
            delta = aggregated + self.mlp(self.mlp_norm(aggregated))
            delta = self.dropout(delta) * valid
            gate = torch.tanh(self.gates[level])[None]
            outputs.append((feature + gate * delta) * valid)
        return tuple(outputs)


class TDynPtTransformerClsHead(nn.Module):
    """
    Shared 1D MSDy-head for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=5,
        act_layer=nn.ReLU,
        empty_cls = [],
        gate_activation_kargs: dict = None,
        selective_dynamic_scale: dict = None,
    ):
        super().__init__()
        self.act = act_layer()
        
        assert num_layers-1 >0

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()

        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            
            cls_subnet_conv = DTFAM(dim=in_dim, o_dim= feat_dim, ka=kernel_size, conv_type = 'others', gate_activation=gate_activation_kargs["type"],
                gate_activation_kargs = gate_activation_kargs)

            selective_cfg = selective_dynamic_scale or {"enabled": False}
            active_layers = tuple(int(layer) for layer in
                                  selective_cfg.get("active_layers", (0, 1)))
            scale_module = (SelectiveDynamicScale
                            if selective_cfg.get("enabled", False)
                            and idx in active_layers
                            else DynamicScale_chk)
            scale_kwargs = {}
            if scale_module is SelectiveDynamicScale:
                scale_kwargs = {
                    "reduction": int(selective_cfg.get("reduction", 8)),
                    "min_hidden_dim": int(
                        selective_cfg.get("min_hidden_dim", 32)),
                    "channel_wise": bool(
                        selective_cfg.get("channel_wise", True)),
                    "temperature": float(
                        selective_cfg.get("temperature", 1.0)),
                    "selection_gate_init": float(
                        selective_cfg.get("selection_gate_init", 0.0)),
                    "selection_gate_mode": str(
                        selective_cfg.get("selection_gate_mode", "tanh")),
                    "selection_gate_bound": float(
                        selective_cfg.get("selection_gate_bound", 0.25)),
                    "selection_gate_lr_scale": float(
                        selective_cfg.get("selection_gate_lr_scale", 1.0)),
                    "preserve_parent_rng": bool(
                        selective_cfg.get("preserve_parent_rng", True)),
                }
            self.head.append(
                scale_module(
                in_dim,
                out_dim,
                num_convs=1,
                kernel_size=kernel_size,
                padding=1,
                stride=kernel_size // 2,
                num_groups=1,
                num_adjacent_scales=2,
                depth_module=cls_subnet_conv,
                gate_activation=gate_activation_kargs["type"],
                gate_activation_kargs=gate_activation_kargs,
                **scale_kwargs))

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        feats = fpn_feats
        for i in range(len(self.head)):
            feats, fpn_masks = self.head[i](feats, fpn_masks)
            
            for j in range(len(feats)):
                feats[j] =  self.act(feats[j])

        for cur_out, cur_mask in zip(feats, fpn_masks):
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        return out_logits

class TDynPtTransformerRegHead(nn.Module):
    """
    Shared 1D MSDy-head for regression
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=5,
        act_layer=nn.ReLU,
        gate_activation_kargs: dict = None,
        selective_dynamic_scale: dict = None,
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()
        
        assert num_layers-1 > 0

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            
            reg_subnet_conv = DTFAM(dim=in_dim, o_dim= feat_dim, ka=kernel_size, conv_type = 'others', gate_activation=gate_activation_kargs["type"],
                gate_activation_kargs = gate_activation_kargs)

            selective_cfg = selective_dynamic_scale or {"enabled": False}
            active_layers = tuple(int(layer) for layer in
                                  selective_cfg.get("active_layers", (0, 1)))
            scale_module = (SelectiveDynamicScale
                            if selective_cfg.get("enabled", False)
                            and idx in active_layers
                            else DynamicScale_chk)
            scale_kwargs = {}
            if scale_module is SelectiveDynamicScale:
                scale_kwargs = {
                    "reduction": int(selective_cfg.get("reduction", 8)),
                    "min_hidden_dim": int(
                        selective_cfg.get("min_hidden_dim", 32)),
                    "channel_wise": bool(
                        selective_cfg.get("channel_wise", True)),
                    "temperature": float(
                        selective_cfg.get("temperature", 1.0)),
                    "selection_gate_init": float(
                        selective_cfg.get("selection_gate_init", 0.0)),
                    "selection_gate_mode": str(
                        selective_cfg.get("selection_gate_mode", "tanh")),
                    "selection_gate_bound": float(
                        selective_cfg.get("selection_gate_bound", 0.25)),
                    "selection_gate_lr_scale": float(
                        selective_cfg.get("selection_gate_lr_scale", 1.0)),
                    "preserve_parent_rng": bool(
                        selective_cfg.get("preserve_parent_rng", True)),
                }
            self.head.append(scale_module(
                in_dim,
                out_dim,
                num_convs=1,
                kernel_size=kernel_size,
                padding=1,
                stride=kernel_size // 2,
                num_groups=1,
                num_adjacent_scales=2,
                depth_module=reg_subnet_conv,
                gate_activation=gate_activation_kargs['type'],
                gate_activation_kargs=gate_activation_kargs,
                **scale_kwargs)
                )

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )


    def forward(self, fpn_feats, fpn_masks, return_features=False):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels
        
        # apply the classifier for each pyramid level
        out_offsets = tuple()
        feats = fpn_feats
        for i in range(len(self.head)):
            
            feats, fpn_masks = self.head[i](feats, fpn_masks)
            for j in range(len(feats)):
                feats[j] = self.act(feats[j])
        
        for l in range(self.fpn_levels):
            cur_offsets, _  = self.offset_head(feats[l], fpn_masks[l])            
            out_offsets +=  ( F.relu(self.scale[l](cur_offsets)) , )
               
        return (out_offsets, tuple(feats)) if return_features else out_offsets
