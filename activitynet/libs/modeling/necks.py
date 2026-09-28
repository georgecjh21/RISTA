import torch
from torch import nn
from torch.nn import functional as F

from .models import register_neck
from .blocks import MaskedConv1D, LayerNorm


@register_neck("fpn")
class FPN1D(nn.Module):
    """
        Feature pyramid network
    """
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True      # if to apply layer norm at the end
    ):
        super().__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # disable bias if using layer norm
            l_conv = MaskedConv1D(
                in_channels[i], out_channel, 1, bias=(not with_ln))
            # use depthwise conv here for efficiency
            fpn_conv = MaskedConv1D(
                out_channel, out_channel, 3,
                padding=1, bias=(not with_ln), groups=out_channel
            )
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # build laterals, fpn_masks will remain the same with 1x1 convs
        laterals = []
        for i in range(len(self.lateral_convs)):
            x, _ = self.lateral_convs[i](
                inputs[i + self.start_level], fpn_masks[i + self.start_level]
            )
            laterals.append(x)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i-1] += F.interpolate(
                laterals[i],
                scale_factor=self.scale_factor,
                mode='nearest'
            )

        # fpn conv / norm -> outputs
        # mask will remain the same
        fpn_feats = tuple()
        for i in range(used_backbone_levels):
            x, _ = self.fpn_convs[i](
                laterals[i], fpn_masks[i + self.start_level])
            x = self.fpn_norms[i](x)
            fpn_feats += (x, )

        return fpn_feats, fpn_masks

@register_neck('identity')
class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True      # if to apply layer norm at the end
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i + self.start_level] == self.out_channel
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # apply norms, fpn_masks will remain the same with 1x1 convs
        fpn_feats = tuple()
        for i in range(len(self.fpn_norms)):
            x = self.fpn_norms[i](inputs[i + self.start_level])
            fpn_feats += (x, )

        return fpn_feats, fpn_masks


@register_neck("regression_boundary")
class RegressionBoundaryPyramid(nn.Module):
    """Identity-preserving boundary residuals for the regression branch.

    The original normalized pyramid remains the main path.  Each level builds
    a lightweight residual from its first/second temporal differences and the
    contrast to the next coarser level.  A zero-initialized, per-channel gate
    makes the module an exact identity at initialization:

        output = input + tanh(gate) * boundary_residual

    Classification features never pass through this module.
    """

    def __init__(
            self,
            in_channels,
            out_channel,
            scale_factor=2.0,
            start_level=0,
            end_level=-1,
            with_ln=True,
            hidden_dim=128,
            dropout=0.0,
            gate_init=0.0):
        super().__init__()
        if not isinstance(in_channels, (list, tuple)):
            raise TypeError("in_channels must be a list or tuple")
        if end_level == -1:
            end_level = len(in_channels)
        if start_level != 0 or end_level != len(in_channels):
            raise ValueError(
                "regression boundary pyramid must cover every FPN level")
        if any(channel != out_channel for channel in in_channels):
            raise ValueError(
                "regression boundary pyramid requires equal input/output dims")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.in_channels = tuple(in_channels)
        self.out_channel = int(out_channel)
        self.scale_factor = float(scale_factor)
        self.hidden_dim = int(hidden_dim)
        self.num_levels = len(in_channels)

        self.input_projections = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        self.temporal_convs = nn.ModuleList()
        self.output_projections = nn.ModuleList()
        for _ in range(self.num_levels):
            self.input_projections.append(MaskedConv1D(
                3 * self.out_channel, self.hidden_dim, 1, bias=False))
            self.hidden_norms.append(
                LayerNorm(self.hidden_dim) if with_ln else nn.Identity())
            self.temporal_convs.append(MaskedConv1D(
                self.hidden_dim, self.hidden_dim, 3, padding=1,
                groups=self.hidden_dim, bias=False))
            self.output_projections.append(MaskedConv1D(
                self.hidden_dim, self.out_channel, 1, bias=False))

        self.activation = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout)
        self.gates = nn.Parameter(torch.full(
            (self.num_levels, self.out_channel, 1), float(gate_init)))

    @staticmethod
    def _temporal_differences(features):
        padded = F.pad(features, (1, 1), mode="replicate")
        left = padded[..., :-2]
        right = padded[..., 2:]
        first = 0.5 * (right - left)
        second = right - 2.0 * features + left
        return first, second

    def forward(self, inputs, fpn_masks):
        if len(inputs) != self.num_levels:
            raise ValueError("unexpected number of regression feature levels")
        if len(fpn_masks) != self.num_levels:
            raise ValueError("unexpected number of regression masks")

        outputs = []
        for level, (features, mask) in enumerate(zip(inputs, fpn_masks)):
            if features.shape[1] != self.out_channel:
                raise ValueError("unexpected regression feature dimension")
            first, second = self._temporal_differences(features)
            if level + 1 < self.num_levels:
                coarse = F.interpolate(
                    inputs[level + 1], size=features.shape[-1],
                    mode="nearest")
                cross_level = coarse - features
            else:
                cross_level = torch.zeros_like(features)

            evidence = torch.cat((first, second, cross_level), dim=1)
            residual, _ = self.input_projections[level](evidence, mask)
            residual = self.activation(self.hidden_norms[level](residual))
            residual, _ = self.temporal_convs[level](residual, mask)
            residual = self.activation(residual)
            residual = self.dropout(residual)
            residual, _ = self.output_projections[level](residual, mask)

            gate = torch.tanh(self.gates[level]).unsqueeze(0)
            output = features + gate * residual
            output = output * mask.to(output.dtype).detach()
            outputs.append(output)

        return tuple(outputs), fpn_masks
