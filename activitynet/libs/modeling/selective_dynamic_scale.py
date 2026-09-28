import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import DTFAM, LayerNorm


class SelectiveDynamicScale(nn.Module):
    """Mask-aware selective fusion over adjacent temporal pyramid levels.

    The alignment path intentionally matches :class:`DynamicScale_chk`: the
    finer and coarser neighbours are transformed by DTFAM and resized to the
    current level.  Only the fusion rule changes.  A SKNet-style squeeze MLP
    predicts per-sample, per-channel weights over ``[l - 1, l, l + 1]`` and a
    softmax normalizes the available branches.  Missing boundary branches are
    excluded from the softmax rather than represented by zero features.

    ``selection_gate`` uses a checkpoint-preserving residual parameterization::

        legacy = scale_weight * sum(aligned_neighbours) \
                 + output_weight * current
        output = legacy + gate * (selected - legacy)

    The legacy scalar weights and normalization parameters remain in the
    module so they are restored from a trained ``DynamicScale_chk`` checkpoint.
    A zero gate therefore reproduces that trained parent path exactly while
    still allowing the module to learn the full selective replacement.

    ``selection_gate_mode`` controls how the effective gate is obtained:

    * ``unbounded`` preserves every historical P8-SDS experiment.
    * ``fixed`` keeps the configured effective gate constant.
    * ``tanh`` learns a signed gate bounded to
      ``[-selection_gate_bound, selection_gate_bound]``.

    The parameter retains the historical ``selection_gate`` state-dict name.
    In ``tanh`` mode it stores the unconstrained raw value; diagnostics must
    use :meth:`effective_selection_gate` to report the applied coefficient.
    """

    _NUM_BRANCHES = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 1,
        kernel_size: int = 1,
        padding: int = 0,
        stride: int = 1,
        num_groups: int = 1,
        num_adjacent_scales: int = 2,
        depth_module: nn.Module = None,
        norm: str = "GN",
        gate_activation: str = "ReTanH",
        gate_activation_kargs: dict = None,
        reduction: int = 8,
        min_hidden_dim: int = 32,
        channel_wise: bool = True,
        temperature: float = 1.0,
        selection_gate_init: float = 0.0,
        selection_gate_mode: str = "unbounded",
        selection_gate_bound: float = 0.25,
        selection_gate_lr_scale: float = 1.0,
        preserve_parent_rng: bool = False,
    ):
        super().__init__()
        del num_convs, padding, stride, norm, gate_activation
        if num_adjacent_scales != 2:
            raise ValueError(
                "SelectiveDynamicScale requires exactly two adjacent scales")
        if reduction <= 0 or min_hidden_dim <= 0:
            raise ValueError("reduction and min_hidden_dim must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if selection_gate_mode not in ("unbounded", "fixed", "tanh"):
            raise ValueError(
                "selection_gate_mode must be unbounded, fixed, or tanh")
        if selection_gate_bound <= 0.0:
            raise ValueError("selection_gate_bound must be positive")
        if selection_gate_lr_scale <= 0.0:
            raise ValueError("selection_gate_lr_scale must be positive")
        if selection_gate_mode == "tanh" \
                and abs(selection_gate_init) >= selection_gate_bound:
            raise ValueError(
                "bounded selection gate init must lie strictly inside bound")

        self.num_groups = num_groups
        self.num_adjacent_scales = num_adjacent_scales
        self.depth_module = depth_module
        self.channel_wise = bool(channel_wise)
        self.temperature = float(temperature)
        self.selection_gate_mode = str(selection_gate_mode)
        self.selection_gate_bound = float(selection_gate_bound)
        self.selection_gate_lr_scale = float(selection_gate_lr_scale)
        self.preserve_parent_rng = bool(preserve_parent_rng)

        self.dynamic_convs = nn.ModuleList([
            DTFAM(
                dim=in_channels,
                o_dim=in_channels,
                ka=kernel_size,
                gate_activation="GeReTanH",
                gate_activation_kargs=gate_activation_kargs,
            )
            for _ in range(num_adjacent_scales)
        ])

        hidden_dim = max(int(min_hidden_dim), in_channels // int(reduction))
        attention_channels = in_channels if self.channel_wise else 1
        selector_rng_context = (
            torch.random.fork_rng(devices=[])
            if self.preserve_parent_rng else nullcontext())
        with selector_rng_context:
            self.selector = nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    hidden_dim,
                    self._NUM_BRANCHES * attention_channels,
                    kernel_size=1,
                ),
            )
        self.selection_gate_init = float(selection_gate_init)
        if self.selection_gate_mode == "tanh":
            normalized_init = (
                self.selection_gate_init / self.selection_gate_bound)
            selection_gate_parameter_init = math.atanh(normalized_init)
        else:
            selection_gate_parameter_init = self.selection_gate_init
        self.selection_gate_parameter_init = float(
            selection_gate_parameter_init)
        self.selection_gate = nn.Parameter(torch.tensor(
            self.selection_gate_parameter_init, dtype=torch.float32),
            requires_grad=self.selection_gate_mode != "fixed")
        self.selection_gate._lr_scale = self.selection_gate_lr_scale
        if self.selection_gate.requires_grad \
                and self.selection_gate_lr_scale != 1.0:
            self.selection_gate.register_hook(
                lambda gradient: gradient * self.selection_gate_lr_scale)
        # Detached Bx3 summaries from the latest forward pass.  They make the
        # learned scale preference auditable without changing outputs/state.
        self.last_selection_weights = None
        self.resize = lambda feature, size: F.interpolate(
            feature, size=size, mode="nearest")

        # Preserve the complete legacy DynamicScale_chk state.  These names
        # must remain checkpoint-compatible: a zero selection gate is defined
        # as the trained parent output, not merely its initial identity state.
        self.scale_weight = nn.Parameter(torch.zeros(1))
        self.output_weight = nn.Parameter(torch.ones(1))
        self.norm = LayerNorm(out_channels)
        self.act = nn.ReLU()

        # Equal logits are deliberately neutral.  Branch availability, rather
        # than an arbitrary learned bias, determines the initial softmax.
        nn.init.zeros_(self.selector[-1].weight)
        nn.init.zeros_(self.selector[-1].bias)
        self.init_parameters()

    def init_parameters(self):
        for module in self.dynamic_convs:
            module.init_parameters()

    def reset_selection_gate(self):
        """Restore the configured raw gate without touching global RNG."""
        with torch.no_grad():
            self.selection_gate.fill_(self.selection_gate_parameter_init)

    def effective_selection_gate(self):
        """Return the scalar coefficient applied by the residual fusion."""
        if self.selection_gate_mode == "tanh":
            return self.selection_gate_bound * torch.tanh(
                self.selection_gate)
        return self.selection_gate

    def diagnostic_report(self):
        """Return JSON-ready gate and latest selector statistics."""
        report = {
            "raw_gate": float(self.selection_gate.detach().cpu()),
            "effective_gate": float(
                self.effective_selection_gate().detach().cpu()),
            "gate_lr_scale": self.selection_gate_lr_scale,
            "selection_weights": None,
            "selection_entropy": None,
        }
        if self.last_selection_weights is None:
            return report
        means = []
        entropies = []
        for weights in self.last_selection_weights:
            value = weights.detach().float().cpu()
            means.append(value.mean(dim=0).tolist())
            safe = value.clamp_min(1e-12)
            entropies.append(float(
                (-(safe * safe.log()).sum(dim=1)).mean()))
        report["selection_weights"] = means
        report["selection_entropy"] = entropies
        return report

    @staticmethod
    def _masked_gap(feature, mask):
        valid = mask.to(feature.dtype)
        denominator = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return (feature * valid).sum(dim=-1, keepdim=True) / denominator

    def compute_selection_weights(self, candidates, target_mask, valid_branches):
        """Return normalized branch weights for diagnostics and testing."""
        if len(candidates) != self._NUM_BRANCHES:
            raise ValueError("expected [finer, current, coarser] candidates")
        if len(valid_branches) != self._NUM_BRANCHES:
            raise ValueError("valid_branches must have length three")
        if not valid_branches[1]:
            raise ValueError("the current-level branch must always be valid")

        descriptor = self._masked_gap(sum(candidates), target_mask)
        batch_size, channels, _ = descriptor.shape
        attention_channels = channels if self.channel_wise else 1
        logits = self.selector(descriptor).reshape(
            batch_size, self._NUM_BRANCHES, attention_channels, 1)
        logits = logits / self.temperature

        availability = torch.as_tensor(
            valid_branches, dtype=torch.bool, device=logits.device)
        availability = availability.reshape(1, self._NUM_BRANCHES, 1, 1)
        logits = logits.masked_fill(~availability, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=1)

    def forward(self, inputs, fpn_masks):
        if len(inputs) != len(fpn_masks):
            raise ValueError("inputs and fpn_masks must have the same length")

        dynamic_scales = []
        for level, feature in enumerate(inputs):
            dynamic_scales.append([
                module(feature, fpn_masks[level])[0]
                for module in self.dynamic_convs
            ])

        outputs = []
        selection_summaries = []
        for level, current in enumerate(inputs):
            target_mask = fpn_masks[level]
            valid = target_mask.to(current.dtype)
            candidates = [torch.zeros_like(current), current * valid,
                          torch.zeros_like(current)]
            valid_branches = [False, True, False]
            legacy_neighbours = []

            if level > 0:
                finer = self.resize(
                    dynamic_scales[level - 1][0], current.shape[-1:])
                legacy_neighbours.append(finer)
                candidates[0] = finer * valid
                valid_branches[0] = True
            if level + 1 < len(inputs):
                coarser = self.resize(
                    dynamic_scales[level + 1][1], current.shape[-1:])
                legacy_neighbours.append(coarser)
                candidates[2] = coarser * valid
                valid_branches[2] = True

            # Match DynamicScale_chk's operation order exactly so gate=0 can
            # be checked with torch.equal against serialized predictions.
            legacy = (
                sum(legacy_neighbours) * self.scale_weight
                + current * self.output_weight)

            weights = self.compute_selection_weights(
                candidates, target_mask, valid_branches)
            selection_summaries.append(
                weights.detach().mean(dim=2).squeeze(-1))
            selected = sum(
                weights[:, branch] * candidate
                for branch, candidate in enumerate(candidates))
            scale_feature = (
                legacy
                + self.effective_selection_gate() * (selected - legacy))

            if self.depth_module is not None:
                scale_feature, _ = self.depth_module(
                    scale_feature, target_mask)
            outputs.append(scale_feature)

        self.last_selection_weights = tuple(selection_summaries)
        return outputs, fpn_masks
