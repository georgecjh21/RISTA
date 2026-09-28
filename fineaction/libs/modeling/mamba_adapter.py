"""Masked residual Mamba adapters for P7 ActivityNet experiments.

The implementation follows the temporal Mamba interface used by OpenTAD and
Video Mamba Suite, while preserving DyFADet's ``(B, C, T)`` features and
``(B, 1, T)`` masks.  The external CUDA implementation stays an optional
runtime dependency; importing the baseline project must not require Mamba.
"""

import importlib

from torch import nn

from .blocks import AffineDropPath


SUPPORTED_MAMBA_BACKENDS = ("dbm", "vim")
SUPPORTED_MAMBA_POSITIONS = (
    "stem_branch", "stem_only", "branch_only", "neck_only")


def build_mamba_mixer(
        backend, feature_dim, d_state, d_conv, expand, use_fast_path):
    """Build the pinned Video Mamba Suite mixer with an actionable error."""
    if backend not in SUPPORTED_MAMBA_BACKENDS:
        raise ValueError(
            "mamba backend must be one of {}".format(
                SUPPORTED_MAMBA_BACKENDS))
    module_name = (
        "mamba_ssm.modules.mamba_new"
        if backend == "dbm" else "mamba_ssm.modules.mamba_simple")
    try:
        module = importlib.import_module(module_name)
        mixer_cls = module.Mamba
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "P7 {} backend is unavailable. Install the pinned Video Mamba "
            "Suite mamba/ and causal-conv1d/ packages before enabling P7. "
            "Expected import: {}.Mamba".format(backend.upper(), module_name)
        ) from error

    kwargs = {
        "d_model": int(feature_dim),
        "d_state": int(d_state),
        "d_conv": int(d_conv),
        "expand": int(expand),
        "use_fast_path": bool(use_fast_path),
    }
    if backend == "vim":
        kwargs["bimamba_type"] = "v2"
    return mixer_cls(**kwargs)


class MaskedMambaResidualBlock(nn.Module):
    """Apply a pre-normalized Mamba residual without changing time length.

    ``residual_weight=0`` bypasses the mixer exactly.  This is the P7 identity
    gate used to prove that merely wiring the adapter into DyFADet does not
    alter A0 predictions.  Active experiments use the OpenTAD-style affine
    drop-path scale initialization of 1e-4.
    """

    def __init__(
            self,
            feature_dim,
            backend="dbm",
            d_state=16,
            d_conv=4,
            expand=1,
            drop_path_rate=0.3,
            layer_scale_init=1e-4,
            residual_weight=1.0,
            use_fast_path=True):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if d_state <= 0 or d_conv <= 0 or expand <= 0:
            raise ValueError(
                "Mamba state, convolution, and expansion must be positive")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if layer_scale_init < 0.0:
            raise ValueError("layer_scale_init must be non-negative")
        if not 0.0 <= residual_weight <= 1.0:
            raise ValueError("residual_weight must be in [0, 1]")

        self.feature_dim = int(feature_dim)
        self.backend = str(backend)
        self.residual_weight = float(residual_weight)
        self.norm = nn.LayerNorm(self.feature_dim, eps=1e-6)
        self.mixer = build_mamba_mixer(
            self.backend, self.feature_dim, d_state, d_conv, expand,
            use_fast_path)
        if drop_path_rate > 0.0 or layer_scale_init != 1.0:
            self.drop_path = AffineDropPath(
                self.feature_dim,
                drop_prob=float(drop_path_rate),
                init_scale_value=float(layer_scale_init))
        else:
            self.drop_path = nn.Identity()

    def forward(self, features, mask):
        if features.ndim != 3:
            raise ValueError("Mamba features must have shape (B, C, T)")
        if mask.ndim != 3 or mask.shape[1] != 1:
            raise ValueError("Mamba mask must have shape (B, 1, T)")
        if features.shape[0] != mask.shape[0] \
                or features.shape[-1] != mask.shape[-1]:
            raise ValueError("Mamba feature/mask shapes are inconsistent")
        if features.shape[1] != self.feature_dim:
            raise ValueError("unexpected Mamba feature dimension")

        valid = mask.to(features.dtype)
        if self.residual_weight == 0.0:
            return features, mask.bool()

        base = features * valid
        sequence = self.norm(base.transpose(1, 2))
        residual = self.mixer(sequence).transpose(1, 2) * valid
        output = base + self.residual_weight * self.drop_path(residual)
        return output * valid, mask.bool()


def make_mamba_block(feature_dim, config):
    """Construct one block from a validated P7 configuration dictionary."""
    return MaskedMambaResidualBlock(
        feature_dim=feature_dim,
        backend=str(config.get("backend", "dbm")),
        d_state=int(config.get("d_state", 16)),
        d_conv=int(config.get("d_conv", 4)),
        expand=int(config.get("expand", 1)),
        drop_path_rate=float(config.get("drop_path_rate", 0.3)),
        layer_scale_init=float(config.get("layer_scale_init", 1e-4)),
        residual_weight=float(config.get("residual_weight", 1.0)),
        use_fast_path=bool(config.get("use_fast_path", True)),
    )


class MambaPyramidAdapter(nn.Module):
    """Independent per-level Mamba residuals after the DyFADet neck."""

    def __init__(self, feature_dim, num_levels, config):
        super().__init__()
        if num_levels <= 0:
            raise ValueError("num_levels must be positive")
        self.blocks = nn.ModuleList([
            make_mamba_block(feature_dim, config)
            for _ in range(int(num_levels))
        ])

    def forward(self, features, masks):
        if len(features) != len(self.blocks) or len(masks) != len(self.blocks):
            raise ValueError("unexpected number of Mamba pyramid levels")
        outputs = []
        output_masks = []
        for block, feature, mask in zip(self.blocks, features, masks):
            output, output_mask = block(feature, mask)
            outputs.append(output)
            output_masks.append(output_mask)
        return tuple(outputs), tuple(output_masks)
