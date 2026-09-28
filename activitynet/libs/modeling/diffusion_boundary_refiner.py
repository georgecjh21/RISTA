"""Offset-driven diffusion refinement for temporal action boundaries.

All boundaries accepted by this module are ``(start, end)`` segment
coordinates on the common feature grid.  The diffused quantity is the local,
dimensionless residual ``(gt - coarse) / coarse_duration``.  This keeps the
noise scale independent of FPN stride and proposal duration.
"""

import copy
import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """Embed integer diffusion timesteps to ``[N, hidden_dim]``."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4")
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        timesteps = timesteps.reshape(-1).float()
        half = self.hidden_dim // 2
        scale = -math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=timesteps.dtype) * scale
        )
        angles = timesteps[:, None] * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.hidden_dim:
            embedding = F.pad(embedding, (0, self.hidden_dim - embedding.shape[-1]))
        return self.proj(embedding)


class BoundaryProposalEmbedding(nn.Module):
    """Embed coarse ``(start, end)`` using stable local geometry."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, coarse_boundaries: Tensor) -> Tensor:
        if coarse_boundaries.ndim != 2 or coarse_boundaries.shape[-1] != 2:
            raise ValueError("coarse_boundaries must have shape [N, 2]")
        start, end = coarse_boundaries.unbind(-1)
        duration = (end - start).clamp_min(1e-6)
        center = 0.5 * (start + end)
        # Absolute feature-grid values are compressed; local duration remains explicit.
        geometry = torch.stack(
            (torch.sign(center) * torch.log1p(center.abs()),
             torch.log(duration), start / duration, end / duration), dim=-1
        )
        return self.net(geometry)


class ConditionalOffsetDenoiser(nn.Module):
    """Transformer decoder predicting start/end residual noise in parallel."""

    def __init__(
        self,
        reg_feature_dim: int,
        hidden_dim: int,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        num_feature_levels: int = 6,
        condition_on_reg_feature: bool = True,
        boundary_context_samples: int = 0,
    ):
        super().__init__()
        if hidden_dim % 8 != 0:
            raise ValueError("diffusion hidden_dim must be divisible by 8")
        self.condition_on_reg_feature = condition_on_reg_feature
        self.residual_embed = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embed = SinusoidalTimestepEmbedding(hidden_dim)
        self.boundary_embed = BoundaryProposalEmbedding(hidden_dim)
        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)
        self.reg_feature_proj = nn.Linear(reg_feature_dim, hidden_dim)
        self.boundary_context_samples = int(boundary_context_samples)
        self.boundary_context_proj = None
        self.boundary_context_position = None
        if self.boundary_context_samples > 0:
            num_context_tokens = 2 * self.boundary_context_samples
            self.boundary_context_proj = nn.Linear(
                reg_feature_dim, hidden_dim)
            self.boundary_context_position = nn.Embedding(
                num_context_tokens, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))

    def forward(
        self,
        noisy_residual: Tensor,
        timesteps: Tensor,
        reg_features: Tensor,
        coarse_boundaries: Tensor,
        level_ids: Optional[Tensor] = None,
        boundary_context: Optional[Tensor] = None,
    ) -> Tensor:
        n = noisy_residual.shape[0]
        if noisy_residual.shape != (n, 2):
            raise ValueError("noisy_residual must have shape [N, 2]")
        if reg_features.ndim != 2 or reg_features.shape[0] != n:
            raise ValueError("reg_features must have shape [N, C]")
        if coarse_boundaries.shape != (n, 2):
            raise ValueError("coarse_boundaries must have shape [N, 2]")
        if timesteps.numel() == 1 and n != 1:
            timesteps = timesteps.expand(n)
        if timesteps.numel() != n:
            raise ValueError("timesteps must contain one value per candidate")
        if level_ids is None:
            level_ids = torch.zeros(n, device=noisy_residual.device, dtype=torch.long)

        query = (self.residual_embed(noisy_residual)
                 + self.time_embed(timesteps)
                 + self.boundary_embed(coarse_boundaries)
                 + self.level_embed(level_ids.long()))
        memory = self.reg_feature_proj(reg_features)
        if not self.condition_on_reg_feature:
            memory = torch.zeros_like(memory)
        memory = memory[:, None]
        if self.boundary_context_samples > 0:
            expected = (n, 2, self.boundary_context_samples,
                        reg_features.shape[-1])
            if boundary_context is None or boundary_context.shape != expected:
                raise ValueError(
                    "boundary_context must have shape {}".format(expected))
            context = boundary_context.reshape(
                n, 2 * self.boundary_context_samples,
                reg_features.shape[-1])
            context = self.boundary_context_proj(context)
            positions = torch.arange(
                context.shape[1], device=context.device, dtype=torch.long)
            context = context + self.boundary_context_position(positions)[None]
            memory = torch.cat((memory, context), dim=1)
        decoded = self.decoder(query[:, None], memory)[:, 0]
        return self.output(decoded)


class BoundaryQualityGate(nn.Module):
    """Predict candidate boundary tIoU from proposal-local evidence."""

    def __init__(
        self,
        reg_feature_dim: int,
        hidden_dim: int,
        num_feature_levels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.residual_embed = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.boundary_embed = BoundaryProposalEmbedding(hidden_dim)
        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)
        self.reg_feature_proj = nn.Linear(reg_feature_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidate_residual: Tensor,
        reg_features: Tensor,
        coarse_boundaries: Tensor,
        level_ids: Tensor,
    ) -> Tensor:
        hidden = (
            self.residual_embed(candidate_residual)
            + self.boundary_embed(coarse_boundaries)
            + self.level_embed(level_ids.long())
            + self.reg_feature_proj(reg_features)
        )
        return self.output(hidden).squeeze(-1)


class CandidateDeltaRanker(nn.Module):
    """Predict the value of K refined alternatives relative to coarse."""

    def __init__(
        self,
        reg_feature_dim: int,
        hidden_dim: int,
        num_feature_levels: int,
        dropout: float = 0.1,
        context_enabled: bool = False,
        quality_threshold_count: int = 0,
        iou_head_enabled: bool = False,
    ):
        super().__init__()
        self.feature_proj = nn.Linear(reg_feature_dim, hidden_dim)
        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)
        self.coarse_embed = BoundaryProposalEmbedding(hidden_dim)
        self.context_enabled = bool(context_enabled)
        self.quality_threshold_count = int(quality_threshold_count)
        self.iou_head_enabled = bool(iou_head_enabled)
        self.context_proj = None
        if self.context_enabled:
            # coarse contrast, candidate contrast, and their difference;
            # each contrast concatenates start/end C-dimensional evidence.
            self.context_proj = nn.Sequential(
                nn.Linear(reg_feature_dim * 6, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        # start/end residual, absolute residual, log duration ratio, score
        self.candidate_embed = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
        )
        # delta-tIoU regression, improvement probability, listwise utility
        self.output = nn.Linear(hidden_dim, 3)
        self.quality_output = (
            nn.Linear(hidden_dim, self.quality_threshold_count)
            if self.quality_threshold_count > 0 else None)
        self.iou_output = (
            nn.Linear(hidden_dim, 1) if self.iou_head_enabled else None)

    def _encode_candidates(
        self,
        coarse: Tensor,
        candidates: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        proposal_scores: Tensor,
        coarse_context: Optional[Tensor] = None,
        candidate_context: Optional[Tensor] = None,
    ) -> Tensor:

        if coarse.ndim != 2 or coarse.shape[-1] != 2:
            raise ValueError("coarse must have shape [N, 2]")
        if candidates.ndim != 3 or candidates.shape[0] != coarse.shape[0] \
                or candidates.shape[-1] != 2:
            raise ValueError("candidates must have shape [N, K, 2]")
        if reg_features.ndim != 2 or reg_features.shape[0] != coarse.shape[0]:
            raise ValueError("reg_features must have shape [N, C]")
        if level_ids.shape != (coarse.shape[0],):
            raise ValueError("level_ids must have shape [N]")
        if proposal_scores.shape != (coarse.shape[0],):
            raise ValueError("proposal_scores must have shape [N]")
        duration = (coarse[:, 1] - coarse[:, 0]).clamp_min(1e-6)
        residual = (candidates - coarse[:, None]) / duration[:, None, None]
        candidate_duration = (
            candidates[:, :, 1] - candidates[:, :, 0]).clamp_min(1e-6)
        log_duration_ratio = torch.log(
            candidate_duration / duration[:, None])
        score = proposal_scores[:, None].expand_as(log_duration_ratio)
        geometry = torch.cat((
            residual, residual.abs(),
            log_duration_ratio[:, :, None], score[:, :, None],
        ), dim=-1)
        shared = (
            self.feature_proj(reg_features)
            + self.level_embed(level_ids.long())
            + self.coarse_embed(coarse)
        )[:, None]
        candidate_hidden = self.candidate_embed(geometry)
        if self.context_enabled:
            expected_coarse = (coarse.shape[0], reg_features.shape[1] * 2)
            expected_candidate = (
                coarse.shape[0], candidates.shape[1],
                reg_features.shape[1] * 2)
            if coarse_context is None or coarse_context.shape != expected_coarse:
                raise ValueError(
                    "coarse_context must have shape {}".format(
                        expected_coarse))
            if (candidate_context is None
                    or candidate_context.shape != expected_candidate):
                raise ValueError(
                    "candidate_context must have shape {}".format(
                        expected_candidate))
            repeated_coarse_context = coarse_context[:, None].expand(
                -1, candidates.shape[1], -1)
            context_features = torch.cat((
                repeated_coarse_context, candidate_context,
                candidate_context - repeated_coarse_context,
            ), dim=-1)
            candidate_hidden = candidate_hidden + self.context_proj(
                context_features)
        return self.fusion(shared + candidate_hidden)

    def forward(
        self,
        coarse: Tensor,
        candidates: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        proposal_scores: Tensor,
        coarse_context: Optional[Tensor] = None,
        candidate_context: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        hidden = self._encode_candidates(
            coarse, candidates, reg_features, level_ids, proposal_scores,
            coarse_context, candidate_context)
        output = self.output(hidden)
        return output[..., 0], output[..., 1], output[..., 2]

    def predict_quality(
        self,
        coarse: Tensor,
        candidates: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        proposal_scores: Tensor,
    ) -> Tensor:
        if self.quality_output is None:
            raise ValueError("candidate quality prediction is not enabled")
        all_candidates = torch.cat((coarse[:, None], candidates), dim=1)
        hidden = self._encode_candidates(
            coarse, all_candidates, reg_features, level_ids, proposal_scores)
        return self.quality_output(hidden)

    def predict_profile_values(
        self,
        coarse: Tensor,
        candidates: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        proposal_scores: Tensor,
        coarse_context: Optional[Tensor] = None,
        candidate_context: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict relative utility evidence for coarse plus K profiles."""
        if self.iou_output is None:
            raise ValueError("candidate IoU prediction is not enabled")
        all_candidates = torch.cat((coarse[:, None], candidates), dim=1)
        all_context = None
        if self.context_enabled:
            if coarse_context is None or candidate_context is None:
                raise ValueError("profile values require boundary context")
            all_context = torch.cat(
                (coarse_context[:, None], candidate_context), dim=1)
        hidden = self._encode_candidates(
            coarse, all_candidates, reg_features, level_ids, proposal_scores,
            coarse_context, all_context)
        output = self.output(hidden)
        predicted_iou = torch.sigmoid(self.iou_output(hidden).squeeze(-1))
        return output[..., 0], output[..., 1], output[..., 2], predicted_iou


class OracleDistillationUtilityRanker(nn.Module):
    """Lightweight profile-aware predictor with one deployment utility score."""

    def __init__(
        self,
        reg_feature_dim: int,
        hidden_dim: int,
        num_feature_levels: int,
        dropout: float = 0.1,
        num_classes: int = 256,
    ):
        super().__init__()
        context_dim = max(8, hidden_dim // 4)
        shared_dim = max(16, hidden_dim // 2)
        metadata_dim = max(8, hidden_dim // 4)
        self.feature_proj = nn.Sequential(
            nn.Linear(reg_feature_dim, shared_dim), nn.SiLU())
        self.level_embed = nn.Embedding(num_feature_levels, metadata_dim)
        self.class_embed = nn.Embedding(num_classes, metadata_dim)
        self.context_proj = nn.Sequential(
            nn.Linear(reg_feature_dim * 2, context_dim), nn.SiLU())
        self.profile_proj = nn.Sequential(
            nn.Linear(4, metadata_dim), nn.SiLU())
        # relative start/end, absolute shifts, duration ratio, score,
        # log coarse duration, center shift, and FPN level.
        self.geometry_proj = nn.Sequential(
            nn.Linear(9, shared_dim), nn.SiLU())
        fusion_dim = (
            shared_dim * 2 + context_dim * 3 + metadata_dim * 3)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, hidden_dim),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.utility_output = nn.Linear(hidden_dim, 1)
        self.iou_output = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        coarse: Tensor,
        candidates: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        proposal_scores: Tensor,
        class_ids: Tensor,
        coarse_context: Tensor,
        candidate_context: Tensor,
        profile_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if candidates.ndim != 3 or candidates.shape[-1] != 2:
            raise ValueError("candidates must have shape [N, K, 2]")
        n, k = candidates.shape[:2]
        if coarse.shape != (n, 2):
            raise ValueError("coarse must have shape [N, 2]")
        if profile_features.shape == (k, 4):
            profile_features = profile_features[None].expand(n, -1, -1)
        if profile_features.shape != (n, k, 4):
            raise ValueError("profile_features must have shape [K, 4] or [N, K, 4]")
        expected_context = (n, reg_features.shape[-1] * 2)
        if coarse_context.shape != expected_context:
            raise ValueError("coarse_context has an invalid shape")
        if candidate_context.shape != (n, k, expected_context[-1]):
            raise ValueError("candidate_context has an invalid shape")
        duration = (coarse[:, 1] - coarse[:, 0]).clamp_min(1e-6)
        residual = (candidates - coarse[:, None]) / duration[:, None, None]
        candidate_duration = (
            candidates[..., 1] - candidates[..., 0]).clamp_min(1e-6)
        duration_ratio = torch.log(candidate_duration / duration[:, None])
        center_shift = (
            candidates.mean(dim=-1) - coarse.mean(dim=-1)[:, None]
        ) / duration[:, None]
        score = proposal_scores[:, None].expand(n, k)
        log_duration = torch.log(duration)[:, None].expand(n, k)
        level_value = level_ids.float()[:, None].expand(n, k) / max(
            self.level_embed.num_embeddings - 1, 1)
        geometry = torch.cat((
            residual, residual.abs(), duration_ratio[..., None],
            score[..., None], log_duration[..., None],
            center_shift[..., None], level_value[..., None]), dim=-1)
        coarse_encoded = self.context_proj(coarse_context)[:, None].expand(
            -1, k, -1)
        candidate_encoded = self.context_proj(candidate_context)
        shared = self.feature_proj(reg_features)[:, None].expand(-1, k, -1)
        level = self.level_embed(level_ids.long())[:, None].expand(-1, k, -1)
        classes = self.class_embed(class_ids.long().clamp(
            min=0, max=self.class_embed.num_embeddings - 1)
        )[:, None].expand(-1, k, -1)
        hidden = self.fusion(torch.cat((
            shared, self.geometry_proj(geometry), coarse_encoded,
            candidate_encoded, candidate_encoded - coarse_encoded,
            self.profile_proj(profile_features), level, classes), dim=-1))
        utility = self.utility_output(hidden).squeeze(-1)
        predicted_iou = torch.sigmoid(self.iou_output(hidden).squeeze(-1))
        return utility, predicted_iou


class CosineNoiseScheduler(nn.Module):
    """Nichol-Dhariwal cosine beta schedule with device-safe buffers."""

    def __init__(self, num_train_timesteps: int = 1000, schedule: str = "cosine"):
        super().__init__()
        if num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be >= 2")
        if schedule != "cosine":
            raise ValueError("Only the requested cosine beta schedule is supported")
        steps = torch.arange(num_train_timesteps + 1, dtype=torch.float64)
        f = torch.cos(((steps / num_train_timesteps + 0.008) / 1.008) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-5, 0.999)
        alphas_cumprod = torch.cumprod(1 - betas, dim=0).float()
        self.num_train_timesteps = num_train_timesteps
        self.register_buffer("betas", betas.float(), persistent=True)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=True)

    def add_noise(self, clean: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        if clean.shape != noise.shape:
            raise ValueError("clean and noise must have identical shapes")
        alpha = self.alphas_cumprod[timesteps.long()].to(clean.dtype)[:, None]
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise


class DDIMBoundarySampler:
    """Deterministic (eta=0) DDIM reverse process for boundary states."""

    def __init__(
        self,
        scheduler: CosineNoiseScheduler,
        num_inference_steps: int,
        prediction_type: str = "epsilon",
    ):
        if not 1 <= num_inference_steps <= scheduler.num_train_timesteps:
            raise ValueError("num_inference_steps must be within the training schedule")
        if prediction_type not in ("epsilon", "x0"):
            raise ValueError("prediction_type must be 'epsilon' or 'x0'")
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.prediction_type = prediction_type

    @torch.no_grad()
    def sample(
        self,
        denoiser: ConditionalOffsetDenoiser,
        initial_noise: Tensor,
        reg_features: Tensor,
        coarse_boundaries: Tensor,
        level_ids: Tensor,
        boundary_context: Optional[Tensor] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Tensor:
        sample = initial_noise
        steps = (self.num_inference_steps if num_inference_steps is None
                 else int(num_inference_steps))
        if not 1 <= steps <= self.scheduler.num_train_timesteps:
            raise ValueError(
                "num_inference_steps must be within the training schedule")
        timesteps = torch.linspace(
            self.scheduler.num_train_timesteps - 1, 0,
            steps, device=sample.device
        ).round().long()
        for index, timestep in enumerate(timesteps):
            t_batch = timestep.expand(sample.shape[0])
            prediction = denoiser(
                sample, t_batch, reg_features, coarse_boundaries, level_ids,
                boundary_context)
            alpha_t = self.scheduler.alphas_cumprod[timestep].to(sample.dtype)
            if self.prediction_type == "epsilon":
                eps = prediction
                pred_clean = (
                    sample - (1 - alpha_t).sqrt() * eps
                ) / alpha_t.sqrt().clamp_min(1e-6)
                pred_clean = pred_clean.clamp(-10.0, 10.0)
            else:
                pred_clean = prediction.clamp(-10.0, 10.0)
                eps = (
                    sample - alpha_t.sqrt() * pred_clean
                ) / (1 - alpha_t).sqrt().clamp_min(1e-6)
            if index == len(timesteps) - 1:
                sample = pred_clean
            else:
                alpha_prev = self.scheduler.alphas_cumprod[timesteps[index + 1]].to(sample.dtype)
                sample = alpha_prev.sqrt() * pred_clean + (1 - alpha_prev).sqrt() * eps
        return sample


class DiffusionBoundaryRefiner(nn.Module):
    """Training and inference facade for normalized local boundary states."""

    def __init__(self, reg_feature_dim: int, config: Dict, num_feature_levels: int):
        super().__init__()
        hidden_dim = config.get("hidden_dim") or reg_feature_dim
        self.detach_coarse_boundary = config.get("detach_coarse_boundary", True)
        self.min_duration = float(config.get("min_duration", 0.01))
        self.clamp_boundary = bool(config.get("clamp_boundary", True))
        self.inference_seed = int(config.get("inference_seed", 0))
        self.residual_blend_alpha = float(
            config.get("residual_blend_alpha", 1.0))
        if not 0.0 <= self.residual_blend_alpha <= 1.0:
            raise ValueError("residual_blend_alpha must be within [0, 1]")
        max_relative_shift = config.get("max_relative_shift")
        self.max_relative_shift = (
            None if max_relative_shift is None else float(max_relative_shift))
        if (self.max_relative_shift is not None
                and not 0.0 < self.max_relative_shift <= 1.0):
            raise ValueError("max_relative_shift must be within (0, 1]")
        if not config.get("use_ddim", True):
            raise ValueError("This first version implements DDIM inference only")
        self.scheduler = CosineNoiseScheduler(
            int(config.get("num_train_timesteps", 1000)),
            config.get("beta_schedule", "cosine"),
        )
        self.prediction_type = str(config.get("prediction_type", "epsilon"))
        if self.prediction_type not in ("epsilon", "x0"):
            raise ValueError("prediction_type must be 'epsilon' or 'x0'")
        self.state_space = str(config.get("state_space", "endpoint"))
        if self.state_space not in ("endpoint", "center_log_duration"):
            raise ValueError(
                "state_space must be 'endpoint' or 'center_log_duration'")
        self.denoiser = ConditionalOffsetDenoiser(
            reg_feature_dim, hidden_dim,
            int(config.get("num_decoder_layers", 2)),
            float(config.get("dropout", 0.1)),
            num_feature_levels,
            bool(config.get("condition_on_reg_feature", True)),
            len(config.get("boundary_context_offsets", []))
            if config.get("boundary_context_enabled", False) else 0,
        )
        self.boundary_context_enabled = bool(
            config.get("boundary_context_enabled", False))
        self.boundary_context_offsets = tuple(float(value) for value in
                                              config.get(
                                                  "boundary_context_offsets",
                                                  (-2, -1, 0, 1, 2)))
        if self.boundary_context_enabled and not self.boundary_context_offsets:
            raise ValueError(
                "boundary_context_offsets must not be empty when enabled")
        self.sampler = DDIMBoundarySampler(
            self.scheduler, int(config.get("num_inference_steps", 10)),
            self.prediction_type,
        )
        self.boundary_loss_enabled = bool(
            config.get("boundary_loss_enabled", False))
        self.boundary_loss_weight = float(
            config.get("boundary_loss_weight", 0.0))
        self.boundary_loss_snr_cap = float(
            config.get("boundary_loss_snr_cap", 5.0))
        self.boundary_loss_max_residual = float(
            config.get("boundary_loss_max_residual", 2.0))
        if self.boundary_loss_weight < 0.0:
            raise ValueError("boundary_loss_weight must be >= 0")
        if self.boundary_loss_enabled and self.boundary_loss_weight <= 0.0:
            raise ValueError(
                "boundary_loss_enabled requires boundary_loss_weight > 0")
        if self.boundary_loss_snr_cap <= 0.0:
            raise ValueError("boundary_loss_snr_cap must be > 0")
        if self.boundary_loss_max_residual <= 0.0:
            raise ValueError("boundary_loss_max_residual must be > 0")
        self.v15b_enabled = bool(config.get("v15b_enabled", False))
        self.v15b_min_snr_gamma = float(
            config.get("v15b_min_snr_gamma", 5.0))
        self.v15b_tiou_loss_weight = float(
            config.get("v15b_tiou_loss_weight", 0.0))
        self.v15b_endpoint_loss_weight = float(
            config.get("v15b_endpoint_loss_weight", 0.0))
        self.v15b_thresholds = tuple(
            float(value) for value in config.get(
                "v15b_thresholds", (0.6, 0.7)))
        self.v15b_threshold_sigma = float(
            config.get("v15b_threshold_sigma", 0.05))
        self.v15b_threshold_gain = float(
            config.get("v15b_threshold_gain", 1.0))
        self.v15b_endpoint_beta = float(
            config.get("v15b_endpoint_beta", 0.1))
        if self.v15b_min_snr_gamma <= 0.0:
            raise ValueError("v15b_min_snr_gamma must be > 0")
        if self.v15b_tiou_loss_weight < 0.0:
            raise ValueError("v15b_tiou_loss_weight must be >= 0")
        if self.v15b_endpoint_loss_weight < 0.0:
            raise ValueError("v15b_endpoint_loss_weight must be >= 0")
        if self.v15b_threshold_sigma <= 0.0:
            raise ValueError("v15b_threshold_sigma must be > 0")
        if self.v15b_threshold_gain < 0.0:
            raise ValueError("v15b_threshold_gain must be >= 0")
        if self.v15b_endpoint_beta <= 0.0:
            raise ValueError("v15b_endpoint_beta must be > 0")
        if self.v15b_enabled:
            if self.state_space != "center_log_duration":
                raise ValueError(
                    "V15-B requires center_log_duration state space")
            if self.prediction_type != "x0":
                raise ValueError("V15-B requires direct x0 prediction")
            if self.v15b_tiou_loss_weight <= 0.0:
                raise ValueError(
                    "V15-B requires v15b_tiou_loss_weight > 0")
            if self.v15b_endpoint_loss_weight <= 0.0:
                raise ValueError(
                    "V15-B requires v15b_endpoint_loss_weight > 0")
            if not self.v15b_thresholds:
                raise ValueError("V15-B thresholds must not be empty")
            if any(not 0.0 < value < 1.0
                   for value in self.v15b_thresholds):
                raise ValueError(
                    "V15-B thresholds must be within (0, 1)")
        self.v15c1_enabled = bool(config.get("v15c1_enabled", False))
        self.v15c1_identity_start_tiou = float(
            config.get("v15c1_identity_start_tiou", 0.7))
        self.v15c1_identity_full_tiou = float(
            config.get("v15c1_identity_full_tiou", 0.8))
        self.v15c1_identity_max_weight = float(
            config.get("v15c1_identity_max_weight", 2.0))
        self.v15c1_identity_loss_weight = float(
            config.get("v15c1_identity_loss_weight", 0.0))
        self.v15c1_duration_asym_loss_weight = float(
            config.get("v15c1_duration_asym_loss_weight", 0.0))
        self.v15c1_duration_beta = float(
            config.get("v15c1_duration_beta", 0.05))
        if self.v15c1_enabled:
            if not self.v15b_enabled:
                raise ValueError("V15-C1 requires V15-B geometry training")
            if not (
                    0.0 <= self.v15c1_identity_start_tiou
                    < self.v15c1_identity_full_tiou < 1.0):
                raise ValueError(
                    "V15-C1 identity thresholds must satisfy "
                    "0 <= start < full < 1")
            if self.v15c1_identity_max_weight < 1.0:
                raise ValueError(
                    "v15c1_identity_max_weight must be >= 1")
            if self.v15c1_identity_loss_weight <= 0.0:
                raise ValueError(
                    "V15-C1 requires v15c1_identity_loss_weight > 0")
            if self.v15c1_duration_asym_loss_weight <= 0.0:
                raise ValueError(
                    "V15-C1 requires "
                    "v15c1_duration_asym_loss_weight > 0")
            if self.v15c1_duration_beta <= 0.0:
                raise ValueError("v15c1_duration_beta must be > 0")
        self.final_ddim_loss_enabled = bool(
            config.get("final_ddim_loss_enabled", False))
        self.final_ddim_loss_weight = float(
            config.get("final_ddim_loss_weight", 0.0))
        self.final_ddim_train_steps = int(
            config.get("final_ddim_train_steps", 3))
        self.final_ddim_max_samples = int(
            config.get("final_ddim_max_samples", 128))
        self.final_ddim_max_residual = float(
            config.get("final_ddim_max_residual", 2.0))
        if self.final_ddim_loss_weight < 0.0:
            raise ValueError("final_ddim_loss_weight must be >= 0")
        if (self.final_ddim_loss_enabled
                and self.final_ddim_loss_weight <= 0.0):
            raise ValueError(
                "final_ddim_loss_enabled requires final_ddim_loss_weight > 0")
        if not 1 <= self.final_ddim_train_steps <= \
                self.scheduler.num_train_timesteps:
            raise ValueError("final_ddim_train_steps is outside the schedule")
        if self.final_ddim_max_samples <= 0:
            raise ValueError("final_ddim_max_samples must be > 0")
        if self.final_ddim_max_residual <= 0.0:
            raise ValueError("final_ddim_max_residual must be > 0")
        self.candidate_ranker_enabled = bool(
            config.get("candidate_ranker_enabled", False))
        self.candidate_ranker_num_candidates = int(
            config.get("candidate_ranker_num_candidates", 8))
        self.candidate_ranker_delta_margin = float(
            config.get("candidate_ranker_delta_margin", 0.0))
        self.candidate_ranker_improve_threshold = float(
            config.get("candidate_ranker_improve_threshold", 0.5))
        self.candidate_ranker_regression_weight = float(
            config.get("candidate_ranker_regression_weight", 1.0))
        self.candidate_ranker_bce_weight = float(
            config.get("candidate_ranker_bce_weight", 1.0))
        self.candidate_ranker_listwise_weight = float(
            config.get("candidate_ranker_listwise_weight", 1.0))
        self.candidate_ranker_delta_scale = float(
            config.get("candidate_ranker_delta_scale", 1.0))
        self.candidate_ranker_soft_listwise = bool(
            config.get("candidate_ranker_soft_listwise", False))
        self.candidate_ranker_softmax_temperature = float(
            config.get("candidate_ranker_softmax_temperature", 0.01))
        self.candidate_ranker_context_enabled = bool(
            config.get("candidate_ranker_context_enabled", False))
        self.candidate_ranker_context_offsets = tuple(
            float(value) for value in config.get(
                "candidate_ranker_context_offsets", [-2, -1, 0, 1, 2]))
        self.candidate_ranker_target_type = config.get(
            "candidate_ranker_target_type", "delta_tiou")
        self.candidate_ranker_tiou_thresholds = tuple(float(value) for value in
            config.get("candidate_ranker_tiou_thresholds",
                       [0.3, 0.4, 0.5, 0.6, 0.7]))
        self.candidate_ranker_threshold_weights = tuple(float(value) for value in
            config.get("candidate_ranker_threshold_weights",
                       [1.0, 1.0, 1.0, 1.0, 1.0]))
        self.candidate_ranker_score_weight_power = float(
            config.get("candidate_ranker_score_weight_power", 0.0))
        self.candidate_ranker_negative_utility_weight = float(
            config.get("candidate_ranker_negative_utility_weight", 1.0))
        self.candidate_ranker_proximity_sigma = float(
            config.get("candidate_ranker_proximity_sigma", 0.05))
        self.candidate_ranker_proximity_gain = float(
            config.get("candidate_ranker_proximity_gain", 2.0))
        self.candidate_ranker_crossing_bonus = float(
            config.get("candidate_ranker_crossing_bonus", 0.05))
        self.candidate_ranker_harmful_crossing_penalty = float(
            config.get("candidate_ranker_harmful_crossing_penalty", 0.1))
        self.candidate_ranker_quality_enabled = bool(
            config.get("candidate_ranker_quality_enabled", False))
        self.candidate_ranker_quality_focal_gamma = float(
            config.get("candidate_ranker_quality_focal_gamma", 2.0))
        self.candidate_ranker_quality_bce_weight = float(
            config.get("candidate_ranker_quality_bce_weight", 1.0))
        self.candidate_ranker_quality_pairwise_weight = float(
            config.get("candidate_ranker_quality_pairwise_weight", 1.0))
        self.candidate_ranker_quality_listwise_weight = float(
            config.get("candidate_ranker_quality_listwise_weight", 1.0))
        self.candidate_ranker_quality_pairwise_temperature = float(config.get(
            "candidate_ranker_quality_pairwise_temperature", 0.1))
        self.candidate_ranker_quality_margin = float(
            config.get("candidate_ranker_quality_margin", 0.02))
        self.candidate_ranker_quality_harmful_weight = float(
            config.get("candidate_ranker_quality_harmful_weight", 2.0))
        self.candidate_ranker_quality_rescore_enabled = bool(config.get(
            "candidate_ranker_quality_rescore_enabled", False))
        self.candidate_ranker_quality_rescore_gamma = float(config.get(
            "candidate_ranker_quality_rescore_gamma", 0.5))
        self.candidate_ranker_quality_rescore_max_change = float(config.get(
            "candidate_ranker_quality_rescore_max_change", 0.1))
        self.candidate_ranker_iou_enabled = bool(
            config.get("candidate_ranker_iou_enabled", False))
        self.candidate_ranker_iou_weight = float(
            config.get("candidate_ranker_iou_weight", 1.0))
        self.candidate_ranker_rescore_enabled = bool(
            config.get("candidate_ranker_rescore_enabled", False))
        self.candidate_ranker_rescore_gamma = float(
            config.get("candidate_ranker_rescore_gamma", 0.5))
        self.candidate_ranker_rescore_max_change = float(
            config.get("candidate_ranker_rescore_max_change", 0.1))
        self.candidate_ranker_train_topk = int(
            config.get("candidate_ranker_train_topk", 0))
        self.candidate_ranker_num_classes = int(
            config.get("candidate_ranker_num_classes", 256))
        self.candidate_ranker_nms_penalty = float(
            config.get("candidate_ranker_nms_penalty", 0.1))
        self.candidate_ranker_decision_loss_enabled = bool(
            config.get("candidate_ranker_decision_loss_enabled", False))
        self.candidate_ranker_regret_weight = float(
            config.get("candidate_ranker_regret_weight", 1.0))
        self.candidate_ranker_ranking_temperature = float(
            config.get("candidate_ranker_ranking_temperature", 1.0))
        self.candidate_ranker_score_normalization = bool(
            config.get("candidate_ranker_score_normalization", False))
        self.candidate_ranker_recompute_synthetic_context = bool(config.get(
            "candidate_ranker_recompute_synthetic_context", False))
        self.candidate_ranker_synthetic_jitter_std = float(config.get(
            "candidate_ranker_synthetic_jitter_std", 0.0))
        self.candidate_ranker_ensemble_enabled = bool(config.get(
            "candidate_ranker_ensemble_enabled", False))
        self.candidate_ranker_ensemble_size = int(config.get(
            "candidate_ranker_ensemble_size", 5))
        self.candidate_ranker_ensemble_min_agreement = int(config.get(
            "candidate_ranker_ensemble_min_agreement", 3))
        self.candidate_ranker_low_tiou_safety_enabled = bool(config.get(
            "candidate_ranker_low_tiou_safety_enabled", False))
        self.candidate_ranker_low_tiou_safety_thresholds = tuple(
            float(value) for value in config.get(
                "candidate_ranker_low_tiou_safety_thresholds", [0.3, 0.4]))
        self.register_buffer(
            "candidate_ranker_ensemble_score_means", torch.empty(0),
            persistent=False)
        self.register_buffer(
            "candidate_ranker_ensemble_score_stds", torch.empty(0),
            persistent=False)
        self.register_buffer(
            "candidate_ranker_ensemble_zero_thresholds", torch.empty(0),
            persistent=False)
        self.register_buffer(
            "candidate_ranker_ensemble_threshold", torch.tensor(0.0),
            persistent=False)
        self.register_buffer(
            "candidate_ranker_ensemble_max_std", torch.tensor(float("inf")),
            persistent=False)
        self.register_buffer(
            "candidate_ranker_ensemble_max_low_tiou_harm_rate",
            torch.tensor(1.0), persistent=False)
        profile_seeds = tuple(int(value) for value in config.get(
            "candidate_profile_seeds", ()))
        profile_steps = tuple(int(value) for value in config.get(
            "candidate_profile_steps", ()))
        profile_alphas = tuple(float(value) for value in config.get(
            "candidate_profile_alphas", ()))
        self.candidate_ranker_profile_pool = tuple(
            {"seed": seed, "steps": steps, "alpha": alpha,
             "max_relative_shift": config.get(
                 "candidate_profile_max_relative_shift")}
            for seed in profile_seeds
            for steps in profile_steps
            for alpha in profile_alphas)
        selected_count = int(config.get(
            "candidate_profile_selected_count",
            self.candidate_ranker_num_candidates))
        if self.candidate_ranker_profile_pool:
            initial_profiles = self.candidate_ranker_profile_pool[:selected_count]
            if len(initial_profiles) != selected_count:
                raise ValueError(
                    "candidate profile pool is smaller than selected count")
            self.register_buffer(
                "selected_profile_seeds",
                torch.tensor([item["seed"] for item in initial_profiles],
                             dtype=torch.long), persistent=True)
            self.register_buffer(
                "selected_profile_steps",
                torch.tensor([item["steps"] for item in initial_profiles],
                             dtype=torch.long), persistent=True)
            self.register_buffer(
                "selected_profile_alphas",
                torch.tensor([item["alpha"] for item in initial_profiles],
                             dtype=torch.float32), persistent=True)
        else:
            # Keep legacy checkpoints byte-for-byte compatible: non-V8
            # refiners do not add empty persistent state entries.
            self.selected_profile_seeds = torch.empty(0, dtype=torch.long)
            self.selected_profile_steps = torch.empty(0, dtype=torch.long)
            self.selected_profile_alphas = torch.empty(0)
        if self.candidate_ranker_target_type == "oracle_distilled_utility":
            self.register_buffer(
                "candidate_ranker_calibrated_threshold",
                torch.tensor(float(config.get(
                    "candidate_ranker_calibrated_threshold", 0.0))),
                persistent=True)
            if self.candidate_ranker_score_normalization:
                self.register_buffer(
                    "candidate_ranker_score_mean", torch.tensor(float(
                        config.get("candidate_ranker_score_mean", 0.0))),
                    persistent=True)
                self.register_buffer(
                    "candidate_ranker_score_std", torch.tensor(float(
                        config.get("candidate_ranker_score_std", 1.0))),
                    persistent=True)
            else:
                self.candidate_ranker_score_mean = torch.tensor(0.0)
                self.candidate_ranker_score_std = torch.tensor(1.0)
        else:
            self.candidate_ranker_calibrated_threshold = torch.tensor(0.0)
            self.candidate_ranker_score_mean = torch.tensor(0.0)
            self.candidate_ranker_score_std = torch.tensor(1.0)
        if self.candidate_ranker_num_candidates <= 0:
            raise ValueError("candidate_ranker_num_candidates must be > 0")
        if not 0.0 <= self.candidate_ranker_improve_threshold <= 1.0:
            raise ValueError(
                "candidate_ranker_improve_threshold must be within [0, 1]")
        if min(self.candidate_ranker_regression_weight,
               self.candidate_ranker_bce_weight,
               self.candidate_ranker_listwise_weight) < 0.0:
            raise ValueError("candidate ranker loss weights must be >= 0")
        if self.candidate_ranker_delta_scale <= 0.0:
            raise ValueError("candidate_ranker_delta_scale must be > 0")
        if self.candidate_ranker_softmax_temperature <= 0.0:
            raise ValueError(
                "candidate_ranker_softmax_temperature must be > 0")
        if self.candidate_ranker_context_enabled:
            offsets = self.candidate_ranker_context_offsets
            if len(offsets) < 3 or len(offsets) % 2 == 0:
                raise ValueError(
                    "candidate_ranker_context_offsets must have odd length >= 3")
            midpoint = len(offsets) // 2
            if not (all(offset < 0 for offset in offsets[:midpoint])
                    and offsets[midpoint] == 0
                    and all(offset > 0 for offset in offsets[midpoint + 1:])):
                raise ValueError(
                    "candidate_ranker_context_offsets must be negative, 0, positive")
        if self.candidate_ranker_target_type not in (
                "delta_tiou", "threshold_crossing", "dense_detection",
                "multi_threshold_quality", "profiled_relative_iou",
                "oracle_distilled_utility"):
            raise ValueError("unsupported candidate_ranker_target_type")
        if (not self.candidate_ranker_tiou_thresholds
                or len(self.candidate_ranker_tiou_thresholds)
                != len(self.candidate_ranker_threshold_weights)
                or any(value <= 0.0 or value > 1.0
                       for value in self.candidate_ranker_tiou_thresholds)
                or any(value <= 0.0
                       for value in self.candidate_ranker_threshold_weights)):
            raise ValueError(
                "candidate ranker tIoU thresholds and weights are invalid")
        if self.candidate_ranker_score_weight_power < 0.0:
            raise ValueError("candidate_ranker_score_weight_power must be >= 0")
        if self.candidate_ranker_negative_utility_weight < 1.0:
            raise ValueError(
                "candidate_ranker_negative_utility_weight must be >= 1")
        if self.candidate_ranker_proximity_sigma <= 0.0:
            raise ValueError("candidate_ranker_proximity_sigma must be > 0")
        if min(self.candidate_ranker_proximity_gain,
               self.candidate_ranker_crossing_bonus,
               self.candidate_ranker_harmful_crossing_penalty) < 0.0:
            raise ValueError("candidate ranker dense utility weights must be >= 0")
        if self.candidate_ranker_quality_enabled \
                and self.candidate_ranker_context_enabled:
            raise ValueError(
                "candidate quality ranker currently requires non-context mode")
        if (self.candidate_ranker_quality_enabled
                and not self.candidate_ranker_enabled):
            raise ValueError(
                "candidate quality prediction requires candidate ranker")
        if (self.candidate_ranker_quality_enabled
                and self.candidate_ranker_target_type
                != "multi_threshold_quality"):
            raise ValueError(
                "candidate quality prediction requires multi-threshold targets")
        if self.candidate_ranker_quality_focal_gamma < 0.0:
            raise ValueError("candidate quality focal gamma must be >= 0")
        if min(self.candidate_ranker_quality_bce_weight,
               self.candidate_ranker_quality_pairwise_weight,
               self.candidate_ranker_quality_listwise_weight) < 0.0:
            raise ValueError("candidate quality loss weights must be >= 0")
        if self.candidate_ranker_quality_pairwise_temperature <= 0.0:
            raise ValueError("candidate quality pairwise temperature must be > 0")
        if self.candidate_ranker_quality_margin < 0.0:
            raise ValueError("candidate quality margin must be >= 0")
        if self.candidate_ranker_quality_harmful_weight < 1.0:
            raise ValueError("candidate quality harmful weight must be >= 1")
        if self.candidate_ranker_quality_rescore_gamma < 0.0:
            raise ValueError("candidate quality rescore gamma must be >= 0")
        if not 0.0 <= self.candidate_ranker_quality_rescore_max_change <= 1.0:
            raise ValueError(
                "candidate quality rescore max change must be within [0, 1]")
        if self.candidate_ranker_iou_weight < 0.0:
            raise ValueError("candidate ranker IoU weight must be >= 0")
        if self.candidate_ranker_rescore_gamma < 0.0:
            raise ValueError("candidate ranker rescore gamma must be >= 0")
        if not 0.0 <= self.candidate_ranker_rescore_max_change <= 1.0:
            raise ValueError(
                "candidate ranker rescore max change must be within [0, 1]")
        if self.candidate_ranker_target_type == "profiled_relative_iou":
            if not self.candidate_ranker_context_enabled:
                raise ValueError("profiled relative IoU ranker requires context")
            if not self.candidate_ranker_iou_enabled:
                raise ValueError("profiled relative IoU ranker requires IoU head")
            if not self.candidate_ranker_profile_pool:
                raise ValueError("profiled relative IoU ranker requires profiles")
            if selected_count != self.candidate_ranker_num_candidates:
                raise ValueError(
                    "selected profile count must match ranker candidate count")
        if self.candidate_ranker_target_type == "oracle_distilled_utility":
            if not self.candidate_ranker_context_enabled:
                raise ValueError("oracle distillation ranker requires context")
            if not self.candidate_ranker_iou_enabled:
                raise ValueError("oracle distillation ranker requires IoU supervision")
            if not self.candidate_ranker_profile_pool:
                raise ValueError("oracle distillation ranker requires profiles")
            if selected_count != self.candidate_ranker_num_candidates:
                raise ValueError(
                    "selected profile count must match ranker candidate count")
            if self.candidate_ranker_train_topk < self.candidate_ranker_num_candidates:
                raise ValueError(
                    "oracle distillation train_topk must be at least candidate count")
            if self.candidate_ranker_num_classes <= 0:
                raise ValueError("candidate_ranker_num_classes must be > 0")
            if self.candidate_ranker_nms_penalty < 0.0:
                raise ValueError("candidate_ranker_nms_penalty must be >= 0")
            if self.candidate_ranker_regret_weight < 0.0:
                raise ValueError("candidate_ranker_regret_weight must be >= 0")
            if self.candidate_ranker_ranking_temperature <= 0.0:
                raise ValueError(
                    "candidate_ranker_ranking_temperature must be > 0")
            if self.candidate_ranker_synthetic_jitter_std < 0.0:
                raise ValueError(
                    "candidate_ranker_synthetic_jitter_std must be >= 0")
            if (self.candidate_ranker_recompute_synthetic_context
                    and self.candidate_ranker_synthetic_jitter_std <= 0.0):
                raise ValueError(
                    "recomputed synthetic context requires positive jitter")
        if self.candidate_ranker_ensemble_enabled:
            if self.candidate_ranker_target_type != \
                    "oracle_distilled_utility":
                raise ValueError(
                    "candidate ranker ensemble requires oracle distillation")
            if self.candidate_ranker_ensemble_size < 2:
                raise ValueError(
                    "candidate_ranker_ensemble_size must be at least 2")
            if not 1 <= self.candidate_ranker_ensemble_min_agreement \
                    <= self.candidate_ranker_ensemble_size:
                raise ValueError(
                    "candidate ranker ensemble agreement is out of range")
        if self.candidate_ranker_low_tiou_safety_enabled:
            if not self.candidate_ranker_ensemble_enabled:
                raise ValueError(
                    "low-tIoU safety requires the candidate-ranker ensemble")
            if (not self.candidate_ranker_low_tiou_safety_thresholds
                    or any(not 0.0 < value < 1.0
                           for value in
                           self.candidate_ranker_low_tiou_safety_thresholds)):
                raise ValueError(
                    "low-tIoU safety thresholds must be within (0, 1)")
        self.candidate_ranker = None
        self.candidate_ranker_ensemble = nn.ModuleList()
        if self.candidate_ranker_enabled:
            ranker_hidden_dim = int(
                config.get("candidate_ranker_hidden_dim", 128))
            if ranker_hidden_dim <= 0:
                raise ValueError("candidate_ranker_hidden_dim must be > 0")
            if self.candidate_ranker_target_type == "oracle_distilled_utility":
                self.candidate_ranker = OracleDistillationUtilityRanker(
                    reg_feature_dim, ranker_hidden_dim, num_feature_levels,
                    float(config.get("candidate_ranker_dropout", 0.1)),
                    self.candidate_ranker_num_classes)
            else:
                self.candidate_ranker = CandidateDeltaRanker(
                    reg_feature_dim, ranker_hidden_dim, num_feature_levels,
                    float(config.get("candidate_ranker_dropout", 0.1)),
                    context_enabled=self.candidate_ranker_context_enabled,
                    quality_threshold_count=(
                        len(self.candidate_ranker_tiou_thresholds)
                        if self.candidate_ranker_quality_enabled else 0),
                    iou_head_enabled=self.candidate_ranker_iou_enabled,
                )
            if bool(config.get("candidate_ranker_zero_init_delta", False)):
                nn.init.zeros_(self.candidate_ranker.output.weight[0])
                nn.init.zeros_(self.candidate_ranker.output.bias[0])
        self.quality_gate_enabled = bool(
            config.get("quality_gate_enabled", False))
        self.quality_gate_threshold = float(
            config.get("quality_gate_threshold", 0.5))
        self.quality_gate_margin = float(
            config.get("quality_gate_margin", 0.0))
        self.quality_gate_loss_weight = float(
            config.get("quality_gate_loss_weight", 1.0))
        self.quality_gate_noise_std = float(
            config.get("quality_gate_noise_std", 0.25))
        if not 0.0 <= self.quality_gate_threshold <= 1.0:
            raise ValueError("quality_gate_threshold must be within [0, 1]")
        if not 0.0 <= self.quality_gate_margin <= 1.0:
            raise ValueError("quality_gate_margin must be within [0, 1]")
        if self.quality_gate_loss_weight < 0.0:
            raise ValueError("quality_gate_loss_weight must be >= 0")
        if self.quality_gate_noise_std <= 0.0:
            raise ValueError("quality_gate_noise_std must be > 0")
        self.quality_gate = None
        if self.quality_gate_enabled:
            gate_hidden_dim = int(config.get("quality_gate_hidden_dim", 128))
            if gate_hidden_dim <= 0:
                raise ValueError("quality_gate_hidden_dim must be > 0")
            self.quality_gate = BoundaryQualityGate(
                reg_feature_dim, gate_hidden_dim, num_feature_levels,
                float(config.get("quality_gate_dropout", 0.1)),
            )

    @staticmethod
    def residual_target(coarse: Tensor, target: Tensor) -> Tensor:
        """Legacy normalized start/end residual used by pre-V15 experiments."""
        duration = (coarse[:, 1] - coarse[:, 0]).clamp_min(1e-6)
        return (target - coarse) / duration[:, None]

    def state_target(self, coarse: Tensor, target: Tensor) -> Tensor:
        """Encode a target boundary in the configured diffusion state space."""
        if self.state_space == "endpoint":
            return self.residual_target(coarse, target)
        coarse_duration = (
            coarse[:, 1] - coarse[:, 0]).clamp_min(self.min_duration)
        target_duration = (
            target[:, 1] - target[:, 0]).clamp_min(self.min_duration)
        center_shift = (
            target.mean(dim=-1) - coarse.mean(dim=-1)
        ) / coarse_duration
        log_duration_ratio = torch.log(target_duration / coarse_duration)
        return torch.stack((center_shift, log_duration_ratio), dim=-1)

    def decode_state(self, coarse: Tensor, state: Tensor) -> Tensor:
        """Decode a boundary state; center/log-duration is legal by design."""
        duration = (
            coarse[:, 1] - coarse[:, 0]).clamp_min(self.min_duration)
        if self.state_space == "endpoint":
            return coarse + state * duration[:, None]
        center = coarse.mean(dim=-1) + state[:, 0] * duration
        decoded_duration = duration * torch.exp(
            state[:, 1].clamp(-10.0, 10.0))
        return torch.stack((
            center - 0.5 * decoded_duration,
            center + 0.5 * decoded_duration,
        ), dim=-1)

    @staticmethod
    def temporal_iou(candidate: Tensor, target: Tensor) -> Tensor:
        intersection = (
            torch.minimum(candidate[:, 1], target[:, 1])
            - torch.maximum(candidate[:, 0], target[:, 0])
        ).clamp_min(0.0)
        union = (
            (candidate[:, 1] - candidate[:, 0]).clamp_min(0.0)
            + (target[:, 1] - target[:, 0]).clamp_min(0.0)
            - intersection
        ).clamp_min(1e-6)
        return intersection / union

    def _quality_gate_training_loss(
        self,
        coarse: Tensor,
        target: Tensor,
        clean_residual: Tensor,
        predicted_clean: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
    ) -> Tensor:
        perturbed = clean_residual + torch.randn_like(clean_residual) * \
            self.quality_gate_noise_std
        candidates = torch.cat((
            torch.zeros_like(clean_residual),
            clean_residual,
            predicted_clean.detach().clamp(-2.0, 2.0),
            perturbed,
        ), dim=0).detach()
        repeated_coarse = coarse.detach().repeat(4, 1)
        repeated_target = target.detach().repeat(4, 1)
        candidate_boundaries = self.decode_state(
            repeated_coarse, candidates)
        quality_target = self.temporal_iou(
            candidate_boundaries, repeated_target).detach()
        logits = self.quality_gate(
            candidates, reg_features.detach().repeat(4, 1),
            repeated_coarse, level_ids.detach().repeat(4),
        )
        return F.binary_cross_entropy_with_logits(
            logits.float(), quality_target.float(), reduction="mean")

    def candidate_ranker_loss(
        self,
        coarse: Tensor,
        candidates: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        proposal_scores: Tensor,
        target_delta_tiou: Tensor,
        bce_pos_weight: Optional[Tensor] = None,
        coarse_context: Optional[Tensor] = None,
        candidate_context: Optional[Tensor] = None,
    ):
        if not self.candidate_ranker_enabled:
            raise ValueError("candidate ranker is not enabled")
        if target_delta_tiou.shape != candidates.shape[:2]:
            raise ValueError("target_delta_tiou must have shape [N, K]")
        predicted_delta, improve_logits, ranking_logits = \
            self.candidate_ranker(
                coarse, candidates, reg_features, level_ids,
                proposal_scores, coarse_context, candidate_context)
        sample_weights = proposal_scores.detach().float().clamp_min(1e-6).pow(
            self.candidate_ranker_score_weight_power)
        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)

        def weighted_sample_mean(values):
            return (values * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)

        negative_multiplier = torch.where(
            target_delta_tiou < -1e-6,
            target_delta_tiou.new_full(
                target_delta_tiou.shape,
                self.candidate_ranker_negative_utility_weight),
            torch.ones_like(target_delta_tiou))
        regression_per_candidate = F.smooth_l1_loss(
            predicted_delta.float() * self.candidate_ranker_delta_scale,
            target_delta_tiou.float() * self.candidate_ranker_delta_scale,
            reduction="none") * negative_multiplier
        regression_loss = weighted_sample_mean(
            regression_per_candidate.mean(dim=1))
        improve_target = (target_delta_tiou > 1e-6).to(
            improve_logits.dtype)
        bce_per_candidate = F.binary_cross_entropy_with_logits(
            improve_logits.float(), improve_target.float(),
            pos_weight=(bce_pos_weight.float()
                        if bce_pos_weight is not None else None),
            reduction="none") * negative_multiplier
        bce_loss = weighted_sample_mean(bce_per_candidate.mean(dim=1))
        best_delta, best_candidate = target_delta_tiou.max(dim=1)
        # Class zero is exact coarse fallback; refined candidates are 1..K.
        listwise_target = torch.where(
            best_delta > 1e-6, best_candidate + 1,
            torch.zeros_like(best_candidate))
        coarse_logit = ranking_logits.new_zeros(
            (ranking_logits.shape[0], 1))
        all_ranking_logits = torch.cat(
            (coarse_logit, ranking_logits), dim=1).float()
        if self.candidate_ranker_soft_listwise:
            coarse_delta = target_delta_tiou.new_zeros(
                (target_delta_tiou.shape[0], 1))
            target_probabilities = torch.softmax(
                torch.cat((coarse_delta, target_delta_tiou), dim=1).float()
                / self.candidate_ranker_softmax_temperature, dim=1)
            listwise_per_sample = -(
                target_probabilities
                * F.log_softmax(all_ranking_logits, dim=1)
            ).sum(dim=1)
        else:
            listwise_per_sample = F.cross_entropy(
                all_ranking_logits, listwise_target.long(), reduction="none")
        # If every refined option is harmful, penalize ranking mistakes more.
        harmful_sample = (target_delta_tiou.max(dim=1).values < -1e-6).to(
            listwise_per_sample.dtype)
        listwise_per_sample = listwise_per_sample * (
            1.0 + harmful_sample
            * (self.candidate_ranker_negative_utility_weight - 1.0))
        listwise_loss = weighted_sample_mean(listwise_per_sample)
        total = (
            self.candidate_ranker_regression_weight * regression_loss
            + self.candidate_ranker_bce_weight * bce_loss
            + self.candidate_ranker_listwise_weight * listwise_loss)
        return {
            "total": total,
            "regression": regression_loss,
            "bce": bce_loss,
            "listwise": listwise_loss,
            "predicted_delta": predicted_delta,
            "improve_logits": improve_logits,
            "ranking_logits": ranking_logits,
        }

    def candidate_quality_ranker_loss(
        self,
        coarse: Tensor,
        candidates: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        proposal_scores: Tensor,
        coarse_tiou: Tensor,
        candidate_tiou: Tensor,
    ):
        if not self.candidate_ranker_quality_enabled:
            raise ValueError("candidate quality ranker is not enabled")
        if coarse_tiou.shape != (coarse.shape[0],):
            raise ValueError("coarse_tiou must have shape [N]")
        if candidate_tiou.shape != candidates.shape[:2]:
            raise ValueError("candidate_tiou must have shape [N, K]")
        quality_logits = self.candidate_ranker.predict_quality(
            coarse, candidates, reg_features, level_ids, proposal_scores)
        all_tiou = torch.cat((coarse_tiou[:, None], candidate_tiou), dim=1)
        thresholds = all_tiou.new_tensor(
            self.candidate_ranker_tiou_thresholds)
        threshold_weights = all_tiou.new_tensor(
            self.candidate_ranker_threshold_weights)
        threshold_weights = threshold_weights / threshold_weights.sum()
        quality_targets = (
            all_tiou[:, :, None] >= thresholds[None, None, :]
        ).to(quality_logits.dtype)
        bce = F.binary_cross_entropy_with_logits(
            quality_logits.float(), quality_targets.float(), reduction="none")
        probabilities = torch.sigmoid(quality_logits.float())
        target_probability = torch.where(
            quality_targets.bool(), probabilities, 1.0 - probabilities)
        focal_weight = (1.0 - target_probability).pow(
            self.candidate_ranker_quality_focal_gamma)
        sample_weights = proposal_scores.detach().float().clamp_min(1e-6).pow(
            self.candidate_ranker_score_weight_power)
        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)
        weighted_threshold_bce = (
            bce * focal_weight * threshold_weights[None, None, :]
        ).sum(dim=-1).mean(dim=1)
        quality_bce_loss = (
            weighted_threshold_bce * sample_weights).sum() \
            / sample_weights.sum().clamp_min(1e-6)

        quality_scores = (
            probabilities * threshold_weights[None, None, :]).sum(dim=-1)
        coarse_quality = quality_scores[:, 0]
        candidate_quality = quality_scores[:, 1:]
        predicted_advantage = candidate_quality - coarse_quality[:, None]
        true_delta = candidate_tiou - coarse_tiou[:, None]
        candidate_targets = quality_targets[:, 1:]
        coarse_targets = quality_targets[:, :1]
        crossing_importance = (
            (candidate_targets - coarse_targets).abs()
            * threshold_weights[None, None, :]).sum(dim=-1)
        pair_importance = true_delta.abs() + crossing_importance
        pair_importance = pair_importance * torch.where(
            true_delta < -1e-6,
            true_delta.new_full(
                true_delta.shape,
                self.candidate_ranker_quality_harmful_weight),
            torch.ones_like(true_delta))
        active_pairs = true_delta.abs() > 1e-6
        pair_importance = pair_importance * active_pairs.to(pair_importance.dtype)
        pairwise = F.softplus(
            -torch.sign(true_delta) * predicted_advantage
            / self.candidate_ranker_quality_pairwise_temperature)
        weighted_pairs = pair_importance * sample_weights[:, None]
        pairwise_loss = (
            (pairwise * weighted_pairs).sum()
            / weighted_pairs.sum().clamp_min(1e-6))

        best_candidate = all_tiou.argmax(dim=1)
        listwise_per_sample = F.cross_entropy(
            quality_scores / self.candidate_ranker_quality_pairwise_temperature,
            best_candidate.long(), reduction="none")
        listwise_loss = (
            listwise_per_sample * sample_weights).sum() \
            / sample_weights.sum().clamp_min(1e-6)
        total = (
            self.candidate_ranker_quality_bce_weight * quality_bce_loss
            + self.candidate_ranker_quality_pairwise_weight * pairwise_loss
            + self.candidate_ranker_quality_listwise_weight * listwise_loss)
        return {
            "total": total,
            # Keep generic logging keys compatible with the ranker trainer.
            "regression": pairwise_loss,
            "bce": quality_bce_loss,
            "listwise": listwise_loss,
            "quality_logits": quality_logits,
            "quality_scores": quality_scores,
            "predicted_advantage": predicted_advantage,
        }

    def selected_candidate_profiles(self):
        if self.selected_profile_seeds.numel() == 0:
            return []
        max_shift = None
        if self.candidate_ranker_profile_pool:
            max_shift = self.candidate_ranker_profile_pool[0].get(
                "max_relative_shift")
        return [{
            "seed": int(seed), "steps": int(steps), "alpha": float(alpha),
            "max_relative_shift": max_shift,
        } for seed, steps, alpha in zip(
            self.selected_profile_seeds.detach().cpu().tolist(),
            self.selected_profile_steps.detach().cpu().tolist(),
            self.selected_profile_alphas.detach().cpu().tolist())]

    def set_selected_candidate_profiles(self, profiles):
        expected = self.candidate_ranker_num_candidates
        if len(profiles) != expected:
            raise ValueError(
                "expected {} selected candidate profiles".format(expected))
        device = self.selected_profile_seeds.device
        self.selected_profile_seeds.copy_(torch.tensor(
            [item["seed"] for item in profiles], dtype=torch.long,
            device=device))
        self.selected_profile_steps.copy_(torch.tensor(
            [item["steps"] for item in profiles], dtype=torch.long,
            device=device))
        self.selected_profile_alphas.copy_(torch.tensor(
            [item["alpha"] for item in profiles], dtype=torch.float32,
            device=device))

    def candidate_profile_features(self, device=None):
        """Return normalized seed/steps/alpha/shift descriptors for V9."""
        profiles = self.selected_candidate_profiles()
        if not profiles:
            return torch.empty((0, 4), device=device)
        max_seed = max(max(item["seed"] for item in profiles), 1)
        max_steps = max(max(item["steps"] for item in profiles), 1)
        features = torch.tensor([[
            item["seed"] / max_seed,
            math.log1p(item["steps"]) / math.log1p(max_steps),
            item["alpha"],
            float(item.get("max_relative_shift") or 0.0),
        ] for item in profiles], dtype=torch.float32, device=device)
        return features

    def set_candidate_ranker_calibrated_threshold(self, value):
        if self.candidate_ranker_target_type != "oracle_distilled_utility":
            raise ValueError("calibrated utility threshold is V9-only")
        self.candidate_ranker_calibrated_threshold.copy_(
            torch.as_tensor(value, dtype=torch.float32,
                            device=self.candidate_ranker_calibrated_threshold.device))

    def set_candidate_ranker_score_normalization(self, mean, std):
        if not self.candidate_ranker_score_normalization:
            raise ValueError("candidate score normalization is disabled")
        if float(std) <= 0.0:
            raise ValueError("candidate score normalization std must be > 0")
        self.candidate_ranker_score_mean.copy_(torch.as_tensor(
            mean, dtype=torch.float32,
            device=self.candidate_ranker_score_mean.device))
        self.candidate_ranker_score_std.copy_(torch.as_tensor(
            std, dtype=torch.float32,
            device=self.candidate_ranker_score_std.device))

    def install_candidate_ranker_ensemble(self, members, calibration):
        """Install independently trained V9.2 members from checkpoint metadata."""
        if not self.candidate_ranker_ensemble_enabled:
            raise ValueError("candidate ranker ensemble is disabled")
        if len(members) != self.candidate_ranker_ensemble_size:
            raise ValueError(
                "checkpoint ensemble size does not match the V9.2 config")
        device = next(self.candidate_ranker.parameters()).device
        rankers, means, stds, zero_thresholds = [], [], [], []
        for member in members:
            ranker = copy.deepcopy(self.candidate_ranker)
            ranker.load_state_dict(member["state_dict"], strict=True)
            ranker.to(device)
            ranker.eval()
            ranker.requires_grad_(False)
            mean = float(member["score_normalization"]["mean"])
            std = float(member["score_normalization"]["std"])
            if std <= 0.0:
                raise ValueError("ensemble member score std must be > 0")
            rankers.append(ranker)
            means.append(mean)
            stds.append(std)
            zero_thresholds.append(-mean / std)
        minimum_agreement = int(calibration["minimum_agreement"])
        if not 1 <= minimum_agreement <= len(rankers):
            raise ValueError("ensemble minimum agreement is out of range")
        maximum_std = float(calibration["maximum_std"])
        if maximum_std < 0.0:
            raise ValueError("ensemble maximum std must be >= 0")
        self.candidate_ranker_ensemble = nn.ModuleList(rankers)
        self.candidate_ranker_ensemble_score_means = torch.tensor(
            means, dtype=torch.float32, device=device)
        self.candidate_ranker_ensemble_score_stds = torch.tensor(
            stds, dtype=torch.float32, device=device)
        self.candidate_ranker_ensemble_zero_thresholds = torch.tensor(
            zero_thresholds, dtype=torch.float32, device=device)
        self.candidate_ranker_ensemble_threshold.copy_(torch.as_tensor(
            calibration["threshold"], dtype=torch.float32,
            device=self.candidate_ranker_ensemble_threshold.device))
        self.candidate_ranker_ensemble_max_std.copy_(torch.as_tensor(
            maximum_std, dtype=torch.float32,
            device=self.candidate_ranker_ensemble_max_std.device))
        self.candidate_ranker_ensemble_max_low_tiou_harm_rate.copy_(
            torch.as_tensor(
                calibration.get("maximum_low_tiou_harm_rate", 1.0),
                dtype=torch.float32,
                device=self.candidate_ranker_ensemble_max_low_tiou_harm_rate.device))
        self.candidate_ranker_ensemble_min_agreement = minimum_agreement

    def _candidate_ranker_ensemble_predictions(
        self,
        coarse: Tensor,
        candidates: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        proposal_scores: Tensor,
        class_ids: Tensor,
        coarse_context: Tensor,
        candidate_context: Tensor,
        profile_features: Tensor,
    ):
        if len(self.candidate_ranker_ensemble) != \
                self.candidate_ranker_ensemble_size:
            raise RuntimeError(
                "V9.2 ensemble metadata was not installed from checkpoint")
        utilities, ious = [], []
        for ranker in self.candidate_ranker_ensemble:
            utility, iou = ranker(
                coarse, candidates, reg_features, level_ids, proposal_scores,
                class_ids, coarse_context, candidate_context, profile_features)
            utilities.append(utility)
            ious.append(iou)
        raw_utility = torch.stack(utilities)
        means = self.candidate_ranker_ensemble_score_means[:, None, None]
        stds = self.candidate_ranker_ensemble_score_stds[:, None, None]
        normalized_utility = (raw_utility - means) / stds.clamp_min(1e-6)
        return raw_utility, normalized_utility, torch.stack(ious)

    @torch.no_grad()
    def select_oracle_distilled_ensemble_candidates(
        self,
        coarse: Tensor,
        candidates: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        proposal_scores: Tensor,
        class_ids: Tensor,
        coarse_context: Tensor,
        candidate_context: Tensor,
    ):
        profile_features = self.candidate_profile_features(coarse.device)
        raw_utility, normalized_utility, predicted_iou = \
            self._candidate_ranker_ensemble_predictions(
                coarse, candidates, reg_features, level_ids, proposal_scores,
                class_ids, coarse_context, candidate_context, profile_features)
        ensemble_utility = normalized_utility.mean(dim=0)
        best_utility, best_candidate = ensemble_utility.max(dim=1)
        gather_index = best_candidate[None, :, None].expand(
            normalized_utility.shape[0], -1, 1)
        member_selected = normalized_utility.gather(
            2, gather_index).squeeze(2)
        member_best = normalized_utility.argmax(dim=2)
        votes = (
            (member_best == best_candidate[None, :])
            & (member_selected
               > self.candidate_ranker_ensemble_zero_thresholds[:, None]))
        agreement_count = votes.sum(dim=0)
        uncertainty = member_selected.std(dim=0, unbiased=False)
        threshold = float(self.candidate_ranker_ensemble_threshold.item())
        maximum_std = float(self.candidate_ranker_ensemble_max_std.item())
        accepts = (
            (best_utility > threshold)
            & (agreement_count
               >= self.candidate_ranker_ensemble_min_agreement)
            & (uncertainty <= maximum_std))
        selected = coarse.clone()
        accepted_indices = torch.nonzero(accepts, as_tuple=True)[0]
        if accepted_indices.numel():
            selected[accepted_indices] = candidates[
                accepted_indices, best_candidate[accepted_indices]]

        mean_iou = predicted_iou.mean(dim=0)
        score_factors = torch.ones_like(proposal_scores)
        if self.candidate_ranker_rescore_enabled and accepted_indices.numel():
            zero_profile = profile_features.new_zeros((1, 4))
            coarse_ious = []
            for ranker in self.candidate_ranker_ensemble:
                _, coarse_iou = ranker(
                    coarse, coarse[:, None], reg_features, level_ids,
                    proposal_scores, class_ids, coarse_context,
                    coarse_context[:, None], zero_profile)
                coarse_ious.append(coarse_iou[:, 0])
            reference_iou = torch.stack(
                coarse_ious).mean(dim=0)[accepted_indices].clamp_min(1e-3)
            candidate_iou = mean_iou[
                accepted_indices, best_candidate[accepted_indices]]
            factor = 1.0 + self.candidate_ranker_rescore_gamma * (
                candidate_iou / reference_iou - 1.0)
            change = self.candidate_ranker_rescore_max_change
            score_factors[accepted_indices] = factor.clamp(
                min=1.0 - change, max=1.0 + change)

        stats = {
            "v9_accepts": int(accepts.sum().item()),
            "v9_fallbacks": int((~accepts).sum().item()),
            "v9_accept_ratio": float(accepts.float().mean().item())
            if accepts.numel() else 0.0,
            "v9_rescored": int(accepted_indices.numel())
            if self.candidate_ranker_rescore_enabled else 0,
            "v9_mean_score_factor": float(
                score_factors[accepted_indices].mean().item())
            if (self.candidate_ranker_rescore_enabled
                and accepted_indices.numel()) else 1.0,
            "v92_ensemble_members": len(self.candidate_ranker_ensemble),
            "v92_minimum_agreement":
                self.candidate_ranker_ensemble_min_agreement,
            "v92_calibrated_threshold": threshold,
            "v92_maximum_std": maximum_std,
            "v92_mean_agreement": float(
                agreement_count.float().mean().item())
            if agreement_count.numel() else 0.0,
            "v92_mean_accepted_agreement": float(
                agreement_count[accepts].float().mean().item())
            if accepts.any() else 0.0,
            "v92_accepted_agreement_sum": float(
                agreement_count[accepts].float().sum().item())
            if accepts.any() else 0.0,
            "v92_mean_uncertainty": float(uncertainty.mean().item())
            if uncertainty.numel() else 0.0,
            "v92_mean_accepted_uncertainty": float(
                uncertainty[accepts].mean().item())
            if accepts.any() else 0.0,
            "v92_accepted_uncertainty_sum": float(
                uncertainty[accepts].sum().item())
            if accepts.any() else 0.0,
            "v92_mean_ensemble_utility": float(
                best_utility[accepts].mean().item())
            if accepts.any() else 0.0,
            "v92_mean_predicted_iou": float(mean_iou[
                accepted_indices, best_candidate[accepted_indices]
            ].mean().item()) if accepted_indices.numel() else 0.0,
        }
        if self.candidate_ranker_low_tiou_safety_enabled:
            stats.update({
                "v93_low_tiou_safety_enabled": 1.0,
                "v93_calibrated_max_low_tiou_harm_rate": float(
                    self.candidate_ranker_ensemble_max_low_tiou_harm_rate
                    .item()),
            })
        return selected, score_factors, stats

    def candidate_oracle_distillation_loss(
        self,
        coarse: Tensor,
        candidates: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        proposal_scores: Tensor,
        class_ids: Tensor,
        coarse_context: Tensor,
        candidate_context: Tensor,
        target_utility: Tensor,
        candidate_tiou: Tensor,
        iou_weights: Optional[Tensor] = None,
    ):
        profile_features = self.candidate_profile_features(coarse.device)
        predicted_utility, predicted_iou = self.candidate_ranker(
            coarse, candidates, reg_features, level_ids, proposal_scores,
            class_ids, coarse_context, candidate_context, profile_features)
        scaled_target = target_utility.float() * self.candidate_ranker_delta_scale
        utility_weights = torch.ones_like(target_utility)
        utility_weights = torch.where(
            target_utility < 0.0,
            utility_weights * self.candidate_ranker_negative_utility_weight,
            utility_weights)
        utility_weights = utility_weights * (
            1.0 + target_utility.abs() / target_utility.abs().mean().clamp_min(1e-4))
        utility_loss = (
            F.smooth_l1_loss(
                predicted_utility.float(), scaled_target,
                reduction="none") * utility_weights
        ).sum() / utility_weights.sum().clamp_min(1.0)
        if iou_weights is None:
            iou_weights = torch.ones_like(candidate_tiou)
        iou_loss = (
            F.smooth_l1_loss(
                predicted_iou.float(), candidate_tiou.float(),
                reduction="none") * iou_weights
        ).sum() / iou_weights.sum().clamp_min(1.0)
        listwise_loss = predicted_utility.new_zeros(())
        regret_loss = predicted_utility.new_zeros(())
        if self.candidate_ranker_decision_loss_enabled:
            coarse_score = predicted_utility.new_zeros(
                (predicted_utility.shape[0], 1))
            scores = torch.cat((coarse_score, predicted_utility), dim=1)
            coarse_target = target_utility.new_zeros(
                (target_utility.shape[0], 1))
            targets = torch.cat((coarse_target, target_utility), dim=1)
            oracle_target, oracle_index = targets.max(dim=1)
            temperature = self.candidate_ranker_ranking_temperature
            sample_weight = 1.0 + (
                oracle_target.clamp_min(0.0)
                * self.candidate_ranker_delta_scale)
            listwise_per_sample = F.cross_entropy(
                scores / temperature, oracle_index, reduction="none")
            listwise_loss = (
                listwise_per_sample * sample_weight
            ).sum() / sample_weight.sum().clamp_min(1.0)
            regret = (oracle_target[:, None] - targets).clamp_min(0.0)
            predicted_probability = torch.softmax(
                scores / temperature, dim=1)
            regret_loss = (
                predicted_probability * regret
                * self.candidate_ranker_delta_scale
            ).sum(dim=1)
            regret_loss = (
                regret_loss * sample_weight
            ).sum() / sample_weight.sum().clamp_min(1.0)
        total = (
            utility_loss
            + self.candidate_ranker_iou_weight * iou_loss
            + self.candidate_ranker_listwise_weight * listwise_loss
            + self.candidate_ranker_regret_weight * regret_loss)
        return {
            "total": total,
            "utility": utility_loss,
            "iou": iou_loss,
            "listwise": listwise_loss,
            "regret": regret_loss,
            "predicted_utility": predicted_utility,
            "predicted_iou": predicted_iou,
        }

    @torch.no_grad()
    def select_oracle_distilled_candidates(
        self,
        coarse: Tensor,
        candidates: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        proposal_scores: Tensor,
        class_ids: Tensor,
        coarse_context: Tensor,
        candidate_context: Tensor,
    ):
        if self.candidate_ranker_ensemble_enabled:
            return self.select_oracle_distilled_ensemble_candidates(
                coarse, candidates, reg_features, level_ids, proposal_scores,
                class_ids, coarse_context, candidate_context)
        profile_features = self.candidate_profile_features(coarse.device)
        predicted_utility, predicted_iou = self.candidate_ranker(
            coarse, candidates, reg_features, level_ids, proposal_scores,
            class_ids, coarse_context, candidate_context, profile_features)
        best_utility, best_candidate = predicted_utility.max(dim=1)
        normalized_utility = best_utility
        if self.candidate_ranker_score_normalization:
            normalized_utility = (
                best_utility - self.candidate_ranker_score_mean
            ) / self.candidate_ranker_score_std.clamp_min(1e-6)
        threshold = float(self.candidate_ranker_calibrated_threshold.item())
        accepts = normalized_utility > threshold
        selected = coarse.clone()
        accepted_indices = torch.nonzero(accepts, as_tuple=True)[0]
        if accepted_indices.numel():
            selected[accepted_indices] = candidates[
                accepted_indices, best_candidate[accepted_indices]]
        score_factors = torch.ones_like(proposal_scores)
        if self.candidate_ranker_rescore_enabled and accepted_indices.numel():
            zero_profile = profile_features.new_zeros((1, 4))
            _, coarse_iou = self.candidate_ranker(
                coarse, coarse[:, None], reg_features, level_ids,
                proposal_scores, class_ids, coarse_context,
                coarse_context[:, None], zero_profile)
            candidate_iou = predicted_iou[
                accepted_indices, best_candidate[accepted_indices]]
            reference_iou = coarse_iou[accepted_indices, 0].clamp_min(1e-3)
            factor = 1.0 + self.candidate_ranker_rescore_gamma * (
                candidate_iou / reference_iou - 1.0)
            change = self.candidate_ranker_rescore_max_change
            score_factors[accepted_indices] = factor.clamp(
                min=1.0 - change, max=1.0 + change)
        stats = {
            "v9_accepts": int(accepts.sum().item()),
            "v9_fallbacks": int((~accepts).sum().item()),
            "v9_accept_ratio": float(accepts.float().mean().item())
            if accepts.numel() else 0.0,
            "v9_calibrated_threshold": threshold,
            "v9_mean_predicted_utility": float(
                best_utility[accepts].mean().item()) if accepts.any() else 0.0,
            "v9_mean_normalized_utility": float(
                normalized_utility[accepts].mean().item())
            if accepts.any() else 0.0,
            "v9_score_mean": float(self.candidate_ranker_score_mean.item()),
            "v9_score_std": float(self.candidate_ranker_score_std.item()),
            "v9_mean_predicted_iou": float(predicted_iou[
                accepted_indices, best_candidate[accepted_indices]
            ].mean().item()) if accepted_indices.numel() else 0.0,
            "v9_rescored": int(accepted_indices.numel())
            if self.candidate_ranker_rescore_enabled else 0,
            "v9_mean_score_factor": float(
                score_factors[accepted_indices].mean().item())
            if (self.candidate_ranker_rescore_enabled
                and accepted_indices.numel()) else 1.0,
        }
        return selected, score_factors, stats

    def candidate_profile_ranker_loss(
        self,
        coarse: Tensor,
        candidates: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        proposal_scores: Tensor,
        target_utility: Tensor,
        coarse_tiou: Tensor,
        candidate_tiou: Tensor,
        coarse_context: Tensor,
        candidate_context: Tensor,
        bce_pos_weight: Optional[Tensor] = None,
    ):
        """V8 relative utility, fallback ranking, and IoU calibration loss."""
        predicted, improve_logits, ranking_logits, predicted_iou = \
            self.candidate_ranker.predict_profile_values(
                coarse, candidates, reg_features, level_ids, proposal_scores,
                coarse_context, candidate_context)
        predicted_advantage = predicted[:, 1:] - predicted[:, :1]
        scaled_target = target_utility.float() * self.candidate_ranker_delta_scale
        sample_weights = proposal_scores.detach().float().clamp_min(1e-6).pow(
            self.candidate_ranker_score_weight_power)
        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)
        harmful = torch.where(
            target_utility < -1e-6,
            target_utility.new_full(
                target_utility.shape,
                self.candidate_ranker_negative_utility_weight),
            torch.ones_like(target_utility))
        regression_per = F.smooth_l1_loss(
            predicted_advantage.float(), scaled_target,
            reduction="none") * harmful
        regression_loss = (
            regression_per.mean(dim=1) * sample_weights).sum() \
            / sample_weights.sum().clamp_min(1e-6)

        improvement_target = (target_utility > 1e-6).to(improve_logits.dtype)
        candidate_improve_logits = improve_logits[:, 1:]
        bce_per = F.binary_cross_entropy_with_logits(
            candidate_improve_logits.float(), improvement_target.float(),
            pos_weight=bce_pos_weight, reduction="none") * harmful
        bce_loss = (bce_per.mean(dim=1) * sample_weights).sum() \
            / sample_weights.sum().clamp_min(1e-6)

        relative_ranking = ranking_logits - ranking_logits[:, :1]
        all_target = torch.cat((
            target_utility.new_zeros((target_utility.shape[0], 1)),
            target_utility), dim=1)
        target_probabilities = torch.softmax(
            all_target.float() / self.candidate_ranker_softmax_temperature,
            dim=1)
        listwise_per = -(
            target_probabilities
            * F.log_softmax(relative_ranking.float(), dim=1)).sum(dim=1)
        listwise_loss = (listwise_per * sample_weights).sum() \
            / sample_weights.sum().clamp_min(1e-6)

        all_tiou = torch.cat((coarse_tiou[:, None], candidate_tiou), dim=1)
        iou_per = F.smooth_l1_loss(
            predicted_iou.float(), all_tiou.float(), reduction="none").mean(1)
        iou_loss = (iou_per * sample_weights).sum() \
            / sample_weights.sum().clamp_min(1e-6)
        total = (
            self.candidate_ranker_regression_weight * regression_loss
            + self.candidate_ranker_bce_weight * bce_loss
            + self.candidate_ranker_listwise_weight * listwise_loss
            + self.candidate_ranker_iou_weight * iou_loss)
        return {
            "total": total, "regression": regression_loss,
            "bce": bce_loss, "listwise": listwise_loss,
            "iou": iou_loss,
            "predicted_delta": predicted_advantage,
            "improve_logits": candidate_improve_logits,
            "ranking_logits": relative_ranking[:, 1:],
            "all_ranking_logits": relative_ranking,
            "predicted_iou": predicted_iou,
        }

    @torch.no_grad()
    def select_profile_ranked_candidates(
        self,
        coarse: Tensor,
        candidates: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        proposal_scores: Tensor,
        coarse_context: Tensor,
        candidate_context: Tensor,
    ):
        predicted, improve_logits, ranking_logits, predicted_iou = \
            self.candidate_ranker.predict_profile_values(
                coarse, candidates, reg_features, level_ids, proposal_scores,
                coarse_context, candidate_context)
        advantage = predicted[:, 1:] - predicted[:, :1]
        ranking_advantage = ranking_logits[:, 1:] - ranking_logits[:, :1]
        improve_probability = torch.sigmoid(improve_logits[:, 1:])
        eligible = (
            (advantage > self.candidate_ranker_delta_margin)
            & (improve_probability
               >= self.candidate_ranker_improve_threshold)
            & (ranking_advantage > 0.0))
        selection_score = ranking_advantage + advantage
        selection_score = selection_score.masked_fill(
            ~eligible, float("-inf"))
        best_score, best_candidate = selection_score.max(dim=1)
        accepts = torch.isfinite(best_score)
        selected = coarse.clone()
        selected_indices = torch.nonzero(accepts, as_tuple=True)[0]
        if selected_indices.numel():
            selected[selected_indices] = candidates[
                selected_indices, best_candidate[selected_indices]]
        score_factors = torch.ones_like(proposal_scores)
        if self.candidate_ranker_rescore_enabled and selected_indices.numel():
            coarse_iou = predicted_iou[selected_indices, 0].clamp_min(1e-3)
            candidate_iou = predicted_iou[
                selected_indices, best_candidate[selected_indices] + 1]
            factor = 1.0 + self.candidate_ranker_rescore_gamma * (
                candidate_iou / coarse_iou - 1.0)
            change = self.candidate_ranker_rescore_max_change
            score_factors[selected_indices] = factor.clamp(
                min=1.0 - change, max=1.0 + change)
        stats = {
            "v8_accepts": int(accepts.sum().item()),
            "v8_fallbacks": int((~accepts).sum().item()),
            "v8_accept_ratio": float(accepts.float().mean().item())
            if accepts.numel() else 0.0,
            "v8_mean_predicted_utility": float(
                advantage[selected_indices,
                          best_candidate[selected_indices]].mean().item())
            if selected_indices.numel() else 0.0,
            "v8_mean_predicted_iou": float(
                predicted_iou[selected_indices,
                              best_candidate[selected_indices] + 1].mean().item())
            if selected_indices.numel() else 0.0,
            "v8_rescored": int(selected_indices.numel())
            if self.candidate_ranker_rescore_enabled else 0,
            "v8_mean_score_factor": float(
                score_factors[selected_indices].mean().item())
            if (self.candidate_ranker_rescore_enabled
                and selected_indices.numel()) else 1.0,
        }
        return selected, score_factors, stats

    @torch.no_grad()
    def select_quality_ranked_candidates(
        self,
        coarse: Tensor,
        candidates: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        proposal_scores: Tensor,
    ):
        if not self.candidate_ranker_quality_enabled:
            raise ValueError("candidate quality ranker is not enabled")
        quality_logits = self.candidate_ranker.predict_quality(
            coarse, candidates, reg_features, level_ids, proposal_scores)
        weights = quality_logits.new_tensor(
            self.candidate_ranker_threshold_weights)
        weights = weights / weights.sum()
        quality_scores = (
            torch.sigmoid(quality_logits) * weights[None, None, :]
        ).sum(dim=-1)
        coarse_quality = quality_scores[:, 0]
        candidate_quality = quality_scores[:, 1:]
        best_quality, best_candidate = candidate_quality.max(dim=1)
        advantage = best_quality - coarse_quality
        accepts = advantage > self.candidate_ranker_quality_margin
        selected = coarse.clone()
        proposal_indices = torch.nonzero(accepts, as_tuple=True)[0]
        if proposal_indices.numel() > 0:
            selected[proposal_indices] = candidates[
                proposal_indices, best_candidate[proposal_indices]]
        score_factors = torch.ones_like(proposal_scores)
        if (self.candidate_ranker_quality_rescore_enabled
                and proposal_indices.numel() > 0):
            ratio = (
                best_quality[proposal_indices].clamp_min(1e-6)
                / coarse_quality[proposal_indices].clamp_min(1e-6))
            factor = 1.0 + self.candidate_ranker_quality_rescore_gamma * (
                ratio - 1.0)
            max_change = self.candidate_ranker_quality_rescore_max_change
            score_factors[proposal_indices] = factor.clamp(
                min=1.0 - max_change, max=1.0 + max_change)
        stats = {
            "candidate_quality_accepts": int(accepts.sum().item()),
            "candidate_quality_fallbacks": int((~accepts).sum().item()),
            "candidate_quality_accept_ratio": float(
                accepts.float().mean().item()) if accepts.numel() else 0.0,
            "candidate_quality_mean_coarse": float(
                coarse_quality.mean().item()) if coarse_quality.numel() else 0.0,
            "candidate_quality_mean_selected": float(
                best_quality[proposal_indices].mean().item())
            if proposal_indices.numel() else 0.0,
            "candidate_quality_mean_advantage": float(
                advantage[proposal_indices].mean().item())
            if proposal_indices.numel() else 0.0,
            "candidate_quality_rescored": int(
                proposal_indices.numel()
                if self.candidate_ranker_quality_rescore_enabled else 0),
            "candidate_quality_mean_score_factor": float(
                score_factors[proposal_indices].mean().item())
            if (self.candidate_ranker_quality_rescore_enabled
                and proposal_indices.numel()) else 1.0,
        }
        return selected, score_factors, stats

    @torch.no_grad()
    def select_ranked_candidates(
        self,
        coarse: Tensor,
        candidates: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        proposal_scores: Tensor,
        coarse_context: Optional[Tensor] = None,
        candidate_context: Optional[Tensor] = None,
    ):
        if not self.candidate_ranker_enabled:
            raise ValueError("candidate ranker is not enabled")
        predicted_delta, improve_logits, ranking_logits = \
            self.candidate_ranker(
                coarse, candidates, reg_features, level_ids,
                proposal_scores, coarse_context, candidate_context)
        improve_probability = torch.sigmoid(improve_logits)
        eligible = (
            (predicted_delta > self.candidate_ranker_delta_margin)
            & (improve_probability
               >= self.candidate_ranker_improve_threshold)
            & (ranking_logits > 0.0))
        masked_ranking = ranking_logits.masked_fill(
            ~eligible, float("-inf"))
        best_ranking, best_candidate = masked_ranking.max(dim=1)
        accepts = torch.isfinite(best_ranking)
        selected = coarse.clone()
        proposal_indices = torch.nonzero(accepts, as_tuple=True)[0]
        if proposal_indices.numel() > 0:
            selected[proposal_indices] = candidates[
                proposal_indices, best_candidate[proposal_indices]]
        stats = {
            "candidate_ranker_accepts": int(accepts.sum().item()),
            "candidate_ranker_fallbacks": int((~accepts).sum().item()),
            "candidate_ranker_accept_ratio": float(
                accepts.float().mean().item()) if accepts.numel() else 0.0,
            "candidate_ranker_mean_predicted_delta": float(
                predicted_delta[proposal_indices,
                                best_candidate[proposal_indices]].mean().item())
            if proposal_indices.numel() else 0.0,
            "candidate_ranker_mean_improve_probability": float(
                improve_probability[proposal_indices,
                                    best_candidate[proposal_indices]].mean().item())
            if proposal_indices.numel() else 0.0,
        }
        if self.candidate_ranker_target_type != "delta_tiou":
            stats["candidate_ranker_mean_predicted_utility"] = stats[
                "candidate_ranker_mean_predicted_delta"]
        return selected, stats

    def _boundary_tiou_training_loss(
        self,
        coarse: Tensor,
        target: Tensor,
        predicted_clean: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """SNR-weighted localization loss on reconstructed clean boundaries."""
        bounded_state = predicted_clean.clamp(
            -self.boundary_loss_max_residual,
            self.boundary_loss_max_residual,
        )
        candidate = self.decode_state(coarse, bounded_state)
        localization_error = 1.0 - self.temporal_iou(candidate, target)
        alpha = self.scheduler.alphas_cumprod[
            timesteps.long()].to(predicted_clean.dtype)
        snr = alpha / (1.0 - alpha).clamp_min(1e-6)
        snr_weight = (snr / self.boundary_loss_snr_cap).clamp(max=1.0)
        return (localization_error.float() * snr_weight.float()).mean()

    def _v15b_min_snr_weights(self, timesteps: Tensor) -> Tensor:
        """Return mean-preserving Min-SNR weights for direct x0 prediction.

        For an x0 target the Min-SNR objective uses ``min(SNR, gamma)``.
        Normalizing within the sampled batch preserves the V15-A loss scale;
        the change is therefore timestep allocation rather than an implicit
        learning-rate change.
        """
        alpha = self.scheduler.alphas_cumprod[
            timesteps.long()].to(dtype=torch.float32)
        snr = alpha / (1.0 - alpha).clamp_min(1e-8)
        weights = snr.clamp(max=self.v15b_min_snr_gamma)
        return weights / weights.mean().clamp_min(1e-12)

    def _v15b_geometry_losses(
        self,
        coarse: Tensor,
        target: Tensor,
        predicted_clean: Tensor,
        timestep_weights: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Decode random-timestep x0 and align it with temporal geometry."""
        candidate = self.decode_state(coarse, predicted_clean)
        predicted_tiou = self.temporal_iou(candidate, target)

        thresholds = predicted_tiou.new_tensor(self.v15b_thresholds)
        distance = (
            predicted_tiou.detach()[:, None] - thresholds[None]).abs()
        proximity = torch.exp(
            -0.5 * (distance / self.v15b_threshold_sigma).square()
        ).amax(dim=1)
        geometry_weights = (
            1.0 + self.v15b_threshold_gain * proximity)
        combined_weights = timestep_weights * geometry_weights
        combined_weights = (
            combined_weights
            / combined_weights.mean().clamp_min(1e-8))

        tiou_loss = (
            (1.0 - predicted_tiou).float()
            * combined_weights.float()).mean()

        target_duration = (
            target[:, 1] - target[:, 0]).clamp_min(self.min_duration)
        normalized_error = (
            candidate - target) / target_duration[:, None]
        endpoint_error = F.smooth_l1_loss(
            normalized_error.float(),
            torch.zeros_like(normalized_error, dtype=torch.float32),
            reduction="none",
            beta=self.v15b_endpoint_beta,
        ).mean(dim=-1)
        endpoint_loss = (
            endpoint_error * combined_weights.float()).mean()
        return tiou_loss, endpoint_loss

    def _v15c1_identity_weights(
        self,
        coarse: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Continuously increase no-change supervision with coarse quality.

        Proposals below ``start`` receive no identity supervision. The weight
        rises linearly to one between ``start`` and ``full`` and then rises
        continuously to ``identity_max_weight`` at tIoU 1.0.
        """
        coarse_tiou = self.temporal_iou(coarse, target).detach()
        start = self.v15c1_identity_start_tiou
        full = self.v15c1_identity_full_tiou
        transition = ((coarse_tiou - start) / (full - start)).clamp(0.0, 1.0)
        high_quality = ((coarse_tiou - full) / (1.0 - full)).clamp(0.0, 1.0)
        return transition + (
            self.v15c1_identity_max_weight - 1.0) * high_quality

    def _v15c1_prior_losses(
        self,
        clean: Tensor,
        predicted_clean: Tensor,
        identity_prediction: Tensor,
        timestep_weights: Tensor,
        identity_weights: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Train an identity atom and suppress unnecessary duration expansion."""
        sample_weights = (
            timestep_weights.float() * identity_weights.float())
        identity_error = identity_prediction.float().square().mean(dim=-1)
        identity_loss = (identity_error * sample_weights).mean()

        # A positive log-duration target is allowed when the GT truly requires
        # expansion. Only the excess beyond that target is penalized. For the
        # identity branch any positive duration residual is unnecessary.
        allowed_positive_duration = clean[:, 1].detach().float().clamp_min(0.0)
        correction_excess = (
            predicted_clean[:, 1].float() - allowed_positive_duration
        ).clamp_min(0.0)
        identity_expansion = identity_prediction[:, 1].float().clamp_min(0.0)
        duration_error = (
            F.smooth_l1_loss(
                correction_excess,
                torch.zeros_like(correction_excess),
                reduction="none",
                beta=self.v15c1_duration_beta,
            )
            + F.smooth_l1_loss(
                identity_expansion,
                torch.zeros_like(identity_expansion),
                reduction="none",
                beta=self.v15c1_duration_beta,
            )
        )
        duration_asym_loss = (duration_error * sample_weights).mean()
        return identity_loss, duration_asym_loss

    def _differentiable_ddim_residual(
        self,
        initial_noise: Tensor,
        reg_features: Tensor,
        coarse: Tensor,
        level_ids: Tensor,
        boundary_context: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the deterministic DDIM update without disabling autograd."""
        sample = initial_noise
        timesteps = torch.linspace(
            self.scheduler.num_train_timesteps - 1, 0,
            self.final_ddim_train_steps, device=sample.device,
        ).round().long()
        for index, timestep in enumerate(timesteps):
            prediction = self.denoiser(
                sample, timestep.expand(sample.shape[0]), reg_features,
                coarse, level_ids, boundary_context)
            alpha_t = self.scheduler.alphas_cumprod[
                timestep].to(sample.dtype)
            if self.prediction_type == "epsilon":
                predicted_noise = prediction
                predicted_clean = (
                    sample - (1.0 - alpha_t).sqrt() * predicted_noise
                ) / alpha_t.sqrt().clamp_min(1e-6)
                predicted_clean = predicted_clean.clamp(-10.0, 10.0)
            else:
                predicted_clean = prediction.clamp(-10.0, 10.0)
                predicted_noise = (
                    sample - alpha_t.sqrt() * predicted_clean
                ) / (1.0 - alpha_t).sqrt().clamp_min(1e-6)
            if index == len(timesteps) - 1:
                sample = predicted_clean
            else:
                alpha_previous = self.scheduler.alphas_cumprod[
                    timesteps[index + 1]].to(sample.dtype)
                sample = (
                    alpha_previous.sqrt() * predicted_clean
                    + (1.0 - alpha_previous).sqrt() * predicted_noise)
        return sample

    def _final_ddim_training_loss(
        self,
        coarse: Tensor,
        target: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        boundary_context: Optional[Tensor] = None,
    ) -> Tensor:
        num_candidates = coarse.shape[0]
        if num_candidates > self.final_ddim_max_samples:
            selected = torch.randperm(
                num_candidates, device=coarse.device
            )[:self.final_ddim_max_samples]
            coarse = coarse[selected]
            target = target[selected]
            reg_features = reg_features[selected]
            level_ids = level_ids[selected]
            if boundary_context is not None:
                boundary_context = boundary_context[selected]
        initial_noise = torch.randn_like(coarse)
        residual = self._differentiable_ddim_residual(
            initial_noise, reg_features, coarse, level_ids,
            boundary_context)
        residual = residual.clamp(
            -self.final_ddim_max_residual,
            self.final_ddim_max_residual)
        # Match the residual scaling used by the actual evaluation path.
        residual = self.residual_blend_alpha * residual
        candidate = self.decode_state(coarse, residual)
        return (1.0 - self.temporal_iou(candidate, target)).float().mean()

    def training_loss(
        self, coarse: Tensor, target: Tensor, reg_features: Tensor, level_ids: Tensor,
        return_components: bool = False,
        boundary_context: Optional[Tensor] = None,
        return_v15b_components: bool = False,
        return_v15c1_components: bool = False,
    ):
        if coarse.shape[0] == 0:
            zero = reg_features.sum() * 0.0
            if return_v15c1_components:
                return (zero, zero, zero, zero, zero, zero, zero,
                        zero, zero)
            if return_v15b_components:
                return (zero, zero, zero, zero, zero, zero, zero)
            return (zero, zero, zero, zero, zero) \
                if return_components else zero
        coarse_condition = coarse.detach() if self.detach_coarse_boundary else coarse
        clean = self.state_target(coarse_condition, target)
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (clean.shape[0],), device=clean.device
        )
        noise = torch.randn_like(clean)
        noisy = self.scheduler.add_noise(clean, noise, timesteps)
        prediction = self.denoiser(
            noisy, timesteps, reg_features, coarse_condition, level_ids,
            boundary_context)
        denoising_target = noise if self.prediction_type == "epsilon" else clean
        denoising_error = F.mse_loss(
            prediction.float(), denoising_target.float(), reduction="none"
        ).mean(dim=-1)
        if self.v15b_enabled:
            timestep_weights = self._v15b_min_snr_weights(timesteps)
            denoising_loss = (
                denoising_error * timestep_weights).mean()
        else:
            timestep_weights = torch.ones_like(denoising_error)
            denoising_loss = denoising_error.mean()
        boundary_loss = denoising_loss.new_zeros(())
        final_ddim_loss = denoising_loss.new_zeros(())
        quality_gate_loss = denoising_loss.new_zeros(())
        v15b_tiou_loss = denoising_loss.new_zeros(())
        v15b_endpoint_loss = denoising_loss.new_zeros(())
        v15c1_identity_loss = denoising_loss.new_zeros(())
        v15c1_duration_asym_loss = denoising_loss.new_zeros(())
        predicted_clean = None
        if (self.boundary_loss_enabled or self.quality_gate_enabled
                or self.v15b_enabled):
            if self.prediction_type == "x0":
                predicted_clean = prediction
            else:
                alpha = self.scheduler.alphas_cumprod[
                    timesteps.long()].to(noisy.dtype)[:, None]
                predicted_clean = (
                    noisy - (1 - alpha).sqrt() * prediction
                ) / alpha.sqrt().clamp_min(1e-6)
        if self.boundary_loss_enabled:
            boundary_loss = self._boundary_tiou_training_loss(
                coarse_condition, target, predicted_clean, timesteps)
        if self.quality_gate_enabled:
            quality_gate_loss = self._quality_gate_training_loss(
                coarse_condition, target, clean, predicted_clean,
                reg_features, level_ids,
            )
        if self.v15b_enabled:
            v15b_tiou_loss, v15b_endpoint_loss = \
                self._v15b_geometry_losses(
                    coarse_condition, target, predicted_clean,
                    timestep_weights)
        if self.v15c1_enabled:
            identity_weights = self._v15c1_identity_weights(
                coarse_condition, target)
            identity_clean = torch.zeros_like(clean)
            identity_noisy = self.scheduler.add_noise(
                identity_clean, torch.randn_like(identity_clean), timesteps)
            identity_prediction = self.denoiser(
                identity_noisy, timesteps, reg_features, coarse_condition,
                level_ids, boundary_context)
            (v15c1_identity_loss,
             v15c1_duration_asym_loss) = self._v15c1_prior_losses(
                 clean, predicted_clean, identity_prediction,
                 timestep_weights, identity_weights)
        if self.final_ddim_loss_enabled:
            final_ddim_loss = self._final_ddim_training_loss(
                coarse_condition, target, reg_features, level_ids,
                boundary_context)
        total = (
            denoising_loss
            + self.boundary_loss_weight * boundary_loss
            + self.final_ddim_loss_weight * final_ddim_loss
            + self.quality_gate_loss_weight * quality_gate_loss
            + self.v15b_tiou_loss_weight * v15b_tiou_loss
            + self.v15b_endpoint_loss_weight * v15b_endpoint_loss
            + self.v15c1_identity_loss_weight * v15c1_identity_loss
            + self.v15c1_duration_asym_loss_weight
            * v15c1_duration_asym_loss
        )
        if return_v15c1_components:
            return (
                total, denoising_loss, boundary_loss, final_ddim_loss,
                quality_gate_loss, v15b_tiou_loss, v15b_endpoint_loss,
                v15c1_identity_loss, v15c1_duration_asym_loss)
        if return_v15b_components:
            return (
                total, denoising_loss, boundary_loss, final_ddim_loss,
                quality_gate_loss, v15b_tiou_loss, v15b_endpoint_loss)
        if return_components:
            return (total, denoising_loss, boundary_loss,
                    final_ddim_loss, quality_gate_loss)
        return total

    @torch.no_grad()
    def refine(
        self,
        coarse: Tensor,
        reg_features: Tensor,
        level_ids: Tensor,
        valid_range: Optional[Tuple[float, float]] = None,
        return_stats: bool = False,
        boundary_context: Optional[Tensor] = None,
        inference_seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        residual_blend_alpha: Optional[float] = None,
        max_relative_shift: Optional[float] = None,
    ):
        if coarse.shape[0] == 0:
            empty_stats = {
                "quality_gate_fallbacks": 0,
                "quality_gate_accepts": 0,
                "mean_coarse_quality": 0.0,
                "mean_refined_quality": 0.0,
                "mean_quality_advantage": 0.0,
            }
            return (coarse, 0, empty_stats) if return_stats else (coarse, 0)
        blend_alpha = (self.residual_blend_alpha
                       if residual_blend_alpha is None
                       else float(residual_blend_alpha))
        if not 0.0 <= blend_alpha <= 1.0:
            raise ValueError("residual_blend_alpha must be within [0, 1]")
        shift_limit = (self.max_relative_shift
                       if max_relative_shift is None
                       else float(max_relative_shift))
        if shift_limit is not None and not 0.0 < shift_limit <= 1.0:
            raise ValueError("max_relative_shift must be within (0, 1]")
        if blend_alpha == 0.0:
            identity_stats = {
                "quality_gate_fallbacks": 0,
                "quality_gate_accepts": 0,
                "mean_coarse_quality": 0.0,
                "mean_refined_quality": 0.0,
                "mean_quality_advantage": 0.0,
            }
            return ((coarse, 0, identity_stats)
                    if return_stats else (coarse, 0))
        generator = torch.Generator(device=coarse.device)
        generator.manual_seed(
            self.inference_seed if inference_seed is None
            else int(inference_seed))
        initial_noise = torch.randn(
            coarse.shape, device=coarse.device, dtype=coarse.dtype, generator=generator
        )
        residual = self.sampler.sample(
            self.denoiser, initial_noise, reg_features, coarse, level_ids,
            boundary_context, num_inference_steps=num_inference_steps
        )
        duration = (coarse[:, 1] - coarse[:, 0]).clamp_min(self.min_duration)
        blended_residual = blend_alpha * residual
        fallback = torch.zeros(
            coarse.shape[0], dtype=torch.bool, device=coarse.device)
        if shift_limit is not None:
            fallback |= (blended_residual.abs().amax(dim=-1)
                         > shift_limit)
        gate_fallback = torch.zeros_like(fallback)
        coarse_quality = torch.zeros_like(duration)
        refined_quality = torch.zeros_like(duration)
        if self.quality_gate_enabled:
            coarse_quality = torch.sigmoid(self.quality_gate(
                torch.zeros_like(blended_residual), reg_features, coarse,
                level_ids))
            refined_quality = torch.sigmoid(self.quality_gate(
                blended_residual, reg_features, coarse, level_ids))
            gate_fallback = (
                (refined_quality < self.quality_gate_threshold)
                | (refined_quality < coarse_quality + self.quality_gate_margin)
            )
            fallback |= gate_fallback
        blended_residual = torch.where(
            fallback[:, None], torch.zeros_like(blended_residual),
            blended_residual)
        refined = self.decode_state(coarse, blended_residual)
        refined, fixed = self.sanitize(refined, coarse, valid_range)
        stats = {
            "quality_gate_fallbacks": int(gate_fallback.sum().item()),
            "quality_gate_accepts": int((~gate_fallback).sum().item())
            if self.quality_gate_enabled else 0,
            "mean_coarse_quality": float(coarse_quality.mean().item())
            if self.quality_gate_enabled else 0.0,
            "mean_refined_quality": float(refined_quality.mean().item())
            if self.quality_gate_enabled else 0.0,
            "mean_quality_advantage": float(
                (refined_quality - coarse_quality).mean().item())
            if self.quality_gate_enabled else 0.0,
        }
        return (refined, fixed, stats) if return_stats else (refined, fixed)

    def sanitize(
        self, refined: Tensor, coarse: Tensor,
        valid_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Tensor, int]:
        invalid_numeric = ~torch.isfinite(refined).all(dim=-1)
        refined = torch.where(invalid_numeric[:, None], coarse, refined)
        invalid_bounds = torch.zeros_like(invalid_numeric)
        if self.clamp_boundary and valid_range is not None:
            low, high = valid_range
            invalid_bounds = ((refined[:, 0] < low) | (refined[:, 1] > high))
            refined = refined.clamp(float(low), float(high))
        start, end = refined.unbind(-1)
        invalid_duration = end - start < self.min_duration
        center = 0.5 * (start + end)
        start = torch.where(invalid_duration, center - 0.5 * self.min_duration, start)
        end = torch.where(invalid_duration, center + 0.5 * self.min_duration, end)
        if self.clamp_boundary and valid_range is not None:
            low, high = map(float, valid_range)
            start = start.clamp(low, max(low, high - self.min_duration))
            end = end.clamp(min(high, low + self.min_duration), high)
        fixed = torch.stack((start, end), dim=-1)
        count = int((invalid_numeric | invalid_bounds | invalid_duration).sum().item())
        return fixed, count
