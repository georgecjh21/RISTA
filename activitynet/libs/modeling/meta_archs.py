import math
import torch
from torch import nn
from torch.nn import functional as F

# from .blocks import get_module_activated_ratio
from .losses import ctr_diou_loss_1d, ctr_giou_loss_1d, sigmoid_focal_loss
from .models import register_meta_arch, make_backbone, make_neck, make_generator
from ..utils import batched_nms

from .heads import (
    A0PreservingResidualAlignmentHead,
    BoundaryAwareProposalGenerator,
    ClusterProposalSelectorHead,
    ClusterScoreSuppressionHead,
    DenseIoUQualityHead,
    PtTransformerClsHead,
    PtTransformerRegHead,
    RegressionSGPAdapter,
    StartEndDistributionHead,
    TDynPtTransformerClsHead,
    TDynPtTransformerRegHead,
    TridentRelativeBoundaryHead,
)
from .diffusion_boundary_refiner import DiffusionBoundaryRefiner
from .mamba_adapter import (
    MambaPyramidAdapter, SUPPORTED_MAMBA_POSITIONS)


@register_meta_arch("DyFADet")
class DyFADet(nn.Module):
    """
        DyFADet based TAD model for single stage action localization
    """
    def __init__(
            self,
            max_seq_len,  # max sequence length (used for training)
            max_buffer_len_factor,  # max buffer size (defined a factor of max_seq_len)
            
            input_dim,  # input feat dim
            backbone_type,  # a string defines which backbone we use
            backbone_arch,  # a tuple defines # layers in embed / stem / branch
            embd_kernel_size,  # kernel size of the embedding network
            embd_dim,  # output feat channel of the embedding network
            embd_with_ln,  # attach layernorm to embedding network
            scale_factor,  # scale factor between branch layers
            n_encoder_win_size,  # window size w for encooder
            mlp_dim,  # the numnber of dim in encoder
            use_abs_pe,  # if to use abs position encoding
            k,  # the K to generate large kernel
            
            fpn_type,  # a string defines which fpn we use
            fpn_dim,  # feature dim on FPN,
            fpn_with_ln,  # if to apply layer norm at the end of fpn

            regression_range,  # regression range on each level of FPN
            head_dim,  # feature dim for head
            head_num_layers,  # number of layers in the head (including the classifier)
            head_kernel_size,  # kernel size for reg/cls heads
            head_act, # act function for head
            dyn_head, # the dict of param for dyn_head 
            dyn_head_flag, # if use dyn_head

            input_noise,  # add gaussian noise with the variance, play a similar role to position embedding
            init_conv_vars,  # initialization of gaussian variance for the weight
            num_classes,  # number of action classes
            train_cfg,  # other cfg for training
            test_cfg,  # other cfg for testing
            diffusion_refiner=None,  # optional local boundary diffusion config
            regression_boundary_pyramid=None,
            start_end_distribution=None,
            trident_head=None,
            regression_sgp_adapter=None,
            temporal_consistency=None,
            high_iou_boundary=None,
            dense_iou_quality=None,
            cluster_selector=None,
            task_alignment=None,
            residual_alignment=None,
            boundary_proposal_generator=None,
            mamba_adapter=None,
            selective_dynamic_scale=None,
            mamba_task_routing=None,
            loss_cfg=None,
            training_cfg=None,
            ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor ** i for i in range(backbone_arch[-1] + 1)]

        self.input_noise = input_noise
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor

        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
            
        self.num_classes = num_classes    
        if head_act =='grelu':
            act_layer = nn.GELU
        elif head_act =='lrelu':
            act_layer = nn.LeakyReLU
        else:
            act_layer = nn.ReLU
        
        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_encoder_win_size, int):
            self.encoder_win_size = [n_encoder_win_size] * len(self.fpn_strides)
        else:
            assert len(n_encoder_win_size) == len(self.fpn_strides)
            self.encoder_win_size = n_encoder_win_size
        

        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.encoder_win_size)):
            stride = s * w if w > 1 else s
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor
        
    
        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

        self.mamba_cfg = mamba_adapter or {"enabled": False}
        self.selective_scale_cfg = (
            selective_dynamic_scale
            or {"enabled": False, "variant": "identity"})
        self.selective_scale_enabled = bool(
            self.selective_scale_cfg.get("enabled", False))
        self.selective_scale_variant = str(
            self.selective_scale_cfg.get("variant", "identity"))
        if self.selective_scale_variant not in {
                "identity", "cls_only", "reg_only", "cls_reg"}:
            raise ValueError(
                "unknown selective DynamicScale variant: {}".format(
                    self.selective_scale_variant))
        if self.selective_scale_enabled \
                and self.selective_scale_variant == "identity":
            raise ValueError(
                "enabled SelectiveDynamicScale requires a non-identity variant")
        if not self.selective_scale_enabled \
                and self.selective_scale_variant != "identity":
            raise ValueError(
                "disabled SelectiveDynamicScale must use identity variant")
        self.mamba_enabled = bool(self.mamba_cfg.get("enabled", False))
        self.mamba_task_cfg = (
            mamba_task_routing or {"enabled": False, "variant": "shared"})
        self.mamba_task_enabled = bool(
            self.mamba_task_cfg.get("enabled", False))
        self.mamba_task_variant = str(
            self.mamba_task_cfg.get("variant", "shared"))
        if self.mamba_task_variant not in {"shared", "mtg1", "td1"}:
            raise ValueError(
                "unknown Mamba task routing variant: {}".format(
                    self.mamba_task_variant))
        if self.mamba_task_enabled \
                and self.mamba_task_variant == "shared":
            raise ValueError(
                "enabled Mamba task routing requires mtg1 or td1")
        if not self.mamba_task_enabled \
                and self.mamba_task_variant != "shared":
            raise ValueError(
                "disabled Mamba task routing must use shared variant")
        if self.mamba_task_enabled and not self.mamba_enabled:
            raise ValueError("Mamba task routing requires an active Mamba")
        self.fusion_gradient_diagnostics = bool(
            self.mamba_task_cfg.get("gradient_diagnostics", False))
        self.mamba_position = str(
            self.mamba_cfg.get("position", "stem_branch"))
        if self.mamba_enabled \
                and self.mamba_position not in SUPPORTED_MAMBA_POSITIONS:
            raise ValueError(
                "unsupported Mamba position: {}".format(
                    self.mamba_position))

        self.diffusion_cfg = diffusion_refiner or {"enabled": False}
        self.diffusion_enabled = bool(self.diffusion_cfg.get("enabled", False))
        self.regression_boundary_cfg = (
            regression_boundary_pyramid or {"enabled": False})
        self.regression_boundary_enabled = bool(
            self.regression_boundary_cfg.get("enabled", False))
        self.start_end_distribution_cfg = (
            start_end_distribution or {"enabled": False})
        self.start_end_distribution_enabled = bool(
            self.start_end_distribution_cfg.get("enabled", False))
        self.start_end_distribution_weight = float(
            self.start_end_distribution_cfg.get(
                "distribution_loss_weight", 1.0))
        self.start_end_tiou_weight = float(
            self.start_end_distribution_cfg.get("tiou_loss_weight", 0.25))
        if self.start_end_distribution_weight < 0.0:
            raise ValueError("distribution_loss_weight must be non-negative")
        if self.start_end_tiou_weight < 0.0:
            raise ValueError("tiou_loss_weight must be non-negative")
        self.trident_cfg = trident_head or {"enabled": False}
        self.trident_enabled = bool(self.trident_cfg.get("enabled", False))
        self.trident_aux_weight = float(
            self.trident_cfg.get("aux_loss_weight", 1.0))
        self.trident_iou_weight_power = float(
            self.trident_cfg.get("iou_weight_power", 0.2))
        if self.trident_aux_weight < 0.0:
            raise ValueError("Trident aux_loss_weight must be non-negative")
        if self.trident_iou_weight_power < 0.0:
            raise ValueError("Trident iou_weight_power must be non-negative")
        if self.trident_enabled and self.start_end_distribution_enabled:
            raise ValueError(
                "Trident and V10-C start/end heads are mutually exclusive")
        self.boundary_proposal_cfg = (
            boundary_proposal_generator or {"enabled": False})
        self.boundary_proposal_enabled = bool(
            self.boundary_proposal_cfg.get("enabled", False))
        if self.boundary_proposal_enabled and (
                self.diffusion_enabled
                or self.start_end_distribution_enabled
                or self.trident_enabled):
            raise ValueError(
                "V13-C proposal generation must be trained independently "
                "from diffusion and other boundary heads")
        self.boundary_proposal_distribution_weight = float(
            self.boundary_proposal_cfg.get(
                "distribution_loss_weight", 1.0))
        self.boundary_proposal_coverage_weight = float(
            self.boundary_proposal_cfg.get("coverage_loss_weight", 1.0))
        self.boundary_proposal_improvement_weight = float(
            self.boundary_proposal_cfg.get(
                "improvement_loss_weight", 0.5))
        self.boundary_proposal_accept_threshold = float(
            self.boundary_proposal_cfg.get(
                "accept_threshold", 0.55))
        self.boundary_proposal_max_score_boost = float(
            self.boundary_proposal_cfg.get(
                "max_score_boost", 0.05))
        if not 0.0 < self.boundary_proposal_accept_threshold < 1.0:
            raise ValueError("proposal accept_threshold must be in (0, 1)")
        if self.boundary_proposal_max_score_boost < 0.0:
            raise ValueError("proposal max_score_boost must be non-negative")

        self.regression_sgp_cfg = (
            regression_sgp_adapter or {"enabled": False})
        self.regression_sgp_enabled = bool(
            self.regression_sgp_cfg.get("enabled", False))

        self.temporal_consistency_cfg = (
            temporal_consistency or {"enabled": False})
        self.temporal_consistency_enabled = bool(
            self.temporal_consistency_cfg.get("enabled", False))
        self.temporal_consistency_weight = float(
            self.temporal_consistency_cfg.get("loss_weight", 1.0))
        self.temporal_consistency_cls_weight = float(
            self.temporal_consistency_cfg.get(
                "classification_weight", 0.0))
        self.temporal_consistency_reg_weight = float(
            self.temporal_consistency_cfg.get("regression_weight", 1.0))
        self.temporal_consistency_drop_probability = float(
            self.temporal_consistency_cfg.get("drop_probability", 0.1))
        self.temporal_consistency_noise_std = float(
            self.temporal_consistency_cfg.get("noise_std", 0.01))
        self.temporal_consistency_teacher_detach = bool(
            self.temporal_consistency_cfg.get("teacher_detach", True))
        if self.temporal_consistency_weight < 0.0:
            raise ValueError("TRC loss_weight must be non-negative")
        if (self.temporal_consistency_cls_weight < 0.0
                or self.temporal_consistency_reg_weight < 0.0):
            raise ValueError("TRC component weights must be non-negative")
        if not 0.0 <= self.temporal_consistency_drop_probability < 1.0:
            raise ValueError("TRC drop_probability must be in [0, 1)")
        if self.temporal_consistency_noise_std < 0.0:
            raise ValueError("TRC noise_std must be non-negative")

        self.high_iou_boundary_cfg = (
            high_iou_boundary or {"enabled": False})
        self.high_iou_boundary_enabled = bool(
            self.high_iou_boundary_cfg.get("enabled", False))
        self.high_iou_thresholds = tuple(float(value) for value in
            self.high_iou_boundary_cfg.get("thresholds", (0.6, 0.7)))
        self.high_iou_margin_weight = float(
            self.high_iou_boundary_cfg.get("margin_loss_weight", 0.5))
        self.endpoint_precision_weight = float(
            self.high_iou_boundary_cfg.get("endpoint_loss_weight", 0.25))
        self.endpoint_precision_beta = float(
            self.high_iou_boundary_cfg.get("endpoint_beta", 0.05))
        if self.high_iou_boundary_enabled:
            if not self.high_iou_thresholds:
                raise ValueError(
                    "high-IoU boundary thresholds must not be empty")
            if any(not 0.0 < value < 1.0
                   for value in self.high_iou_thresholds):
                raise ValueError(
                    "high-IoU boundary thresholds must be within (0, 1)")
            if tuple(sorted(set(self.high_iou_thresholds))) != \
                    self.high_iou_thresholds:
                raise ValueError(
                    "high-IoU boundary thresholds must be unique and sorted")
            if (self.high_iou_margin_weight < 0.0
                    or self.endpoint_precision_weight < 0.0):
                raise ValueError(
                    "high-IoU boundary loss weights must be non-negative")
            if (self.high_iou_margin_weight == 0.0
                    and self.endpoint_precision_weight == 0.0):
                raise ValueError(
                    "high-IoU boundary requires a non-zero loss weight")
            if self.endpoint_precision_beta <= 0.0:
                raise ValueError(
                    "high-IoU endpoint beta must be positive")
        self.dense_iou_quality_cfg = (
            dense_iou_quality or {"enabled": False})
        self.dense_iou_quality_enabled = bool(
            self.dense_iou_quality_cfg.get("enabled", False))
        self.dense_iou_quality_weight = float(
            self.dense_iou_quality_cfg.get("quality_loss_weight", 1.0))
        self.dense_iou_calibration_weight = float(
            self.dense_iou_quality_cfg.get(
                "calibration_loss_weight", 0.25))
        self.dense_iou_quality_beta = float(
            self.dense_iou_quality_cfg.get("focal_beta", 2.0))
        self.dense_iou_proposal_target_enabled = bool(
            self.dense_iou_quality_cfg.get(
                "proposal_target_enabled", False))
        self.dense_iou_proposal_target_topk = int(
            self.dense_iou_quality_cfg.get(
                "proposal_target_topk", 2000))
        self.dense_iou_ranking_enabled = bool(
            self.dense_iou_quality_cfg.get("ranking_enabled", False))
        self.dense_iou_pairwise_weight = float(
            self.dense_iou_quality_cfg.get("pairwise_loss_weight", 1.0))
        self.dense_iou_listwise_weight = float(
            self.dense_iou_quality_cfg.get("listwise_loss_weight", 0.5))
        self.dense_iou_ranking_topk = int(
            self.dense_iou_quality_cfg.get("ranking_topk", 32))
        self.dense_iou_overlap_threshold = float(
            self.dense_iou_quality_cfg.get(
                "ranking_overlap_threshold", 0.3))
        self.dense_iou_pair_margin = float(
            self.dense_iou_quality_cfg.get("ranking_quality_margin", 0.02))
        self.dense_iou_pair_temperature = float(
            self.dense_iou_quality_cfg.get(
                "pairwise_temperature", 0.25))
        self.dense_iou_listwise_temperature = float(
            self.dense_iou_quality_cfg.get(
                "listwise_temperature", 0.25))
        self.dense_iou_target_temperature = float(
            self.dense_iou_quality_cfg.get(
                "listwise_target_temperature", 0.1))
        self.dense_iou_ranking_thresholds = tuple(
            float(value) for value in self.dense_iou_quality_cfg.get(
                "ranking_tiou_thresholds", (0.5, 0.6, 0.7)))
        self.dense_iou_proximity_sigma = float(
            self.dense_iou_quality_cfg.get("threshold_proximity_sigma", 0.05))
        self.dense_iou_proximity_gain = float(
            self.dense_iou_quality_cfg.get("threshold_proximity_gain", 2.0))
        self.dense_iou_crossing_gain = float(
            self.dense_iou_quality_cfg.get("threshold_crossing_gain", 2.0))
        self.dense_iou_ranking_source = str(
            self.dense_iou_quality_cfg.get(
                "ranking_prediction_source", "calibrated_logits"))
        self.dense_iou_safety_thresholds = tuple(
            float(value) for value in self.dense_iou_quality_cfg.get(
                "high_tiou_safety_thresholds", ()))
        self.dense_iou_safety_weight = float(
            self.dense_iou_quality_cfg.get(
                "high_tiou_harm_weight", 0.0))
        self.conservative_selection_enabled = bool(
            self.dense_iou_quality_cfg.get(
                "conservative_selection_enabled", False))
        self.conservative_selection_anchor_topk = int(
            self.dense_iou_quality_cfg.get(
                "selection_anchor_topk", 200))
        self.conservative_selection_candidate_topk = int(
            self.dense_iou_quality_cfg.get(
                "selection_candidate_topk", 512))
        self.conservative_selection_overlap = float(
            self.dense_iou_quality_cfg.get(
                "selection_overlap_threshold", 0.7))
        self.conservative_selection_min_gain = float(
            self.dense_iou_quality_cfg.get(
                "selection_min_quality_gain", 0.1))
        self.conservative_selection_min_score_ratio = float(
            self.dense_iou_quality_cfg.get(
                "selection_min_score_ratio", 0.1))
        if self.dense_iou_quality_enabled:
            if self.dense_iou_quality_weight < 0.0:
                raise ValueError(
                    "dense IoU quality loss weight must be non-negative")
            if self.dense_iou_calibration_weight < 0.0:
                raise ValueError(
                    "dense IoU calibration loss weight must be non-negative")
            if self.dense_iou_quality_beta < 0.0:
                raise ValueError(
                    "dense IoU quality focal beta must be non-negative")
            if (self.dense_iou_proposal_target_enabled
                    and self.dense_iou_proposal_target_topk <= 0):
                raise ValueError(
                    "proposal_target_topk must be positive")
            if self.dense_iou_ranking_enabled:
                if self.dense_iou_ranking_source not in (
                        "calibrated_logits", "quality_logits"):
                    raise ValueError(
                        "unsupported dense IoU ranking prediction source")
                if (self.dense_iou_pairwise_weight <= 0.0
                        or self.dense_iou_listwise_weight < 0.0):
                    raise ValueError(
                        "dense IoU ranking weights are invalid")
                if self.dense_iou_ranking_topk < 2:
                    raise ValueError(
                        "dense IoU ranking_topk must be at least two")
                if not 0.0 <= self.dense_iou_overlap_threshold <= 1.0:
                    raise ValueError(
                        "dense IoU overlap threshold must be in [0, 1]")
                if self.dense_iou_pair_margin < 0.0:
                    raise ValueError(
                        "dense IoU ranking margin must be non-negative")
                if (self.dense_iou_pair_temperature <= 0.0
                        or self.dense_iou_listwise_temperature <= 0.0
                        or self.dense_iou_target_temperature <= 0.0):
                    raise ValueError(
                        "dense IoU ranking temperatures must be positive")
                if (not self.dense_iou_ranking_thresholds
                        or any(not 0.0 < value < 1.0
                               for value
                               in self.dense_iou_ranking_thresholds)):
                    raise ValueError(
                        "dense IoU ranking thresholds must be in (0, 1)")
                if self.dense_iou_proximity_sigma <= 0.0:
                    raise ValueError(
                        "dense IoU proximity sigma must be positive")
                if (self.dense_iou_proximity_gain < 0.0
                        or self.dense_iou_crossing_gain < 0.0):
                    raise ValueError(
                        "dense IoU threshold gains must be non-negative")
                if (any(not 0.0 < value < 1.0
                        for value in self.dense_iou_safety_thresholds)
                        or tuple(sorted(set(
                            self.dense_iou_safety_thresholds))) !=
                        self.dense_iou_safety_thresholds):
                    raise ValueError(
                        "high-tIoU safety thresholds must be unique, "
                        "sorted, and within (0, 1)")
                if self.dense_iou_safety_weight < 0.0:
                    raise ValueError(
                        "high-tIoU harm weight must be non-negative")
            if self.conservative_selection_enabled:
                if not self.dense_iou_ranking_enabled:
                    raise ValueError(
                        "conservative selection requires ranking training")
                if self.dense_iou_ranking_source != "quality_logits":
                    raise ValueError(
                        "conservative selection must rank quality logits")
                if float(self.dense_iou_quality_cfg.get(
                        "score_fusion_beta", 0.0)) != 0.0:
                    raise ValueError(
                        "conservative selection must preserve A0 scores")
                if (self.conservative_selection_anchor_topk <= 0
                        or self.conservative_selection_candidate_topk <= 0):
                    raise ValueError(
                        "conservative selection top-k values must be positive")
                if (self.conservative_selection_candidate_topk
                        < self.conservative_selection_anchor_topk):
                    raise ValueError(
                        "candidate top-k must cover every anchor")
                if not 0.0 <= self.conservative_selection_overlap <= 1.0:
                    raise ValueError(
                        "selection overlap threshold must be in [0, 1]")
                if self.conservative_selection_min_gain < 0.0:
                    raise ValueError(
                        "selection quality gain must be non-negative")
                if not 0.0 <= \
                        self.conservative_selection_min_score_ratio <= 1.0:
                    raise ValueError(
                        "selection score ratio must be in [0, 1]")
            if (self.dense_iou_quality_weight == 0.0
                    and self.dense_iou_calibration_weight == 0.0
                    and not self.dense_iou_ranking_enabled):
                raise ValueError(
                    "dense IoU quality head requires a non-zero objective")
        self.cluster_selector_cfg = cluster_selector or {"enabled": False}
        self.cluster_selector_enabled = bool(
            self.cluster_selector_cfg.get("enabled", False))
        self.cluster_selector_type = str(
            self.cluster_selector_cfg.get(
                "selector_type", "pointwise_gain"))
        self.cluster_selection_enabled = bool(
            self.cluster_selector_cfg.get("cluster_enabled", True))
        self.cluster_replacement_enabled = bool(
            self.cluster_selector_cfg.get(
                "replacement_enabled", True))
        self.cluster_overlap_threshold = float(
            self.cluster_selector_cfg.get("cluster_threshold", 0.7))
        self.cluster_max_members = int(
            self.cluster_selector_cfg.get(
                "max_candidates_per_cluster", 16))
        self.cluster_max_train_clusters = int(
            self.cluster_selector_cfg.get(
                "max_train_clusters_per_video", 256))
        self.cluster_context_offsets = tuple(
            float(value) for value in self.cluster_selector_cfg.get(
                "context_offsets", (-2.0, -1.0, 0.0, 1.0, 2.0)))
        self.cluster_positive_gain_threshold = float(
            self.cluster_selector_cfg.get(
                "positive_gain_threshold", 0.01))
        self.cluster_min_predicted_gain = float(
            self.cluster_selector_cfg.get(
                "minimum_predicted_gain", 0.02))
        self.cluster_min_confidence = float(
            self.cluster_selector_cfg.get(
                "minimum_replace_confidence", 0.5))
        self.cluster_max_replacement_fraction = float(
            self.cluster_selector_cfg.get(
                "max_replacement_fraction", 0.01))
        self.cluster_gain_loss_weight = float(
            self.cluster_selector_cfg.get("gain_loss_weight", 1.0))
        self.cluster_pairwise_loss_weight = float(
            self.cluster_selector_cfg.get(
                "pairwise_loss_weight", 0.5))
        self.cluster_abstention_loss_weight = float(
            self.cluster_selector_cfg.get(
                "abstention_loss_weight", 1.0))
        self.cluster_pairwise_margin = float(
            self.cluster_selector_cfg.get("pairwise_margin", 0.01))
        self.cluster_pairwise_temperature = float(
            self.cluster_selector_cfg.get(
                "pairwise_temperature", 0.1))
        self.cluster_negative_weight = float(
            self.cluster_selector_cfg.get("negative_gain_weight", 2.0))
        self.cluster_safety_thresholds = tuple(
            float(value) for value in self.cluster_selector_cfg.get(
                "high_tiou_safety_thresholds", (0.85, 0.90, 0.95)))
        self.cluster_safety_weight = float(
            self.cluster_selector_cfg.get(
                "high_tiou_harm_weight", 4.0))
        self.cluster_ap_thresholds = tuple(
            float(value) for value in self.cluster_selector_cfg.get(
                "ap_tiou_thresholds",
                (0.50, 0.55, 0.60, 0.65, 0.70,
                 0.75, 0.80, 0.85, 0.90, 0.95)))
        self.cluster_listwise_temperature = float(
            self.cluster_selector_cfg.get(
                "listwise_temperature", 0.10))
        self.cluster_listwise_target_temperature = float(
            self.cluster_selector_cfg.get(
                "listwise_target_temperature", 0.10))
        self.cluster_ap_credit_sigma = float(
            self.cluster_selector_cfg.get("ap_credit_sigma", 0.025))
        self.cluster_score_suppression_enabled = bool(
            self.cluster_selector_cfg.get(
                "score_suppression_enabled", False))
        self.cluster_score_max_logit_suppression = float(
            self.cluster_selector_cfg.get(
                "maximum_score_logit_suppression", 1.0))
        self.cluster_score_max_suppression_fraction = float(
            self.cluster_selector_cfg.get(
                "max_score_suppression_fraction", 0.05))
        self.cluster_score_min_suppression = float(
            self.cluster_selector_cfg.get(
                "minimum_score_logit_suppression", 0.05))
        self.cluster_score_min_confidence = float(
            self.cluster_selector_cfg.get(
                "minimum_score_suppression_confidence", 0.5))
        self.cluster_score_positive_threshold = float(
            self.cluster_selector_cfg.get(
                "score_suppression_positive_threshold", 0.05))
        self.cluster_score_amount_loss_weight = float(
            self.cluster_selector_cfg.get(
                "score_suppression_amount_loss_weight", 1.0))
        self.cluster_score_confidence_loss_weight = float(
            self.cluster_selector_cfg.get(
                "score_suppression_confidence_loss_weight", 1.0))
        if self.cluster_selector_enabled:
            if self.num_classes != 1:
                raise ValueError(
                    "P2-D1 cluster selector currently requires "
                    "class-agnostic proposals")
            if (self.diffusion_enabled or self.boundary_proposal_enabled
                    or self.dense_iou_quality_enabled):
                raise ValueError(
                    "P2-D1 cluster selector must be isolated from other "
                    "proposal selectors/refiners")
            if not 0.0 < self.cluster_overlap_threshold <= 1.0:
                raise ValueError(
                    "cluster selector threshold must be within (0, 1]")
            if self.cluster_max_members < 2:
                raise ValueError(
                    "cluster selector max candidates must be at least two")
            if self.cluster_max_train_clusters <= 0:
                raise ValueError(
                    "cluster selector max train clusters must be positive")
            if len(self.cluster_context_offsets) < 3 \
                    or len(self.cluster_context_offsets) % 2 == 0:
                raise ValueError(
                    "cluster selector context offsets must have odd "
                    "length >= 3")
            if tuple(sorted(set(self.cluster_context_offsets))) != \
                    self.cluster_context_offsets:
                raise ValueError(
                    "cluster selector context offsets must be unique "
                    "and sorted")
            if self.cluster_positive_gain_threshold < 0.0 \
                    or self.cluster_min_predicted_gain < 0.0:
                raise ValueError(
                    "cluster selector gain thresholds must be non-negative")
            if not 0.0 <= self.cluster_min_confidence <= 1.0:
                raise ValueError(
                    "cluster selector confidence must be within [0, 1]")
            if not 0.0 <= self.cluster_max_replacement_fraction <= 1.0:
                raise ValueError(
                    "cluster selector replacement fraction must be "
                    "within [0, 1]")
            if min(
                    self.cluster_gain_loss_weight,
                    self.cluster_pairwise_loss_weight,
                    self.cluster_abstention_loss_weight) < 0.0:
                raise ValueError(
                    "cluster selector loss weights must be non-negative")
            if (self.cluster_gain_loss_weight
                    + self.cluster_pairwise_loss_weight
                    + self.cluster_abstention_loss_weight) <= 0.0:
                raise ValueError(
                    "cluster selector requires a non-zero objective")
            if self.cluster_pairwise_margin < 0.0 \
                    or self.cluster_pairwise_temperature <= 0.0:
                raise ValueError(
                    "cluster selector pairwise settings are invalid")
            if self.cluster_negative_weight < 1.0:
                raise ValueError(
                    "cluster selector negative weight must be >= 1")
            if any(not 0.0 < value < 1.0
                   for value in self.cluster_safety_thresholds):
                raise ValueError(
                    "cluster selector safety thresholds must be in (0, 1)")
            if tuple(sorted(set(self.cluster_safety_thresholds))) != \
                    self.cluster_safety_thresholds:
                raise ValueError(
                    "cluster selector safety thresholds must be unique "
                    "and sorted")
            if self.cluster_safety_weight < 0.0:
                raise ValueError(
                    "cluster selector safety weight must be non-negative")
            if self.cluster_selector_type not in (
                    "pointwise_gain", "ap_listwise"):
                raise ValueError(
                    "cluster selector_type must be pointwise_gain or "
                    "ap_listwise")
            if self.cluster_selector_type == "ap_listwise":
                expected_thresholds = tuple(
                    0.50 + 0.05 * index for index in range(10))
                if len(self.cluster_ap_thresholds) != 10 or any(
                        abs(actual - expected) > 1e-8
                        for actual, expected in zip(
                            self.cluster_ap_thresholds,
                            expected_thresholds)):
                    raise ValueError(
                        "P5 AP-listwise selector requires every official "
                        "ActivityNet threshold from 0.50 to 0.95")
                if self.cluster_listwise_temperature <= 0.0 \
                        or self.cluster_listwise_target_temperature <= 0.0:
                    raise ValueError(
                        "P5 listwise temperatures must be positive")
                if self.cluster_ap_credit_sigma <= 0.0:
                    raise ValueError(
                        "P5 AP credit sigma must be positive")
            if self.cluster_score_suppression_enabled:
                if self.cluster_score_max_logit_suppression <= 0.0:
                    raise ValueError(
                        "score suppression logit cap must be positive")
                if not 0.0 <= \
                        self.cluster_score_max_suppression_fraction <= 1.0:
                    raise ValueError(
                        "score suppression fraction must be in [0, 1]")
                if not 0.0 <= self.cluster_score_min_suppression <= \
                        self.cluster_score_max_logit_suppression:
                    raise ValueError(
                        "minimum score suppression must be within the cap")
                if not 0.0 <= self.cluster_score_min_confidence <= 1.0:
                    raise ValueError(
                        "score suppression confidence must be in [0, 1]")
                if not 0.0 <= self.cluster_score_positive_threshold <= \
                        self.cluster_score_max_logit_suppression:
                    raise ValueError(
                        "score suppression target threshold is invalid")
                if min(
                        self.cluster_score_amount_loss_weight,
                        self.cluster_score_confidence_loss_weight) < 0.0:
                    raise ValueError(
                        "score suppression loss weights must be non-negative")
                if (self.cluster_score_amount_loss_weight
                        + self.cluster_score_confidence_loss_weight) <= 0.0:
                    raise ValueError(
                        "score suppression requires a non-zero objective")
        self.task_alignment_cfg = (
            task_alignment or {"enabled": False})
        self.task_alignment_enabled = bool(
            self.task_alignment_cfg.get("enabled", False))
        self.task_alignment_alpha = float(
            self.task_alignment_cfg.get("alpha", 1.0))
        self.task_alignment_beta = float(
            self.task_alignment_cfg.get("beta", 2.0))
        self.task_alignment_soft_label_mix = float(
            self.task_alignment_cfg.get("soft_label_mix", 0.5))
        self.task_alignment_minimum_positive_weight = float(
            self.task_alignment_cfg.get(
                "minimum_positive_weight", 0.5))
        self.task_alignment_maximum_positive_weight = float(
            self.task_alignment_cfg.get(
                "maximum_positive_weight", 2.0))
        self.task_alignment_pairwise_weight = float(
            self.task_alignment_cfg.get("pairwise_loss_weight", 0.25))
        self.task_alignment_ranking_topk = int(
            self.task_alignment_cfg.get("ranking_topk", 32))
        self.task_alignment_overlap_threshold = float(
            self.task_alignment_cfg.get(
                "ranking_overlap_threshold", 0.3))
        self.task_alignment_quality_margin = float(
            self.task_alignment_cfg.get(
                "ranking_quality_margin", 0.05))
        self.task_alignment_pairwise_temperature = float(
            self.task_alignment_cfg.get(
                "pairwise_temperature", 0.25))
        self.task_alignment_thresholds = tuple(
            float(value) for value in self.task_alignment_cfg.get(
                "ranking_tiou_thresholds", (0.5, 0.6, 0.7)))
        self.task_alignment_proximity_sigma = float(
            self.task_alignment_cfg.get(
                "threshold_proximity_sigma", 0.05))
        self.task_alignment_proximity_gain = float(
            self.task_alignment_cfg.get(
                "threshold_proximity_gain", 2.0))
        if self.task_alignment_enabled:
            if self.task_alignment_alpha < 0.0:
                raise ValueError(
                    "task alignment alpha must be non-negative")
            if self.task_alignment_beta <= 0.0:
                raise ValueError(
                    "task alignment beta must be positive")
            if not 0.0 <= self.task_alignment_soft_label_mix <= 1.0:
                raise ValueError(
                    "task alignment soft_label_mix must be in [0, 1]")
            if not 0.0 < \
                    self.task_alignment_minimum_positive_weight <= \
                    self.task_alignment_maximum_positive_weight:
                raise ValueError(
                    "task alignment positive weights are invalid")
            if self.task_alignment_pairwise_weight < 0.0:
                raise ValueError(
                    "task alignment pairwise weight must be non-negative")
            if self.task_alignment_ranking_topk < 2:
                raise ValueError(
                    "task alignment ranking_topk must be at least two")
            if not 0.0 <= \
                    self.task_alignment_overlap_threshold <= 1.0:
                raise ValueError(
                    "task alignment overlap threshold must be in [0, 1]")
            if self.task_alignment_quality_margin < 0.0:
                raise ValueError(
                    "task alignment quality margin must be non-negative")
            if self.task_alignment_pairwise_temperature <= 0.0:
                raise ValueError(
                    "task alignment pairwise temperature must be positive")
            if (not self.task_alignment_thresholds
                    or any(not 0.0 < value < 1.0
                           for value in self.task_alignment_thresholds)):
                raise ValueError(
                    "task alignment thresholds must be in (0, 1)")
            if self.task_alignment_proximity_sigma <= 0.0:
                raise ValueError(
                    "task alignment proximity sigma must be positive")
            if self.task_alignment_proximity_gain < 0.0:
                raise ValueError(
                    "task alignment proximity gain must be non-negative")
        self.residual_alignment_cfg = (
            residual_alignment or {"enabled": False})
        self.residual_alignment_enabled = bool(
            self.residual_alignment_cfg.get("enabled", False))
        self.residual_alignment_weight = float(
            self.residual_alignment_cfg.get("residual_weight", 1.0))
        self.residual_alignment_cls_consistency_weight = float(
            self.residual_alignment_cfg.get(
                "classification_consistency_loss_weight", 1.0))
        self.residual_alignment_reg_consistency_weight = float(
            self.residual_alignment_cfg.get(
                "regression_consistency_loss_weight", 1.0))
        self.residual_alignment_magnitude_weight = float(
            self.residual_alignment_cfg.get(
                "residual_magnitude_loss_weight", 0.1))
        self.residual_alignment_confidence_weight = float(
            self.residual_alignment_cfg.get(
                "confidence_loss_weight", 0.5))
        self.residual_alignment_confidence_gain_margin = float(
            self.residual_alignment_cfg.get(
                "confidence_gain_margin", 0.005))
        self.residual_alignment_confidence_harm_margin = float(
            self.residual_alignment_cfg.get(
                "confidence_harm_margin", 0.005))
        if self.residual_alignment_enabled:
            if self.num_classes != 1:
                raise ValueError(
                    "P4-B residual alignment currently requires "
                    "class-agnostic ActivityNet proposals")
            if not 0.0 <= self.residual_alignment_weight <= 1.0:
                raise ValueError(
                    "residual alignment weight must be in [0, 1]")
            if min(
                    self.residual_alignment_cls_consistency_weight,
                    self.residual_alignment_reg_consistency_weight,
                    self.residual_alignment_magnitude_weight,
                    self.residual_alignment_confidence_weight,
                    self.residual_alignment_confidence_gain_margin,
                    self.residual_alignment_confidence_harm_margin) < 0.0:
                raise ValueError(
                    "residual alignment loss settings must be non-negative")
        self.diffusion_weight = float((loss_cfg or {}).get("diffusion_weight", 1.0))
        self.refine_topk = int(self.diffusion_cfg.get("refine_topk", 200))
        self.candidate_ranker_train_topk = int(
            self.diffusion_cfg.get("candidate_ranker_train_topk",
                                   self.refine_topk))
        # Diagnostic-only switch set explicitly by eval.py. Normal inference
        # must never consult ground truth.
        self.oracle_diagnostic_enabled = False
        self.oracle_num_candidates = 1
        self.oracle_candidate_profiles = []
        # V13 Phase A exposes the existing pre-NMS proposal pool and GT only
        # to the diagnostic evaluator. It does not alter normal inference.
        self.v13_phase_a_enabled = False
        # V14 Phase A exports a bounded, score-sorted pre-NMS proposal cache
        # for cross-fitted learnability probes. The switch is diagnostic-only
        # and is enabled explicitly by the standalone V14 runner.
        self.v14_probe_enabled = False
        self.v14_probe_max_proposals = 2000
        self.candidate_ranker_profile_discovery = False
        self.candidate_ranker_training_data = False
        
        
        self.dyn_head = dyn_head_flag
        self.dyn_type = dyn_head['dyn_type']
        if dyn_head:
            gate_activation_kargs = dyn_head

        if backbone_type == 'DynE':
            self.backbone = make_backbone(
                'DynE',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'mlp_dim': mlp_dim,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln': embd_with_ln,
                    'path_pdrop': self.train_droppath,
                    'encoder_win_size': self.encoder_win_size,
                    'use_abs_pe': use_abs_pe,
                    'k': k,
                    'init_conv_vars': init_conv_vars,
                    'mamba_adapter': self.mamba_cfg,
                }
            )
        else:
            print(backbone_type)
            raise TypeError("Unknown backbone type.")
            
        # fpn network: convs
        assert fpn_type in ['fpn', 'identity']
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels': [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel': fpn_dim,
                'scale_factor': scale_factor,
                'with_ln': fpn_with_ln
            }
        )
        self.mamba_neck_adapter = None
        if self.mamba_enabled and self.mamba_position == "neck_only":
            self.mamba_neck_adapter = MambaPyramidAdapter(
                fpn_dim, len(self.fpn_strides), self.mamba_cfg)
        self.mamba_cls_gate = None
        self.mamba_reg_gate = None
        self.mamba_task_gate_bound = float(
            self.mamba_task_cfg.get("gate_bound", 0.5))
        self.mamba_task_gate_lr_scale = float(
            self.mamba_task_cfg.get("gate_lr_scale", 1.0))
        if self.mamba_task_enabled and self.mamba_task_variant == "mtg1":
            if self.mamba_task_gate_bound <= 0.0:
                raise ValueError("Mamba task gate bound must be positive")
            if self.mamba_task_gate_lr_scale <= 0.0:
                raise ValueError("Mamba task gate LR scale must be positive")
            self.mamba_cls_gate = self._make_mamba_task_gate(
                float(self.mamba_task_cfg.get("cls_gate_init", 1.0)))
            self.mamba_reg_gate = self._make_mamba_task_gate(
                float(self.mamba_task_cfg.get("reg_gate_init", 1.0)))

        # location generator: points
        
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len': max_seq_len * max_buffer_len_factor,
                'fpn_levels': len(self.fpn_strides),
                'scale_factor': scale_factor,
                'regression_range': self.reg_range,
                'strides': self.fpn_strides
            }
        )
        
        if self.dyn_head:
            cls_selective_cfg = dict(self.selective_scale_cfg)
            cls_selective_cfg["enabled"] = (
                self.selective_scale_enabled
                and self.selective_scale_variant in {"cls_only", "cls_reg"})
            reg_selective_cfg = dict(self.selective_scale_cfg)
            reg_selective_cfg["enabled"] = (
                self.selective_scale_enabled
                and self.selective_scale_variant in {"reg_only", "cls_reg"})
            self.cls_head = TDynPtTransformerClsHead(
                fpn_dim, head_dim, self.num_classes,
                kernel_size=head_kernel_size,
                prior_prob=self.train_cls_prior_prob,
                num_layers=head_num_layers,
                act_layer = act_layer,
                empty_cls=train_cfg['head_empty_cls'],
                gate_activation_kargs=gate_activation_kargs,
                selective_dynamic_scale=cls_selective_cfg)
            
            self.reg_head = TDynPtTransformerRegHead(
                fpn_dim, head_dim, len(self.fpn_strides),
                kernel_size=head_kernel_size,
                num_layers=head_num_layers,
                gate_activation_kargs=gate_activation_kargs,
                selective_dynamic_scale=reg_selective_cfg)
            
        else:
            self.cls_head = PtTransformerClsHead(
                fpn_dim, head_dim, self.num_classes,
                kernel_size=head_kernel_size,
                prior_prob=self.train_cls_prior_prob,
                with_ln=True,
                num_layers=head_num_layers,
                empty_cls=train_cfg['head_empty_cls']
            )
            self.reg_head = PtTransformerRegHead(
                fpn_dim, head_dim, len(self.fpn_strides),
                kernel_size=head_kernel_size,
                num_layers=head_num_layers,
                with_ln=True
            )

        self.regression_sgp_adapter = None
        if self.regression_sgp_enabled:
            self.regression_sgp_adapter = RegressionSGPAdapter(
                feature_dim=fpn_dim,
                fpn_levels=len(self.fpn_strides),
                hidden_dim=int(self.regression_sgp_cfg.get(
                    "hidden_dim", 256)),
                kernel_size=int(self.regression_sgp_cfg.get(
                    "kernel_size", 3)),
                expansion=float(self.regression_sgp_cfg.get(
                    "expansion", 1.5)),
                dropout=float(self.regression_sgp_cfg.get(
                    "dropout", 0.0)),
                gate_init=float(self.regression_sgp_cfg.get(
                    "gate_init", 0.0)),
            )

        self.regression_boundary_pyramid = None
        if self.regression_boundary_enabled:
            self.regression_boundary_pyramid = make_neck(
                "regression_boundary",
                **{
                    "in_channels": [fpn_dim] * len(self.fpn_strides),
                    "out_channel": fpn_dim,
                    "scale_factor": scale_factor,
                    "with_ln": fpn_with_ln,
                    "hidden_dim": int(
                        self.regression_boundary_cfg.get("hidden_dim", 128)),
                    "dropout": float(
                        self.regression_boundary_cfg.get("dropout", 0.0)),
                    "gate_init": float(
                        self.regression_boundary_cfg.get("gate_init", 0.0)),
                }
            )

        self.start_end_distribution_head = None
        if self.start_end_distribution_enabled:
            self.start_end_distribution_head = StartEndDistributionHead(
                feature_dim=head_dim,
                fpn_levels=len(self.fpn_strides),
                hidden_dim=int(self.start_end_distribution_cfg.get(
                    "hidden_dim", 128)),
                num_bins=int(self.start_end_distribution_cfg.get(
                    "num_bins", 17)),
                max_residual=float(self.start_end_distribution_cfg.get(
                    "max_residual", 2.0)),
                dropout=float(self.start_end_distribution_cfg.get(
                    "dropout", 0.0)),
                gate_init=float(self.start_end_distribution_cfg.get(
                    "gate_init", 0.0)),
            )

        self.boundary_proposal_generator = None
        if self.boundary_proposal_enabled:
            self.boundary_proposal_generator = \
                BoundaryAwareProposalGenerator(
                    feature_dim=head_dim,
                    fpn_levels=len(self.fpn_strides),
                    hidden_dim=int(self.boundary_proposal_cfg.get(
                        "hidden_dim", 128)),
                    num_bins=int(self.boundary_proposal_cfg.get(
                        "num_bins", 17)),
                    max_residual=float(self.boundary_proposal_cfg.get(
                        "max_residual", 4.0)),
                    num_candidates=int(self.boundary_proposal_cfg.get(
                        "num_candidates", 4)),
                    dropout=float(self.boundary_proposal_cfg.get(
                        "dropout", 0.0)),
                    gate_init=float(self.boundary_proposal_cfg.get(
                        "gate_init", 0.0)),
                )

        self.trident_head = None
        if self.trident_enabled:
            self.trident_head = TridentRelativeBoundaryHead(
                boundary_feature_dim=fpn_dim,
                regression_feature_dim=head_dim,
                num_classes=self.num_classes,
                fpn_levels=len(self.fpn_strides),
                hidden_dim=int(self.trident_cfg.get(
                    "hidden_dim", 128)),
                num_bins=int(self.trident_cfg.get("num_bins", 16)),
                num_layers=int(self.trident_cfg.get(
                    "num_layers", 2)),
                kernel_size=int(self.trident_cfg.get(
                    "kernel_size", 3)),
                dropout=float(self.trident_cfg.get("dropout", 0.0)),
                prior_prob=float(self.trident_cfg.get(
                    "prior_prob", self.train_cls_prior_prob)),
                detach_boundary_features=bool(self.trident_cfg.get(
                    "detach_boundary_features", True)),
                gate_init=float(self.trident_cfg.get("gate_init", 0.0)),
            )

        self.dense_iou_quality_head = None
        if self.dense_iou_quality_enabled:
            self.dense_iou_quality_head = DenseIoUQualityHead(
                feature_dim=head_dim,
                num_classes=self.num_classes,
                fpn_levels=len(self.fpn_strides),
                hidden_dim=int(self.dense_iou_quality_cfg.get(
                    "hidden_dim", 128)),
                num_layers=int(self.dense_iou_quality_cfg.get(
                    "num_layers", 2)),
                kernel_size=int(self.dense_iou_quality_cfg.get(
                    "kernel_size", 3)),
                dropout=float(self.dense_iou_quality_cfg.get(
                    "dropout", 0.0)),
                prior_prob=float(self.dense_iou_quality_cfg.get(
                    "prior_prob", self.train_cls_prior_prob)),
                gate_init=float(self.dense_iou_quality_cfg.get(
                    "gate_init", 0.0)),
                max_logit_shift=float(self.dense_iou_quality_cfg.get(
                    "max_logit_shift", 0.5)),
                score_fusion=str(self.dense_iou_quality_cfg.get(
                    "score_fusion", "bounded_logit_residual")),
                score_fusion_beta=float(
                    self.dense_iou_quality_cfg.get(
                        "score_fusion_beta", 0.0)),
            )

        self.residual_alignment_head = None
        if self.residual_alignment_enabled:
            self.residual_alignment_head = \
                A0PreservingResidualAlignmentHead(
                    feature_dim=head_dim,
                    num_classes=self.num_classes,
                    fpn_levels=len(self.fpn_strides),
                    hidden_dim=int(self.residual_alignment_cfg.get(
                        "hidden_dim", 128)),
                    num_layers=int(self.residual_alignment_cfg.get(
                        "num_layers", 2)),
                    kernel_size=int(self.residual_alignment_cfg.get(
                        "kernel_size", 3)),
                    dropout=float(self.residual_alignment_cfg.get(
                        "dropout", 0.1)),
                    maximum_class_logit_residual=float(
                        self.residual_alignment_cfg.get(
                            "maximum_class_logit_residual", 0.5)),
                    maximum_regression_residual=float(
                        self.residual_alignment_cfg.get(
                            "maximum_regression_residual", 0.5)),
                    residual_weight=self.residual_alignment_weight,
                    minimum_confidence=float(
                        self.residual_alignment_cfg.get(
                            "minimum_confidence", 0.5)),
                    confidence_prior=float(
                        self.residual_alignment_cfg.get(
                            "confidence_prior", 0.1)),
                )

        self.cluster_selector_head = None
        self.cluster_score_suppression_head = None
        if self.cluster_selector_enabled:
            self.cluster_selector_head = ClusterProposalSelectorHead(
                feature_dim=head_dim,
                fpn_levels=len(self.fpn_strides),
                hidden_dim=int(self.cluster_selector_cfg.get(
                    "hidden_dim", 128)),
                level_embedding_dim=int(
                    self.cluster_selector_cfg.get(
                        "level_embedding_dim", 16)),
                geometry_dim=8,
                dropout=float(
                    self.cluster_selector_cfg.get("dropout", 0.1)),
                maximum_gain=float(
                    self.cluster_selector_cfg.get(
                        "maximum_gain", 1.0)),
                replacement_prior=float(
                    self.cluster_selector_cfg.get(
                        "replacement_prior", 0.01)),
                ap_listwise=(
                    self.cluster_selector_type == "ap_listwise"),
            )
            if self.cluster_score_suppression_enabled:
                self.cluster_score_suppression_head = \
                    ClusterScoreSuppressionHead(
                        feature_dim=head_dim,
                        fpn_levels=len(self.fpn_strides),
                        hidden_dim=int(self.cluster_selector_cfg.get(
                            "score_suppression_hidden_dim", 128)),
                        level_embedding_dim=int(
                            self.cluster_selector_cfg.get(
                                "score_suppression_level_embedding_dim",
                                16)),
                        dropout=float(self.cluster_selector_cfg.get(
                            "score_suppression_dropout", 0.1)),
                        maximum_logit_suppression=
                        self.cluster_score_max_logit_suppression,
                        suppression_prior=float(
                            self.cluster_selector_cfg.get(
                                "score_suppression_prior", 0.05)),
                    )

        self.diffusion_refiner = None
        if self.diffusion_enabled:
            self.diffusion_refiner = DiffusionBoundaryRefiner(
                head_dim, self.diffusion_cfg, len(self.fpn_strides)
            )

        self._frozen_modules = []
        self._configure_trainable_parameters(training_cfg or {})
 
        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

    def _configure_trainable_parameters(self, cfg):
        """Apply experiment freeze policy before optimizer construction."""
        refiner_only = bool(cfg.get("refiner_only", False))
        if bool(cfg.get("freeze_backbone", refiner_only)):
            for module in (self.backbone.embd, self.backbone.embd_norm):
                self._frozen_modules.append(module)
                for parameter in module.parameters():
                    parameter.requires_grad = False
        if bool(cfg.get("freeze_encoder", refiner_only)):
            encoder_modules = [
                self.backbone.stem, self.backbone.branch,
                self.backbone.mamba_stem, self.backbone.mamba_branch,
                self.neck, self.mamba_neck_adapter]
            if self.regression_boundary_pyramid is not None:
                encoder_modules.append(self.regression_boundary_pyramid)
            for module in encoder_modules:
                if module is None:
                    continue
                self._frozen_modules.append(module)
                for parameter in module.parameters():
                    parameter.requires_grad = False
        for name, key in (("cls_head", "freeze_cls_branch"),
                          ("reg_head", "freeze_reg_branch")):
            if bool(cfg.get(key, refiner_only)):
                self._frozen_modules.append(getattr(self, name))
                for parameter in getattr(self, name).parameters():
                    parameter.requires_grad = False
        for module, key in (
                (self.trident_head, "freeze_trident_head"),
                (self.regression_sgp_adapter,
                 "freeze_regression_sgp_adapter"),
                (self.dense_iou_quality_head,
                 "freeze_dense_iou_quality_head"),
                (self.residual_alignment_head,
                 "freeze_residual_alignment_head"),
                (self.cluster_selector_head,
                 "freeze_cluster_selector_head"),
                (self.cluster_score_suppression_head,
                 "freeze_cluster_score_suppression_head")):
            if module is not None and bool(cfg.get(key, refiner_only)):
                self._frozen_modules.append(module)
                for parameter in module.parameters():
                    parameter.requires_grad = False
        if self.diffusion_enabled and bool(cfg.get("quality_gate_only", False)):
            if not self.diffusion_refiner.quality_gate_enabled:
                raise ValueError(
                    "quality_gate_only requires quality_gate_enabled=true")
            for name, parameter in self.diffusion_refiner.named_parameters():
                parameter.requires_grad = name.startswith("quality_gate.")
        if self.diffusion_enabled and bool(
                cfg.get("candidate_ranker_only", False)):
            if not self.diffusion_refiner.candidate_ranker_enabled:
                raise ValueError(
                    "candidate_ranker_only requires candidate_ranker_enabled=true")
            for name, parameter in self.diffusion_refiner.named_parameters():
                train_ranker = name.startswith("candidate_ranker.")
                if (self.diffusion_refiner.candidate_ranker_quality_enabled
                        and name.startswith("candidate_ranker.output.")):
                    train_ranker = False
                parameter.requires_grad = train_ranker
        groups = (
            ("Feature Embedding", (self.backbone.embd, self.backbone.embd_norm)),
            ("DynE Encoder", (self.backbone.stem, self.backbone.branch)),
            ("P7 Mamba Stem Adapter", (self.backbone.mamba_stem,)),
            ("P7 Mamba Branch Adapter", (self.backbone.mamba_branch,)),
            ("Feature Pyramid", (self.neck,)),
            ("P7 Mamba Neck Adapter", (self.mamba_neck_adapter,)),
            ("Regression Boundary Pyramid",
             (self.regression_boundary_pyramid,)),
            ("Start/End Distribution Head",
             (self.start_end_distribution_head,)),
            ("V13-C Boundary Proposal Generator",
             (self.boundary_proposal_generator,)),
            ("V11 Trident Relative Boundary Head", (self.trident_head,)),
            ("V11 Regression SGP Adapter",
             (self.regression_sgp_adapter,)),
            ("V12 Dense IoU Quality Head",
             (self.dense_iou_quality_head,)),
            ("P4-B A0-Preserving Residual Alignment",
             (self.residual_alignment_head,)),
            ("P2-D1 Cluster Proposal Selector",
             (self.cluster_selector_head,)),
            ("P4-A Cluster Score Suppression",
             (self.cluster_score_suppression_head,)),
            ("Classification DyHead + Cls Module", (self.cls_head,)),
            ("Regression DyHead + Reg Module", (self.reg_head,)),
            ("DiffusionBoundaryRefiner", (self.diffusion_refiner,)),
        )
        trainable_groups, frozen_groups = [], []
        for label, modules in groups:
            parameters = [p for module in modules if module is not None
                          for p in module.parameters()]
            if not parameters:
                continue
            count = sum(p.numel() for p in parameters if p.requires_grad)
            frozen = sum(p.numel() for p in parameters if not p.requires_grad)
            entry = "{} ({:,} parameters)".format(label, count or frozen)
            (trainable_groups if count else frozen_groups).append(entry)
        print("Trainable parameters:\n- " + "\n- ".join(trainable_groups))
        if frozen_groups:
            print("Frozen parameters:\n- " + "\n- ".join(frozen_groups))
        trainable_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_total = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable_total + frozen_total
        self.parameter_report = {
            "trainable": trainable_total, "frozen": frozen_total, "total": total,
            "trainable_groups": tuple(trainable_groups),
            "frozen_groups": tuple(frozen_groups),
        }
        print("Parameter totals: trainable={:,}, frozen={:,}, total={:,}".format(
            trainable_total, frozen_total, total))

    def train(self, mode=True):
        super().train(mode)
        if mode:
            # Frozen baseline modules must not update dropout/drop-path behavior.
            for module in getattr(self, "_frozen_modules", []):
                module.eval()
        return self

    def _sample_boundary_context(
            self, feature_maps, feature_masks, boundaries,
            batch_ids, level_ids, offset_values=None):
        """Linearly sample start/end evidence on each proposal's FPN level.

        Boundaries use the base feature-grid coordinate system.  The offsets
        are expressed in cells of the proposal's own FPN level, so dividing
        by that level's stride gives the sampling coordinate directly.
        """
        if offset_values is None:
            if not self.diffusion_refiner.boundary_context_enabled:
                return None
            offset_values = self.diffusion_refiner.boundary_context_offsets
        offsets = boundaries.new_tensor(offset_values)
        num_samples = int(offsets.numel())
        feature_dim = feature_maps[0].shape[-1]
        context = feature_maps[0].new_zeros(
            (boundaries.shape[0], 2, num_samples, feature_dim))
        if boundaries.shape[0] == 0:
            return context
        if batch_ids.shape != (boundaries.shape[0],):
            raise ValueError("batch_ids must contain one id per boundary")
        if level_ids.shape != (boundaries.shape[0],):
            raise ValueError("level_ids must contain one id per boundary")

        for level, (features, masks, stride) in enumerate(zip(
                feature_maps, feature_masks, self.fpn_strides)):
            selected = torch.nonzero(level_ids == level, as_tuple=True)[0]
            if selected.numel() == 0:
                continue
            selected_batches = batch_ids[selected].long()
            level_features = features[selected_batches]
            level_masks = masks[selected_batches]
            max_positions = (
                level_masks.long().sum(dim=-1).clamp_min(1) - 1)
            positions = (
                boundaries[selected].detach()[:, :, None] / float(stride)
                + offsets[None, None, :])
            positions = positions.clamp_min(0.0)
            positions = torch.minimum(
                positions, max_positions[:, None, None].to(positions.dtype))
            low = positions.floor().long()
            high = torch.minimum(low + 1, max_positions[:, None, None])
            flat_low = low.flatten(1)
            flat_high = high.flatten(1)
            gather_shape = (-1, -1, feature_dim)
            low_features = torch.gather(
                level_features, 1,
                flat_low[:, :, None].expand(*gather_shape))
            high_features = torch.gather(
                level_features, 1,
                flat_high[:, :, None].expand(*gather_shape))
            weight = (positions - low.to(positions.dtype)).flatten(1)
            sampled = (
                low_features * (1.0 - weight[:, :, None])
                + high_features * weight[:, :, None])
            context[selected] = sampled.reshape(
                -1, 2, num_samples, feature_dim)
        return context

    @staticmethod
    def _boundary_contrast_descriptor(context):
        """Compress odd symmetric boundary samples into start/end jumps."""
        if context.ndim != 4 or context.shape[1] != 2:
            raise ValueError("context must have shape [N, 2, S, C]")
        if context.shape[2] < 3 or context.shape[2] % 2 == 0:
            raise ValueError("context sample count must be odd and >= 3")
        midpoint = context.shape[2] // 2
        left = context[:, :, :midpoint].mean(dim=2)
        right = context[:, :, midpoint + 1:].mean(dim=2)
        return (right - left).flatten(1)

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def _make_mamba_task_gate(self, effective_init):
        offset = (float(effective_init) - 1.0) / self.mamba_task_gate_bound
        if not -1.0 < offset < 1.0:
            raise ValueError(
                "Mamba task gate init must lie strictly inside "
                "[1-bound, 1+bound]")
        parameter = nn.Parameter(torch.tensor(
            math.atanh(offset), dtype=torch.float32))
        parameter._no_weight_decay = True
        parameter._lr_scale = self.mamba_task_gate_lr_scale
        if self.mamba_task_gate_lr_scale != 1.0:
            parameter.register_hook(
                lambda gradient: gradient * self.mamba_task_gate_lr_scale)
        return parameter

    def effective_mamba_task_gate(self, task):
        parameter = (self.mamba_cls_gate if task == "cls"
                     else self.mamba_reg_gate)
        if parameter is None:
            return None
        return 1.0 + self.mamba_task_gate_bound * torch.tanh(parameter)

    def _route_mamba_features(self, baseline_feats, mamba_feats):
        if not self.mamba_task_enabled:
            return mamba_feats, mamba_feats
        if self.mamba_task_variant == "td1":
            return baseline_feats, mamba_feats
        cls_gate = self.effective_mamba_task_gate("cls")
        reg_gate = self.effective_mamba_task_gate("reg")
        cls_feats = tuple(
            mamba + (cls_gate - 1.0) * (mamba - baseline)
            for baseline, mamba in zip(baseline_feats, mamba_feats))
        reg_feats = tuple(
            mamba + (reg_gate - 1.0) * (mamba - baseline)
            for baseline, mamba in zip(baseline_feats, mamba_feats))
        return cls_feats, reg_feats

    def fusion_diagnostic_report(self):
        report = {
            "task_routing_variant": self.mamba_task_variant,
            "mamba_task_gates": {},
            "selective_modules": {},
        }
        for task in ("cls", "reg"):
            gate = self.effective_mamba_task_gate(task)
            if gate is not None:
                raw = (self.mamba_cls_gate if task == "cls"
                       else self.mamba_reg_gate)
                report["mamba_task_gates"][task] = {
                    "raw": float(raw.detach().cpu()),
                    "effective": float(gate.detach().cpu()),
                    "lr_scale": self.mamba_task_gate_lr_scale,
                }
        for name, module in self.named_modules():
            if hasattr(module, "diagnostic_report") \
                    and module.__class__.__name__ == "SelectiveDynamicScale":
                report["selective_modules"][name] = \
                    module.diagnostic_report()
        return report

    def _perturb_temporal_features(self, inputs, masks):
        """Apply training-only frame erasure and weak Gaussian corruption."""
        valid = masks.to(inputs.dtype)
        perturbed = inputs
        drop_probability = self.temporal_consistency_drop_probability
        if drop_probability > 0.0:
            keep = (
                torch.rand(
                    inputs.shape[0], 1, inputs.shape[-1],
                    device=inputs.device)
                >= drop_probability
            ).to(inputs.dtype)
            perturbed = perturbed * keep
        if self.temporal_consistency_noise_std > 0.0:
            perturbed = perturbed + (
                torch.randn_like(perturbed)
                * self.temporal_consistency_noise_std)
        return perturbed * valid

    def _forward_consistency_predictions(self, inputs, masks):
        """Run the detector branches needed by V11-C on perturbed features."""
        feats, branch_masks = self.backbone(inputs, masks)
        baseline_fpn_feats, fpn_masks = self.neck(feats, branch_masks)
        mamba_fpn_feats = baseline_fpn_feats
        if self.mamba_neck_adapter is not None:
            mamba_fpn_feats, fpn_masks = self.mamba_neck_adapter(
                baseline_fpn_feats, fpn_masks)
        fpn_feats, routed_regression_feats = self._route_mamba_features(
            baseline_fpn_feats, mamba_fpn_feats)
        if self.regression_boundary_pyramid is not None:
            regression_feats, _ = self.regression_boundary_pyramid(
                routed_regression_feats, fpn_masks)
        else:
            regression_feats = routed_regression_feats
        if self.regression_sgp_adapter is not None:
            regression_feats = self.regression_sgp_adapter(
                regression_feats, fpn_masks)

        cls_logits = self.cls_head(fpn_feats, fpn_masks)
        need_features = (
            self.trident_enabled or self.start_end_distribution_enabled)
        if need_features:
            offsets, reg_features = self.reg_head(
                regression_feats, fpn_masks, return_features=True)
        else:
            offsets = self.reg_head(regression_feats, fpn_masks)
            reg_features = None
        if self.start_end_distribution_head is not None:
            offsets, _ = self.start_end_distribution_head(
                reg_features, fpn_masks, offsets)
        if self.trident_head is not None:
            offsets, _, _, _, _ = self.trident_head(
                fpn_feats, reg_features, fpn_masks, offsets)

        cls_logits = [
            item.permute(0, 2, 1) for item in cls_logits]
        if self.trident_head is None:
            offsets = [item.permute(0, 2, 1) for item in offsets]
        else:
            offsets = list(offsets)
        return cls_logits, offsets

    def forward(self, video_list):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        if self.training:
            video_list = [each for each in video_list if each['segments'] is not None]
        batched_inputs, batched_masks = self.preprocessing(video_list)

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        baseline_fpn_feats, fpn_masks = self.neck(feats, masks)
        mamba_fpn_feats = baseline_fpn_feats
        if self.mamba_neck_adapter is not None:
            mamba_fpn_feats, fpn_masks = self.mamba_neck_adapter(
                baseline_fpn_feats, fpn_masks)
        fpn_feats, routed_regression_feats = self._route_mamba_features(
            baseline_fpn_feats, mamba_fpn_feats)
        if self.regression_boundary_pyramid is not None:
            regression_feats, _ = self.regression_boundary_pyramid(
                routed_regression_feats, fpn_masks)
        else:
            regression_feats = routed_regression_feats
        if self.regression_sgp_adapter is not None:
            regression_feats = self.regression_sgp_adapter(
                regression_feats, fpn_masks)

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        
        points = self.point_generator(fpn_feats)

        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        need_reg_features = (
            self.diffusion_enabled or self.start_end_distribution_enabled
            or self.trident_enabled or self.dense_iou_quality_enabled
            or self.boundary_proposal_enabled or self.v14_probe_enabled
            or self.cluster_selector_enabled
            or self.residual_alignment_enabled)
        if need_reg_features:
            out_offsets, reg_features = self.reg_head(
                regression_feats, fpn_masks, return_features=True
            )
        else:
            out_offsets = self.reg_head(regression_feats, fpn_masks)
            reg_features = None
        a0_cls_logits = out_cls_logits
        a0_offsets = out_offsets
        residual_cls_values = None
        residual_reg_values = None
        residual_cls_confidence = None
        residual_reg_confidence = None
        if self.residual_alignment_head is not None:
            (out_cls_logits, out_offsets,
             residual_cls_values, residual_reg_values,
             residual_cls_confidence, residual_reg_confidence) = \
                self.residual_alignment_head(
                    reg_features, fpn_masks,
                    a0_cls_logits, a0_offsets,
                    apply_confidence=not self.training)
        coarse_offsets = out_offsets
        distribution_logits = None
        trident_candidates = None
        boundary_proposal_candidates = None
        boundary_proposal_logits = None
        boundary_proposal_improvement = None
        if self.start_end_distribution_head is not None:
            out_offsets, distribution_logits = \
                self.start_end_distribution_head(
                    reg_features, fpn_masks, coarse_offsets)
        if self.trident_head is not None:
            (out_offsets, trident_candidates, _, _, _) = self.trident_head(
                fpn_feats, reg_features, fpn_masks, coarse_offsets)
        if self.boundary_proposal_generator is not None:
            (boundary_proposal_candidates, boundary_proposal_logits,
             boundary_proposal_improvement) = \
                self.boundary_proposal_generator(
                    reg_features, fpn_masks, coarse_offsets)

        quality_logits = None
        calibrated_cls_logits = out_cls_logits
        if self.dense_iou_quality_head is not None:
            quality_logits = self.dense_iou_quality_head(
                reg_features, fpn_masks, out_offsets)
            calibrated_cls_logits = \
                self.dense_iou_quality_head.calibrate_logits(
                    out_cls_logits, quality_logits)

        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        a0_cls_logits = [x.permute(0, 2, 1) for x in a0_cls_logits]
        calibrated_cls_logits = [
            x.permute(0, 2, 1) for x in calibrated_cls_logits]
        if quality_logits is not None:
            quality_logits = [
                x.permute(0, 2, 1) for x in quality_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        if self.trident_head is None:
            out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        else:
            out_offsets = list(out_offsets)
            trident_candidates = list(trident_candidates)
        coarse_offsets = [x.permute(0, 2, 1) for x in coarse_offsets]
        a0_offsets = [x.permute(0, 2, 1) for x in a0_offsets]
        if residual_cls_values is not None:
            residual_cls_values = [
                x.permute(0, 2, 1) for x in residual_cls_values]
            residual_reg_values = [
                x.permute(0, 2, 1) for x in residual_reg_values]
            residual_cls_confidence = [
                x.permute(0, 2, 1)
                for x in residual_cls_confidence]
            residual_reg_confidence = [
                x.permute(0, 2, 1)
                for x in residual_reg_confidence]
        if distribution_logits is not None:
            distribution_logits = [
                x.permute(0, 3, 1, 2) for x in distribution_logits]
        if boundary_proposal_logits is not None:
            boundary_proposal_logits = [
                x.permute(0, 3, 1, 2)
                for x in boundary_proposal_logits]
        if reg_features is not None:
            reg_features = [x.permute(0, 2, 1) for x in reg_features]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self.label_points(
                points, gt_segments, gt_labels)

            augmented_cls_logits = None
            augmented_offsets = None
            if self.temporal_consistency_enabled:
                perturbed_inputs = self._perturb_temporal_features(
                    batched_inputs, batched_masks)
                augmented_cls_logits, augmented_offsets = \
                    self._forward_consistency_predictions(
                        perturbed_inputs, batched_masks)

            # compute the loss and return
            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets,
                points=points, reg_features=reg_features,
                coarse_offsets=coarse_offsets,
                distribution_logits=distribution_logits,
                boundary_proposal_candidates=boundary_proposal_candidates,
                boundary_proposal_logits=boundary_proposal_logits,
                boundary_proposal_improvement=boundary_proposal_improvement,
                trident_candidates=trident_candidates,
                quality_logits=quality_logits,
                calibrated_cls_logits=calibrated_cls_logits,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                augmented_cls_logits=augmented_cls_logits,
                augmented_offsets=augmented_offsets,
                a0_cls_logits=a0_cls_logits,
                a0_offsets=a0_offsets,
                residual_cls_values=residual_cls_values,
                residual_reg_values=residual_reg_values,
                residual_cls_confidence=residual_cls_confidence,
                residual_reg_confidence=residual_reg_confidence,
            )
            return losses

        else:
            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, points, fpn_masks,
                calibrated_cls_logits, out_offsets, reg_features,
                boundary_proposal_candidates,
                boundary_proposal_improvement,
                quality_logits,
                residual_cls_values,
                residual_reg_values,
                residual_cls_confidence,
                residual_reg_confidence,
            )
            return results

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
            if self.input_noise > 0:
                # trick, adding noise slightly increases the variability between input features.
                noise = torch.randn_like(batched_inputs) * self.input_noise
                batched_inputs += noise
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            # print(max_len)
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset = [], []

        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label
            )
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regressoin range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0

        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask == 0, float('inf'))
        lens.masked_fill_(inside_regress_range == 0, float('inf'))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(
            gt_label, self.num_classes
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    @staticmethod
    def _segment_tiou(first, second):
        """Pairwise temporal IoU for decoded ``[start, end]`` segments."""
        if first.ndim != 2 or first.shape[-1] != 2:
            raise ValueError("first segments must have shape [N, 2]")
        if second.ndim != 2 or second.shape[-1] != 2:
            raise ValueError("second segments must have shape [M, 2]")
        if first.shape[0] == 0 or second.shape[0] == 0:
            return first.new_zeros((first.shape[0], second.shape[0]))
        intersection = (
            torch.minimum(first[:, None, 1], second[None, :, 1])
            - torch.maximum(first[:, None, 0], second[None, :, 0])
        ).clamp_min(0.0)
        first_duration = (
            first[:, 1] - first[:, 0]).clamp_min(0.0)
        second_duration = (
            second[:, 1] - second[:, 0]).clamp_min(0.0)
        union = (
            first_duration[:, None] + second_duration[None]
            - intersection).clamp_min(1e-6)
        return intersection / union

    @torch.no_grad()
    def _collect_pre_nms_proposals(
            self, points, fpn_masks, class_logits, offsets,
            regression_features):
        """Collect the exact A0 pre-NMS pool and its frozen features."""
        segments_all, scores_all, labels_all = [], [], []
        features_all, levels_all = [], []
        for level, (logits, boundary, point, mask, feature) in enumerate(zip(
                class_logits, offsets, points, fpn_masks,
                regression_features)):
            probability = (
                logits.sigmoid() * mask.unsqueeze(-1)).flatten()
            keep = probability > self.test_pre_nms_thresh
            probability = probability[keep]
            flat_indices = keep.nonzero(as_tuple=True)[0]
            count = min(
                self.test_pre_nms_topk, int(flat_indices.numel()))
            probability, order = probability.sort(descending=True)
            probability = probability[:count]
            flat_indices = flat_indices[order[:count]]
            point_indices = torch.div(
                flat_indices, self.num_classes, rounding_mode="floor")
            class_indices = torch.fmod(
                flat_indices, self.num_classes).long()
            if boundary.ndim == 3:
                selected_offsets = boundary[
                    point_indices, class_indices]
            else:
                selected_offsets = boundary[point_indices]
            selected_points = point[point_indices]
            predicted = torch.stack((
                selected_points[:, 0]
                - selected_offsets[:, 0] * selected_points[:, 3],
                selected_points[:, 0]
                + selected_offsets[:, 1] * selected_points[:, 3],
            ), dim=-1)
            duration_keep = (
                predicted[:, 1] - predicted[:, 0]
                > self.test_duration_thresh)
            segments_all.append(predicted[duration_keep])
            scores_all.append(probability[duration_keep])
            labels_all.append(class_indices[duration_keep])
            features_all.append(
                feature[point_indices][duration_keep])
            levels_all.append(torch.full(
                (int(duration_keep.sum().item()),), level,
                dtype=torch.long, device=predicted.device))
        if not segments_all:
            feature_dim = regression_features[0].shape[-1]
            empty = class_logits[0].new_empty(0)
            return (
                empty.reshape(0, 2), empty,
                torch.empty(0, dtype=torch.long, device=empty.device),
                empty.reshape(0, feature_dim),
                torch.empty(0, dtype=torch.long, device=empty.device),
            )
        return (
            torch.cat(segments_all).detach(),
            torch.cat(scores_all).detach(),
            torch.cat(labels_all).detach(),
            torch.cat(features_all).detach(),
            torch.cat(levels_all).detach(),
        )

    @torch.no_grad()
    def _greedy_cluster_indices(self, segments, scores, labels):
        """Return score-anchored, same-class local proposal clusters."""
        if not segments.shape[0]:
            return []
        remaining = torch.ones(
            segments.shape[0], dtype=torch.bool, device=segments.device)
        clusters = []
        for anchor in torch.argsort(
                scores, descending=True, stable=True):
            anchor_index = int(anchor.item())
            if not bool(remaining[anchor_index]):
                continue
            candidates = torch.nonzero(remaining, as_tuple=True)[0]
            overlap = self._segment_tiou(
                segments[anchor_index:anchor_index + 1],
                segments[candidates])[0]
            members = candidates[
                (labels[candidates] == labels[anchor_index])
                & (overlap >= self.cluster_overlap_threshold)]
            if not members.numel():
                members = anchor.reshape(1)
            remaining[members] = False
            members = members[torch.argsort(
                scores[members], descending=True, stable=True)]
            clusters.append(members)
        return clusters

    def _cluster_pair_geometry(
            self, anchor_segments, candidate_segments,
            anchor_scores, candidate_scores):
        anchor_duration = (
            anchor_segments[:, 1] - anchor_segments[:, 0]).clamp_min(1e-6)
        candidate_duration = (
            candidate_segments[:, 1]
            - candidate_segments[:, 0]).clamp_min(1e-6)
        overlap = self._segment_tiou(
            anchor_segments, candidate_segments).diagonal()
        anchor_center = anchor_segments.mean(dim=-1)
        candidate_center = candidate_segments.mean(dim=-1)
        return torch.stack((
            (candidate_segments[:, 0] - anchor_segments[:, 0])
            / anchor_duration,
            (candidate_segments[:, 1] - anchor_segments[:, 1])
            / anchor_duration,
            torch.log(candidate_duration / anchor_duration),
            overlap,
            anchor_scores,
            candidate_scores,
            candidate_scores - anchor_scores,
            (candidate_center - anchor_center).abs() / anchor_duration,
        ), dim=-1)

    @torch.no_grad()
    def _cluster_boundary_descriptors(
            self, regression_features, fpn_masks, segments, levels):
        if not segments.shape[0]:
            feature_dim = regression_features[0].shape[-1]
            return segments.new_empty((0, 2 * feature_dim))
        context = self._sample_boundary_context(
            [feature[None] for feature in regression_features],
            [mask[None] for mask in fpn_masks],
            segments,
            torch.zeros_like(levels),
            levels,
            offset_values=self.cluster_context_offsets,
        )
        return self._boundary_contrast_descriptor(context).detach()

    @torch.no_grad()
    def _cluster_fixed_gt_gains(
            self, segments, labels, members, gt_segments, gt_labels):
        """Return member gains against the single GT matched by the anchor.

        The highest-score member is always ``members[0]``.  Its best
        same-class GT is selected once, then every member is compared with
        that exact GT.  A candidate therefore cannot obtain a positive target
        merely by switching to a different nearby instance.
        """
        gains = segments.new_zeros(members.numel())
        if not members.numel() or not gt_segments.numel():
            return gains, 0.0, -1
        anchor = int(members[0].item())
        same_gt = torch.nonzero(
            gt_labels == labels[anchor], as_tuple=True)[0]
        if not same_gt.numel():
            return gains, 0.0, -1
        anchor_overlap = self._segment_tiou(
            segments[anchor:anchor + 1], gt_segments[same_gt])[0]
        target_position = int(anchor_overlap.argmax().item())
        anchor_quality = float(anchor_overlap[target_position].item())
        if anchor_quality <= 1e-8:
            return gains, anchor_quality, -1
        target_gt = int(same_gt[target_position].item())
        member_quality = self._segment_tiou(
            segments[members],
            gt_segments[target_gt:target_gt + 1])[:, 0]
        return member_quality - anchor_quality, anchor_quality, target_gt

    def _cluster_selector_training_loss(
            self, points, fpn_masks, out_cls_logits, out_offsets,
            regression_features, gt_segments, gt_labels):
        """Supervise existing cluster members against the anchor-matched GT."""
        zero = sum(
            parameter.sum() * 0.0
            for parameter in self.cluster_selector_head.parameters())
        predicted_all, confidence_all, target_all = [], [], []
        weight_all, group_all = [], []
        positive_clusters = 0
        training_clusters = 0
        pair_group_offset = 0
        for batch_index, (segments_gt, labels_gt) in enumerate(zip(
                gt_segments, gt_labels)):
            proposals = self._collect_pre_nms_proposals(
                points,
                [mask[batch_index] for mask in fpn_masks],
                [logit[batch_index] for logit in out_cls_logits],
                [offset[batch_index] for offset in out_offsets],
                [feature[batch_index]
                 for feature in regression_features])
            segments, scores, labels, features, levels = proposals
            clusters = self._greedy_cluster_indices(
                segments, scores, labels)
            clusters = [
                members[:self.cluster_max_members]
                for members in clusters if members.numel() > 1
            ][:self.cluster_max_train_clusters]
            if not clusters:
                continue
            descriptors = self._cluster_boundary_descriptors(
                [feature[batch_index]
                 for feature in regression_features],
                [mask[batch_index] for mask in fpn_masks],
                segments, levels)
            for members in clusters:
                anchor = int(members[0].item())
                true_gain, anchor_quality, _ = \
                    self._cluster_fixed_gt_gains(
                        segments, labels, members,
                        segments_gt, labels_gt)
                repeated_anchor = torch.full_like(members, anchor)
                geometry = self._cluster_pair_geometry(
                    segments[repeated_anchor], segments[members],
                    scores[repeated_anchor], scores[members])
                predicted_gain, confidence_logit = \
                    self.cluster_selector_head(
                        features[repeated_anchor], features[members],
                        descriptors[repeated_anchor], descriptors[members],
                        geometry, levels[repeated_anchor], levels[members])
                target_gain = true_gain
                weights = torch.ones_like(true_gain)
                if self.cluster_selector_type == "ap_listwise":
                    thresholds = true_gain.new_tensor(
                        self.cluster_ap_thresholds)
                    qualities = (true_gain + anchor_quality).clamp(
                        min=0.0, max=1.0)
                    member_credit = (
                        qualities[:, None] >= thresholds[None]).to(
                            true_gain.dtype).mean(dim=1)
                    anchor_credit = (
                        true_gain.new_tensor(anchor_quality)
                        >= thresholds).to(true_gain.dtype).mean()
                    target_gain = member_credit - anchor_credit
                    safety_thresholds = true_gain.new_tensor(
                        self.cluster_safety_thresholds)
                    anchor_safe = (
                        true_gain.new_tensor(anchor_quality)
                        >= safety_thresholds)
                    member_safe = (
                        qualities[:, None] >= safety_thresholds[None])
                    harm_fraction = (
                        anchor_safe[None] & ~member_safe).to(
                            true_gain.dtype).mean(dim=1)
                    weights *= 1.0 + (
                        self.cluster_safety_weight * harm_fraction)
                weights[target_gain < 0.0] *= self.cluster_negative_weight
                if (self.cluster_selector_type != "ap_listwise"
                        and anchor_quality >= min(
                            self.cluster_safety_thresholds)
                        and bool((target_gain < 0.0).any())):
                    weights[target_gain < 0.0] *= (
                        1.0 + self.cluster_safety_weight)
                if bool((
                        target_gain
                        > self.cluster_positive_gain_threshold).any()):
                    positive_clusters += 1
                predicted_all.append(predicted_gain)
                confidence_all.append(confidence_logit)
                target_all.append(target_gain.detach())
                weight_all.append(weights)
                group_all.append(torch.full(
                    (members.numel(),), pair_group_offset,
                    dtype=torch.long, device=members.device))
                pair_group_offset += 1
                training_clusters += 1
        if not predicted_all:
            return zero, zero, zero, {
                "cluster_selector_training_clusters": zero.detach(),
                "cluster_selector_positive_cluster_ratio": zero.detach(),
                "cluster_selector_target_gain_mean": zero.detach(),
                "p5_listwise_top1_accuracy": zero.detach(),
                "p5_target_ap_utility_mean": zero.detach(),
            }
        prediction = torch.cat(predicted_all)
        confidence = torch.cat(confidence_all)
        target = torch.cat(target_all)
        weights = torch.cat(weight_all)
        groups = torch.cat(group_all)
        if self.cluster_selector_type == "ap_listwise":
            listwise_terms = []
            for group in torch.unique(groups):
                selected = torch.nonzero(
                    groups == group, as_tuple=True)[0]
                group_target = target[selected]
                if float(group_target.max().item()) <= \
                        self.cluster_positive_gain_threshold:
                    continue
                target_distribution = torch.softmax(
                    group_target
                    / self.cluster_listwise_target_temperature,
                    dim=0)
                log_prediction = torch.log_softmax(
                    prediction[selected]
                    / self.cluster_listwise_temperature,
                    dim=0)
                listwise_terms.append(
                    -(target_distribution * log_prediction).sum())
            gain_loss = (
                torch.stack(listwise_terms).mean()
                if listwise_terms else zero)
        else:
            gain_loss = (
                F.smooth_l1_loss(
                    prediction, target, reduction="none", beta=0.02)
                * weights).sum() / weights.sum().clamp_min(1.0)

        positive_target = (
            target > self.cluster_positive_gain_threshold).to(
                confidence.dtype)
        positive_count = positive_target.sum()
        negative_count = positive_target.numel() - positive_count
        positive_weight = (
            negative_count / positive_count.clamp_min(1.0)).clamp(
                min=1.0, max=20.0)
        abstention_weights = torch.where(
            positive_target > 0.0, positive_weight,
            torch.ones_like(positive_target)) * weights
        abstention_loss = (
            F.binary_cross_entropy_with_logits(
                confidence, positive_target, reduction="none")
            * abstention_weights).sum() / \
            abstention_weights.sum().clamp_min(1.0)

        pairwise_terms = []
        listwise_correct = 0
        listwise_positive_groups = 0
        for group in torch.unique(groups):
            selected = torch.nonzero(
                groups == group, as_tuple=True)[0]
            group_target = target[selected]
            best_position = int(group_target.argmax().item())
            better = selected[best_position]
            if (self.cluster_selector_type == "ap_listwise"
                    and float(group_target[best_position].item())
                    > self.cluster_positive_gain_threshold):
                listwise_positive_groups += 1
                listwise_correct += int(
                    int(prediction[selected].argmax().item())
                    == best_position)
            lower = selected[group_target[best_position] - group_target
                             > self.cluster_pairwise_margin]
            if lower.numel():
                pairwise_terms.append(F.softplus(
                    -(prediction[better] - prediction[lower])
                    / self.cluster_pairwise_temperature).mean())
        pairwise_loss = (
            torch.stack(pairwise_terms).mean()
            if pairwise_terms else zero)
        return gain_loss, pairwise_loss, abstention_loss, {
            "cluster_selector_training_clusters":
                target.new_tensor(float(training_clusters)),
            "cluster_selector_positive_cluster_ratio":
                target.new_tensor(
                    positive_clusters / max(training_clusters, 1)),
            "cluster_selector_target_gain_mean": target.mean().detach(),
            "p5_listwise_top1_accuracy": target.new_tensor(
                listwise_correct / max(listwise_positive_groups, 1)),
            "p5_target_ap_utility_mean": (
                target.mean().detach()
                if self.cluster_selector_type == "ap_listwise"
                else target.new_zeros(())),
        }

    @torch.no_grad()
    def _cluster_anchor_qualities(
            self, segments, labels, anchors, gt_segments, gt_labels):
        """Maximum same-class tIoU for cluster anchors (training only)."""
        qualities = segments.new_zeros(anchors.numel())
        if not anchors.numel() or not gt_segments.numel():
            return qualities
        overlap = self._segment_tiou(segments[anchors], gt_segments)
        same_class = labels[anchors, None] == gt_labels[None]
        return overlap.masked_fill(~same_class, 0.0).max(dim=1).values

    def _cluster_score_suppression_training_loss(
            self, points, fpn_masks, out_cls_logits, out_offsets,
            regression_features, gt_segments, gt_labels):
        """Learn the P4-A bounded false-positive suppression decision.

        Proposal construction and clustering are discrete and detached.  GT is
        used only here to form a target logit decrement; the inference path
        receives only proposal features, boundary context, score, duration,
        and FPN level.
        """
        zero = sum(
            parameter.sum() * 0.0
            for parameter in self.cluster_score_suppression_head.parameters())
        amount_all, confidence_all, target_all = [], [], []
        training_anchors = 0
        positive_anchors = 0
        for batch_index, (segments_gt, labels_gt) in enumerate(zip(
                gt_segments, gt_labels)):
            proposals = self._collect_pre_nms_proposals(
                points,
                [mask[batch_index] for mask in fpn_masks],
                [logit[batch_index] for logit in out_cls_logits],
                [offset[batch_index] for offset in out_offsets],
                [feature[batch_index]
                 for feature in regression_features])
            segments, scores, labels, features, levels = proposals
            clusters = self._greedy_cluster_indices(
                segments, scores, labels)[:self.cluster_max_train_clusters]
            if not clusters:
                continue
            anchors = torch.stack([members[0] for members in clusters])
            descriptors = self._cluster_boundary_descriptors(
                [feature[batch_index]
                 for feature in regression_features],
                [mask[batch_index] for mask in fpn_masks],
                segments, levels)
            durations = (
                segments[anchors, 1] - segments[anchors, 0]).clamp_min(1e-6)
            predicted_amount, confidence_logit = \
                self.cluster_score_suppression_head(
                    features[anchors], descriptors[anchors],
                    scores[anchors], durations, levels[anchors])
            quality = self._cluster_anchor_qualities(
                segments, labels, anchors, segments_gt, labels_gt)
            safe_score = scores[anchors].clamp(
                min=1e-5, max=1.0 - 1e-5)
            if self.cluster_selector_type == "ap_listwise":
                # Map localization quality to a soft approximation of the
                # exact ActivityNet AP credit averaged over 0.50:0.05:0.95.
                # This makes false-positive suppression optimize the same
                # threshold utility as boundary selection instead of raw
                # tIoU calibration.
                thresholds = quality.new_tensor(
                    self.cluster_ap_thresholds)
                ap_credit = torch.sigmoid(
                    (quality[:, None] - thresholds[None])
                    / self.cluster_ap_credit_sigma).mean(dim=1)
                safe_quality = ap_credit.clamp(
                    min=1e-5, max=1.0 - 1e-5)
            else:
                safe_quality = quality.clamp(
                    min=1e-5, max=1.0 - 1e-5)
            target = (
                torch.logit(safe_score) - torch.logit(safe_quality)
            ).clamp(
                min=0.0,
                max=self.cluster_score_max_logit_suppression)
            amount_all.append(predicted_amount)
            confidence_all.append(confidence_logit)
            target_all.append(target.detach())
            positive_anchors += int((
                target > self.cluster_score_positive_threshold).sum().item())
            training_anchors += int(anchors.numel())
        if not amount_all:
            return zero, zero, {
                "cluster_score_training_anchors": zero.detach(),
                "cluster_score_positive_anchor_ratio": zero.detach(),
                "cluster_score_target_suppression_mean": zero.detach(),
            }
        amount = torch.cat(amount_all)
        confidence = torch.cat(confidence_all)
        target = torch.cat(target_all)
        positive_target = (
            target > self.cluster_score_positive_threshold).to(
                confidence.dtype)
        positive_count = positive_target.sum()
        negative_count = positive_target.numel() - positive_count
        positive_weight = (
            negative_count / positive_count.clamp_min(1.0)).clamp(
                min=1.0, max=20.0)
        weights = torch.where(
            positive_target > 0.0, positive_weight,
            torch.ones_like(positive_target))
        amount_loss = (
            F.smooth_l1_loss(
                amount, target, reduction="none", beta=0.05)
            * weights).sum() / weights.sum().clamp_min(1.0)
        confidence_loss = (
            F.binary_cross_entropy_with_logits(
                confidence, positive_target, reduction="none")
            * weights).sum() / weights.sum().clamp_min(1.0)
        return amount_loss, confidence_loss, {
            "cluster_score_training_anchors":
                target.new_tensor(float(training_anchors)),
            "cluster_score_positive_anchor_ratio":
                target.new_tensor(
                    positive_anchors / max(training_anchors, 1)),
            "cluster_score_target_suppression_mean": target.mean().detach(),
        }

    def _dense_iou_proposal_quality_loss(
            self, points, fpn_masks, out_cls_logits, quality_logits,
            out_offsets, gt_segments, gt_labels):
        """Supervise top pre-NMS candidates with their actual class tIoU.

        Candidate selection uses detached A0 scores.  Every selected candidate
        receives the maximum tIoU to same-class GT, or zero when unmatched.
        Only the quality head receives gradients.
        """
        quality_all = torch.cat(quality_logits, dim=1)
        original_all = torch.cat(out_cls_logits, dim=1).detach()
        valid_all = torch.cat(fpn_masks, dim=1)
        offsets_all = out_offsets.detach()
        concat_points = torch.cat(points, dim=0)
        centers = concat_points[:, 0]
        strides = concat_points[:, 3]
        zero = quality_all.sum() * 0.0
        loss_sum = zero
        target_sum = zero.detach()
        selected_total = 0

        for batch_index, (segments, labels) in enumerate(zip(
                gt_segments, gt_labels)):
            valid_indices = valid_all[batch_index].nonzero(
                as_tuple=True)[0]
            if valid_indices.numel() == 0:
                continue
            original_scores = original_all[
                batch_index, valid_indices].sigmoid().flatten()
            count = min(
                self.dense_iou_proposal_target_topk,
                int(original_scores.numel()))
            if count == 0:
                continue
            selected_flat = torch.topk(
                original_scores, count, sorted=False).indices
            local_point_indices = torch.div(
                selected_flat, self.num_classes, rounding_mode="floor")
            selected_classes = torch.fmod(
                selected_flat, self.num_classes).long()
            point_indices = valid_indices[local_point_indices]
            if offsets_all.ndim == 4:
                selected_offsets = offsets_all[
                    batch_index, point_indices, selected_classes]
            elif offsets_all.ndim == 3:
                selected_offsets = offsets_all[
                    batch_index, point_indices]
            else:
                raise ValueError(
                    "unexpected proposal-quality offset rank")
            selected_centers = centers[point_indices]
            selected_strides = strides[point_indices]
            predicted_segments = torch.stack((
                selected_centers
                - selected_offsets[:, 0] * selected_strides,
                selected_centers
                + selected_offsets[:, 1] * selected_strides,
            ), dim=-1)
            true_quality = quality_all.new_zeros(count)
            if segments.numel() > 0:
                for class_index in selected_classes.unique():
                    candidate_mask = selected_classes == class_index
                    class_targets = segments[labels == class_index]
                    if class_targets.numel() == 0:
                        continue
                    true_quality[candidate_mask] = self._segment_tiou(
                        predicted_segments[candidate_mask],
                        class_targets,
                    ).max(dim=1).values
            predicted_quality = quality_all[
                batch_index, point_indices, selected_classes]
            probability = predicted_quality.sigmoid()
            modulator = (
                probability - true_quality).abs().pow(
                    self.dense_iou_quality_beta)
            loss_sum = loss_sum + (
                F.binary_cross_entropy_with_logits(
                    predicted_quality, true_quality, reduction="none")
                * modulator).sum()
            target_sum = target_sum + true_quality.detach().sum()
            selected_total += count

        if selected_total == 0:
            return zero, zero.detach()
        return (
            loss_sum / float(selected_total),
            target_sum / float(selected_total),
        )

    def _dense_iou_ranking_losses(
            self, points, fpn_masks, out_cls_logits, calibrated_cls_logits,
            out_offsets, gt_segments, gt_labels, quality_logits=None):
        """Rank overlapping high-score candidates by their true class tIoU.

        Candidate mining uses the frozen classification logits. Targets are
        recomputed against every same-class GT segment, so nearby candidates
        outside point-assignment center sampling are not mislabeled as
        background. Only calibrated scores receive gradients.
        """
        calibrated_all = torch.cat(calibrated_cls_logits, dim=1)
        if self.dense_iou_ranking_source == "quality_logits":
            if quality_logits is None:
                raise ValueError(
                    "quality-logit ranking requires quality predictions")
            ranking_all = torch.cat(quality_logits, dim=1)
        else:
            ranking_all = calibrated_all
        original_all = torch.cat(out_cls_logits, dim=1).detach()
        offsets_all = out_offsets.detach()
        valid_all = torch.cat(fpn_masks, dim=1)
        concat_points = torch.cat(points, dim=0)
        centers = concat_points[:, 0]
        strides = concat_points[:, 3]
        zero = calibrated_all.sum() * 0.0
        pairwise_numerator = zero
        pairwise_denominator = zero
        listwise_total = zero
        listwise_groups = 0

        thresholds = calibrated_all.new_tensor(
            self.dense_iou_ranking_thresholds)
        sigma_squared = self.dense_iou_proximity_sigma ** 2

        for batch_index, (segments, labels) in enumerate(zip(
                gt_segments, gt_labels)):
            valid_indices = valid_all[batch_index].nonzero(
                as_tuple=True)[0]
            if valid_indices.numel() < 2 or segments.numel() == 0:
                continue
            present_classes = labels.unique()
            for class_index_tensor in present_classes:
                class_index = int(class_index_tensor.item())
                class_targets = segments[labels == class_index_tensor]
                if class_targets.numel() == 0:
                    continue
                original_scores = original_all[
                    batch_index, valid_indices, class_index]
                count = min(
                    self.dense_iou_ranking_topk,
                    int(original_scores.numel()))
                if count < 2:
                    continue
                selected_positions = torch.topk(
                    original_scores, count, sorted=False).indices
                point_indices = valid_indices[selected_positions]
                if offsets_all.ndim == 4:
                    selected_offsets = offsets_all[
                        batch_index, point_indices, class_index]
                elif offsets_all.ndim == 3:
                    selected_offsets = offsets_all[
                        batch_index, point_indices]
                else:
                    raise ValueError(
                        "unexpected dense ranking offset rank")
                selected_centers = centers[point_indices]
                selected_strides = strides[point_indices]
                predicted_segments = torch.stack((
                    selected_centers
                    - selected_offsets[:, 0] * selected_strides,
                    selected_centers
                    + selected_offsets[:, 1] * selected_strides,
                ), dim=-1)
                true_quality = self._segment_tiou(
                    predicted_segments, class_targets).max(dim=-1).values
                predicted_scores = ranking_all[
                    batch_index, point_indices, class_index]

                neighbourhood_tiou = self._segment_tiou(
                    predicted_segments, predicted_segments)
                quality_difference = (
                    true_quality[:, None] - true_quality[None, :])
                pair_mask = torch.logical_and(
                    quality_difference > self.dense_iou_pair_margin,
                    neighbourhood_tiou
                    >= self.dense_iou_overlap_threshold)
                pair_mask.fill_diagonal_(False)
                if not bool(pair_mask.any()):
                    continue

                proximity = torch.exp(
                    -(
                        true_quality[:, None] - thresholds[None]
                    ).square() / (2.0 * sigma_squared)
                ).max(dim=-1).values
                pair_proximity = torch.maximum(
                    proximity[:, None], proximity[None, :])
                threshold_crossing = torch.logical_and(
                    true_quality[:, None, None] >= thresholds[None, None],
                    true_quality[None, :, None] < thresholds[None, None],
                ).any(dim=-1)
                pair_weights = (
                    1.0
                    + self.dense_iou_proximity_gain * pair_proximity
                    + self.dense_iou_crossing_gain
                    * threshold_crossing.to(pair_proximity.dtype)
                )
                if self.dense_iou_safety_thresholds:
                    safety_thresholds = calibrated_all.new_tensor(
                        self.dense_iou_safety_thresholds)
                    harmful_crossing = torch.logical_and(
                        true_quality[:, None, None]
                        >= safety_thresholds[None, None],
                        true_quality[None, :, None]
                        < safety_thresholds[None, None],
                    ).any(dim=-1)
                    pair_weights = (
                        pair_weights
                        + self.dense_iou_safety_weight
                        * harmful_crossing.to(pair_weights.dtype))
                pair_weights = pair_weights.detach()
                score_difference = (
                    predicted_scores[:, None] - predicted_scores[None, :])
                pair_losses = F.softplus(
                    -score_difference / self.dense_iou_pair_temperature)
                selected_weights = pair_weights[pair_mask]
                pairwise_numerator = pairwise_numerator + (
                    pair_losses[pair_mask] * selected_weights).sum()
                pairwise_denominator = pairwise_denominator + \
                    selected_weights.sum()

                competitors = torch.logical_or(
                    pair_mask.any(dim=0), pair_mask.any(dim=1))
                competitor_scores = predicted_scores[competitors]
                competitor_quality = true_quality[competitors]
                if competitor_scores.numel() >= 2:
                    target_distribution = F.softmax(
                        competitor_quality
                        / self.dense_iou_target_temperature,
                        dim=0).detach()
                    predicted_log_distribution = F.log_softmax(
                        competitor_scores
                        / self.dense_iou_listwise_temperature,
                        dim=0)
                    listwise_total = listwise_total - (
                        target_distribution
                        * predicted_log_distribution).sum()
                    listwise_groups += 1

        if float(pairwise_denominator.detach().item()) > 0.0:
            pairwise_loss = (
                pairwise_numerator / pairwise_denominator)
        else:
            pairwise_loss = zero
        if listwise_groups:
            listwise_loss = listwise_total / float(listwise_groups)
        else:
            listwise_loss = zero
        return pairwise_loss, listwise_loss

    def _task_aligned_pairwise_loss(
            self, scores, qualities, boundaries, batch_ids, class_ids):
        """Rank overlapping same-class positives by their detached tIoU.

        The loss acts directly on the original classification logits.  The
        current boundary prediction supplies only the detached ordering
        target, so the ranking term cannot exploit a trivial gradient through
        its own label.
        """
        zero = scores.sum() * 0.0
        if scores.numel() < 2:
            return zero
        qualities = qualities.detach()
        boundaries = boundaries.detach()
        thresholds = qualities.new_tensor(self.task_alignment_thresholds)
        numerator = zero
        denominator = qualities.new_zeros(())

        for batch_id in torch.unique(batch_ids):
            batch_mask = batch_ids == batch_id
            for class_id in torch.unique(class_ids[batch_mask]):
                selected = torch.nonzero(
                    batch_mask & (class_ids == class_id),
                    as_tuple=True)[0]
                if selected.numel() < 2:
                    continue
                if selected.numel() > self.task_alignment_ranking_topk:
                    keep = torch.topk(
                        scores[selected].detach(),
                        self.task_alignment_ranking_topk).indices
                    selected = selected[keep]

                selected_scores = scores[selected]
                selected_quality = qualities[selected]
                selected_boundaries = boundaries[selected]
                starts = selected_boundaries[:, 0]
                ends = selected_boundaries[:, 1]
                intersection = (
                    torch.minimum(ends[:, None], ends[None, :])
                    - torch.maximum(starts[:, None], starts[None, :])
                ).clamp_min(0.0)
                union = (
                    torch.maximum(ends[:, None], ends[None, :])
                    - torch.minimum(starts[:, None], starts[None, :])
                ).clamp_min(1e-6)
                overlap = intersection / union
                quality_difference = (
                    selected_quality[:, None]
                    - selected_quality[None, :])
                valid_pairs = (
                    (quality_difference
                     > self.task_alignment_quality_margin)
                    & (overlap >= self.task_alignment_overlap_threshold)
                )
                if not bool(valid_pairs.any()):
                    continue

                score_difference = (
                    selected_scores[:, None]
                    - selected_scores[None, :])
                ranking_loss = F.softplus(
                    -score_difference
                    / self.task_alignment_pairwise_temperature)
                threshold_distance = (
                    selected_quality[:, None] - thresholds[None]).abs()
                proximity = torch.exp(
                    -threshold_distance
                    / self.task_alignment_proximity_sigma).amax(dim=-1)
                pair_weight = 1.0 + self.task_alignment_proximity_gain * \
                    torch.maximum(proximity[:, None], proximity[None, :])
                numerator = numerator + (
                    ranking_loss[valid_pairs]
                    * pair_weight[valid_pairs]).sum()
                denominator = denominator + pair_weight[valid_pairs].sum()

        if float(denominator.detach().item()) == 0.0:
            return zero
        return numerator / denominator

    def losses(
            self, fpn_masks,
            out_cls_logits, out_offsets,
            gt_cls_labels, gt_offsets,
            points=None, reg_features=None,
            coarse_offsets=None, distribution_logits=None,
            trident_candidates=None,
            quality_logits=None, calibrated_cls_logits=None,
            gt_segments=None, gt_labels=None,
            augmented_cls_logits=None, augmented_offsets=None,
            boundary_proposal_candidates=None,
            boundary_proposal_logits=None,
            boundary_proposal_improvement=None,
            a0_cls_logits=None, a0_offsets=None,
            residual_cls_values=None, residual_reg_values=None,
            residual_cls_confidence=None,
            residual_reg_confidence=None):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        out_offsets_by_level = out_offsets
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        out_offsets = torch.cat(out_offsets, dim=1)
        pred_offsets_all = out_offsets[pos_mask]
        gt_offsets_positive = torch.stack(gt_offsets)[pos_mask]
        positive_classes = gt_cls[pos_mask].bool()
        if pred_offsets_all.ndim == 3:
            selected_points, selected_classes = torch.where(
                positive_classes)
            pred_offsets = pred_offsets_all[
                selected_points, selected_classes]
            gt_offsets_for_reg = gt_offsets_positive[selected_points]
        elif pred_offsets_all.ndim == 2:
            selected_points = torch.arange(
                pred_offsets_all.shape[0], device=pred_offsets_all.device)
            selected_classes = None
            pred_offsets = pred_offsets_all
            gt_offsets_for_reg = gt_offsets_positive
        else:
            raise ValueError("unexpected regression offset rank")
          
        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)
        
        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask].clone()
        positive_target_mask = gt_target > 0

        localization_iou = None
        if pred_offsets.numel() > 0:
            localization_iou = 1.0 - ctr_giou_loss_1d(
                pred_offsets, gt_offsets_for_reg, reduction='none')
            localization_iou = localization_iou.clamp(
                min=0.0, max=1.0)

        raw_cls_prediction = torch.cat(
            out_cls_logits, dim=1)[valid_mask]
        hard_cls_target = gt_target.clone()
        residual_cls_consistency_loss = \
            raw_cls_prediction.sum() * 0.0
        residual_reg_consistency_loss = \
            raw_cls_prediction.sum() * 0.0
        residual_magnitude_loss = raw_cls_prediction.sum() * 0.0
        residual_confidence_loss = raw_cls_prediction.sum() * 0.0
        residual_cls_positive_ratio = raw_cls_prediction.sum() * 0.0
        residual_reg_positive_ratio = raw_cls_prediction.sum() * 0.0
        a0_localization_iou = None
        if self.residual_alignment_enabled:
            if any(value is None for value in (
                    a0_cls_logits, a0_offsets,
                    residual_cls_values, residual_reg_values,
                    residual_cls_confidence,
                    residual_reg_confidence)):
                raise ValueError(
                    "residual alignment training tensors are missing")
            a0_cls_prediction = torch.cat(
                a0_cls_logits, dim=1)[valid_mask].detach()
            if a0_cls_prediction.shape != raw_cls_prediction.shape:
                raise ValueError(
                    "A0 and residual classification shapes differ")
            residual_cls_consistency_loss = F.smooth_l1_loss(
                raw_cls_prediction, a0_cls_prediction)

            a0_offsets_all = torch.cat(a0_offsets, dim=1).detach()
            if a0_offsets_all.shape != out_offsets.shape:
                raise ValueError(
                    "A0 and residual regression shapes differ")
            residual_reg_consistency_loss = F.smooth_l1_loss(
                out_offsets[valid_mask], a0_offsets_all[valid_mask])
            cls_residual = torch.cat(
                residual_cls_values, dim=1)[valid_mask]
            reg_residual = torch.cat(
                residual_reg_values, dim=1)[valid_mask]
            residual_magnitude_loss = (
                cls_residual.abs().mean()
                + reg_residual.abs().mean())

            a0_pred_offsets_all = a0_offsets_all[pos_mask]
            if a0_pred_offsets_all.ndim == 3:
                a0_pred_offsets = a0_pred_offsets_all[
                    selected_points, selected_classes]
            else:
                a0_pred_offsets = a0_pred_offsets_all
            if a0_pred_offsets.numel() > 0:
                a0_localization_iou = 1.0 - ctr_giou_loss_1d(
                    a0_pred_offsets, gt_offsets_for_reg,
                    reduction='none')
                a0_localization_iou = a0_localization_iou.clamp(
                    min=0.0, max=1.0).detach()

            class_confidence = torch.cat(
                residual_cls_confidence, dim=1)[valid_mask]
            aligned_class_error = F.binary_cross_entropy_with_logits(
                raw_cls_prediction.detach(), hard_cls_target,
                reduction='none')
            a0_class_error = F.binary_cross_entropy_with_logits(
                a0_cls_prediction, hard_cls_target, reduction='none')
            class_gain = a0_class_error - aligned_class_error
            class_positive = (
                class_gain > self.residual_alignment_confidence_gain_margin)
            class_negative = (
                class_gain < -self.residual_alignment_confidence_harm_margin)
            class_supervised = class_positive | class_negative
            if bool(class_supervised.any()):
                residual_confidence_loss = \
                    residual_confidence_loss + \
                    F.binary_cross_entropy_with_logits(
                        class_confidence[class_supervised],
                        class_positive[class_supervised].to(
                            class_confidence.dtype))
                residual_cls_positive_ratio = class_positive[
                    class_supervised].to(
                        class_confidence.dtype).mean().detach()

            if localization_iou is not None \
                    and a0_localization_iou is not None:
                regression_gain = (
                    localization_iou.detach() - a0_localization_iou)
                regression_positive = (
                    regression_gain
                    > self.residual_alignment_confidence_gain_margin)
                regression_negative = (
                    regression_gain
                    < -self.residual_alignment_confidence_harm_margin)
                regression_supervised = (
                    regression_positive | regression_negative)
                regression_confidence = torch.cat(
                    residual_reg_confidence, dim=1)[pos_mask].squeeze(-1)
                if bool(regression_supervised.any()):
                    residual_confidence_loss = \
                        residual_confidence_loss + \
                        F.binary_cross_entropy_with_logits(
                            regression_confidence[regression_supervised],
                            regression_positive[regression_supervised].to(
                                regression_confidence.dtype))
                    residual_reg_positive_ratio = regression_positive[
                        regression_supervised].to(
                            regression_confidence.dtype).mean().detach()
        task_alignment_pairwise_loss = raw_cls_prediction.sum() * 0.0
        task_alignment_mean = raw_cls_prediction.sum() * 0.0
        task_alignment_quality_mean = raw_cls_prediction.sum() * 0.0
        task_alignment_positive_weight = None
        if self.task_alignment_enabled and localization_iou is not None:
            quality_target = torch.zeros_like(gt_target)
            if pred_offsets_all.ndim == 3:
                if int(positive_target_mask.sum().item()) != \
                        localization_iou.numel():
                    raise ValueError(
                        "task alignment class-aware quality mismatch")
                quality_target[positive_target_mask] = \
                    localization_iou.detach()
            else:
                positive_rows = gt_target.sum(dim=-1) > 0
                if int(positive_rows.sum().item()) != \
                        localization_iou.numel():
                    raise ValueError(
                        "task alignment point quality mismatch")
                row_quality = torch.zeros(
                    gt_target.shape[0], device=gt_target.device,
                    dtype=gt_target.dtype)
                row_quality[positive_rows] = localization_iou.detach()
                quality_target = row_quality[:, None].expand_as(
                    gt_target).clone()

            positive_quality = quality_target[positive_target_mask]
            positive_score = raw_cls_prediction[
                positive_target_mask].sigmoid().detach()
            alignment = (
                positive_score.pow(self.task_alignment_alpha)
                * positive_quality.pow(self.task_alignment_beta))
            normalized_alignment = alignment / \
                alignment.mean().clamp_min(1e-6)
            task_alignment_positive_weight = normalized_alignment.clamp(
                min=self.task_alignment_minimum_positive_weight,
                max=self.task_alignment_maximum_positive_weight)
            mix = self.task_alignment_soft_label_mix
            gt_target[positive_target_mask] = (
                (1.0 - mix) + mix * positive_quality)
            task_alignment_mean = alignment.mean()
            task_alignment_quality_mean = positive_quality.mean()

            if (self.task_alignment_pairwise_weight > 0.0
                    and points is not None
                    and positive_quality.numel() > 1):
                entry_rows, entry_classes = torch.where(
                    positive_target_mask)
                concat_points = torch.cat(points, dim=0)
                batch_size = valid_mask.shape[0]
                centers = concat_points[:, 0][None].expand(
                    batch_size, -1)[valid_mask]
                strides = concat_points[:, 3][None].expand(
                    batch_size, -1)[valid_mask]
                valid_batch_ids = torch.arange(
                    batch_size, device=valid_mask.device)[:, None].expand_as(
                        valid_mask)[valid_mask]
                valid_offsets = out_offsets[valid_mask]
                if valid_offsets.ndim == 3:
                    entry_offsets = valid_offsets[
                        entry_rows, entry_classes]
                else:
                    entry_offsets = valid_offsets[entry_rows]
                entry_centers = centers[entry_rows]
                entry_strides = strides[entry_rows]
                entry_boundaries = torch.stack((
                    entry_centers
                    - entry_offsets[:, 0] * entry_strides,
                    entry_centers
                    + entry_offsets[:, 1] * entry_strides,
                ), dim=-1)
                task_alignment_pairwise_loss = \
                    self._task_aligned_pairwise_loss(
                        raw_cls_prediction[positive_target_mask],
                        positive_quality,
                        entry_boundaries,
                        valid_batch_ids[entry_rows],
                        entry_classes,
                    )

        # optional label smoothing is applied after V13-A has replaced the
        # hard positive target with its localization-aware soft target.
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            raw_cls_prediction,
            gt_target,
            reduction='none')
        if task_alignment_positive_weight is not None:
            cls_loss[positive_target_mask] *= \
                task_alignment_positive_weight
        if (self.trident_enabled and localization_iou is not None
                and self.trident_iou_weight_power > 0.0):
            cls_loss[positive_target_mask] *= localization_iou.pow(
                self.trident_iou_weight_power)
        cls_loss = cls_loss.sum()
        cls_loss /= self.loss_normalizer

        dense_iou_quality_loss = cls_loss.new_zeros(())
        dense_iou_target_mean = cls_loss.new_zeros(())
        dense_iou_calibration_loss = cls_loss.new_zeros(())
        dense_iou_pairwise_loss = cls_loss.new_zeros(())
        dense_iou_listwise_loss = cls_loss.new_zeros(())
        if self.dense_iou_quality_enabled:
            if quality_logits is None or calibrated_cls_logits is None:
                raise ValueError(
                    "dense IoU quality training requires quality logits")
            quality_prediction = torch.cat(
                quality_logits, dim=1)[valid_mask]
            calibrated_prediction = torch.cat(
                calibrated_cls_logits, dim=1)[valid_mask]
            quality_target = torch.zeros_like(quality_prediction)
            if localization_iou is not None:
                if int(positive_target_mask.sum().item()) != \
                        localization_iou.numel():
                    raise ValueError(
                        "dense quality target and positive offsets mismatch")
                quality_target[positive_target_mask] = \
                    localization_iou.detach()

            def quality_focal_loss(prediction):
                probability = prediction.sigmoid()
                modulator = (
                    probability - quality_target).abs().pow(
                        self.dense_iou_quality_beta)
                return (
                    F.binary_cross_entropy_with_logits(
                        prediction, quality_target, reduction="none")
                    * modulator).sum() / self.loss_normalizer

            if self.dense_iou_proposal_target_enabled:
                if points is None or gt_segments is None or gt_labels is None:
                    raise ValueError(
                        "proposal-quality targets require points and video GT")
                dense_iou_quality_loss, dense_iou_target_mean = \
                    self._dense_iou_proposal_quality_loss(
                        points, fpn_masks, out_cls_logits, quality_logits,
                        out_offsets, gt_segments, gt_labels)
            else:
                dense_iou_quality_loss = quality_focal_loss(
                    quality_prediction)
                positive_quality = quality_target[
                    positive_target_mask]
                if positive_quality.numel():
                    dense_iou_target_mean = positive_quality.mean().detach()
            if self.dense_iou_calibration_weight > 0.0:
                dense_iou_calibration_loss = quality_focal_loss(
                    calibrated_prediction)
            if self.dense_iou_ranking_enabled:
                if points is None or gt_segments is None or gt_labels is None:
                    raise ValueError(
                        "dense IoU ranking requires points and video GT")
                dense_iou_pairwise_loss, dense_iou_listwise_loss = \
                    self._dense_iou_ranking_losses(
                        points, fpn_masks, out_cls_logits,
                        calibrated_cls_logits, out_offsets,
                        gt_segments, gt_labels, quality_logits)

        cluster_gain_loss = cls_loss.new_zeros(())
        cluster_pairwise_loss = cls_loss.new_zeros(())
        cluster_abstention_loss = cls_loss.new_zeros(())
        cluster_score_amount_loss = cls_loss.new_zeros(())
        cluster_score_confidence_loss = cls_loss.new_zeros(())
        cluster_training_stats = {}
        if self.cluster_selector_enabled:
            if (points is None or reg_features is None
                    or gt_segments is None or gt_labels is None):
                raise ValueError(
                    "cluster selector training requires proposals, "
                    "features, and training GT")
            (cluster_gain_loss, cluster_pairwise_loss,
             cluster_abstention_loss, cluster_training_stats) = \
                self._cluster_selector_training_loss(
                    points, fpn_masks, out_cls_logits,
                    out_offsets_by_level, reg_features,
                    gt_segments, gt_labels)
            if self.cluster_score_suppression_enabled:
                (cluster_score_amount_loss,
                 cluster_score_confidence_loss,
                 cluster_score_training_stats) = \
                    self._cluster_score_suppression_training_loss(
                        points, fpn_masks, out_cls_logits,
                        out_offsets_by_level, reg_features,
                        gt_segments, gt_labels)
                cluster_training_stats.update(
                    cluster_score_training_stats)
            
        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets_for_reg,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        distribution_loss = reg_loss.new_zeros(())
        start_end_tiou_loss = reg_loss.new_zeros(())
        if self.start_end_distribution_enabled and num_pos > 0:
            if coarse_offsets is None or distribution_logits is None:
                raise ValueError(
                    "Start/end training requires coarse offsets and logits")
            coarse_positive_all = torch.cat(
                coarse_offsets, dim=1)[pos_mask]
            coarse_positive = coarse_positive_all[selected_points].detach()
            logits_positive_all = torch.cat(
                distribution_logits, dim=1)[pos_mask]
            logits_positive = logits_positive_all[selected_points]
            target_residual = (
                gt_offsets_for_reg - coarse_positive).clamp(
                    -self.start_end_distribution_head.max_residual,
                    self.start_end_distribution_head.max_residual)
            scaled_target = (
                (target_residual
                 + self.start_end_distribution_head.max_residual)
                / (2.0 * self.start_end_distribution_head.max_residual)
                * (self.start_end_distribution_head.num_bins - 1))
            lower = scaled_target.floor().long()
            upper = torch.clamp(
                lower + 1,
                max=self.start_end_distribution_head.num_bins - 1)
            upper_weight = scaled_target - lower.to(scaled_target.dtype)
            log_probabilities = F.log_softmax(logits_positive, dim=-1)
            lower_logp = log_probabilities.gather(
                -1, lower.unsqueeze(-1)).squeeze(-1)
            upper_logp = log_probabilities.gather(
                -1, upper.unsqueeze(-1)).squeeze(-1)
            distribution_loss = -(
                lower_logp * (1.0 - upper_weight)
                + upper_logp * upper_weight).sum()
            distribution_loss /= (2.0 * self.loss_normalizer)

            intersection = (
                torch.minimum(pred_offsets[:, 0], gt_offsets_for_reg[:, 0])
                + torch.minimum(
                    pred_offsets[:, 1], gt_offsets_for_reg[:, 1]))
            union = (
                torch.maximum(pred_offsets[:, 0], gt_offsets_for_reg[:, 0])
                + torch.maximum(
                    pred_offsets[:, 1], gt_offsets_for_reg[:, 1])
            ).clamp_min(1e-6)
            start_end_tiou_loss = (1.0 - intersection / union).sum()
            start_end_tiou_loss /= self.loss_normalizer

        proposal_distribution_loss = reg_loss.new_zeros(())
        proposal_coverage_loss = reg_loss.new_zeros(())
        proposal_improvement_loss = reg_loss.new_zeros(())
        if self.boundary_proposal_enabled and num_pos > 0:
            if (coarse_offsets is None
                    or boundary_proposal_candidates is None
                    or boundary_proposal_logits is None
                    or boundary_proposal_improvement is None):
                raise ValueError(
                    "V13-C training requires proposal candidates and logits")
            coarse_positive = torch.cat(
                coarse_offsets, dim=1)[pos_mask][selected_points].detach()
            logits_positive = torch.cat(
                boundary_proposal_logits, dim=1
            )[pos_mask][selected_points]
            candidate_positive = torch.cat(
                boundary_proposal_candidates, dim=1
            )[pos_mask][selected_points]
            improvement_positive = torch.cat(
                boundary_proposal_improvement, dim=1
            )[pos_mask][selected_points]
            target_residual = (
                gt_offsets_for_reg - coarse_positive).clamp(
                    -self.boundary_proposal_generator.max_residual,
                    self.boundary_proposal_generator.max_residual)
            scaled_target = (
                (target_residual
                 + self.boundary_proposal_generator.max_residual)
                / (2.0
                   * self.boundary_proposal_generator.max_residual)
                * (self.boundary_proposal_generator.num_bins - 1))
            lower = scaled_target.floor().long()
            upper = torch.clamp(
                lower + 1,
                max=self.boundary_proposal_generator.num_bins - 1)
            upper_weight = scaled_target - lower.to(scaled_target.dtype)
            log_probabilities = F.log_softmax(logits_positive, dim=-1)
            lower_logp = log_probabilities.gather(
                -1, lower.unsqueeze(-1)).squeeze(-1)
            upper_logp = log_probabilities.gather(
                -1, upper.unsqueeze(-1)).squeeze(-1)
            proposal_distribution_loss = -(
                lower_logp * (1.0 - upper_weight)
                + upper_logp * upper_weight).sum()
            proposal_distribution_loss /= (
                2.0 * self.loss_normalizer)

            candidate_losses = torch.stack([
                ctr_diou_loss_1d(
                    candidate_positive[:, candidate_index],
                    gt_offsets_for_reg, reduction='none')
                for candidate_index in range(
                    candidate_positive.shape[1])
            ], dim=1)
            proposal_coverage_loss = candidate_losses.min(dim=1).values.sum()
            proposal_coverage_loss /= self.loss_normalizer

            coarse_tiou = 1.0 - ctr_giou_loss_1d(
                coarse_positive, gt_offsets_for_reg, reduction='none')
            candidate_tiou = torch.stack([
                1.0 - ctr_giou_loss_1d(
                    candidate_positive[:, candidate_index],
                    gt_offsets_for_reg, reduction='none')
                for candidate_index in range(
                    candidate_positive.shape[1])
            ], dim=1)
            improvement_target = (
                candidate_tiou > coarse_tiou[:, None] + 1e-4
            ).to(improvement_positive.dtype)
            proposal_improvement_loss = \
                F.binary_cross_entropy_with_logits(
                    improvement_positive, improvement_target,
                    reduction='sum') / self.loss_normalizer

        trident_aux_loss = reg_loss.new_zeros(())
        if self.trident_enabled and num_pos > 0:
            if trident_candidates is None:
                raise ValueError(
                    "Trident training requires decoded candidates")
            candidate_positive = torch.cat(
                trident_candidates, dim=1)[pos_mask]
            candidate_selected = candidate_positive[
                selected_points, selected_classes]
            trident_aux_loss = ctr_diou_loss_1d(
                candidate_selected, gt_offsets_for_reg, reduction='sum')
            trident_aux_loss /= self.loss_normalizer

        temporal_consistency_cls_loss = reg_loss.new_zeros(())
        temporal_consistency_reg_loss = reg_loss.new_zeros(())
        if self.temporal_consistency_enabled:
            if augmented_cls_logits is None or augmented_offsets is None:
                raise ValueError(
                    "TRC training requires perturbed predictions")
            clean_cls = torch.cat(out_cls_logits, dim=1)[valid_mask]
            augmented_cls = torch.cat(
                augmented_cls_logits, dim=1)[valid_mask]
            clean_probability = clean_cls.sigmoid()
            if self.temporal_consistency_teacher_detach:
                clean_probability = clean_probability.detach()
            augmented_probability = augmented_cls.sigmoid()
            midpoint_probability = (
                0.5 * (clean_probability + augmented_probability)
            ).clamp(min=1e-6, max=1.0 - 1e-6)
            clean_probability = clean_probability.clamp(
                min=1e-6, max=1.0 - 1e-6)
            augmented_probability = augmented_probability.clamp(
                min=1e-6, max=1.0 - 1e-6)

            def bernoulli_kl(probability, midpoint):
                return (
                    probability * torch.log(probability / midpoint)
                    + (1.0 - probability) * torch.log(
                        (1.0 - probability) / (1.0 - midpoint)))

            temporal_consistency_cls_loss = 0.5 * (
                bernoulli_kl(clean_probability, midpoint_probability)
                + bernoulli_kl(
                    augmented_probability, midpoint_probability)
            ).sum() / self.loss_normalizer

            augmented_positive_all = torch.cat(
                augmented_offsets, dim=1)[pos_mask]
            if augmented_positive_all.ndim == 3:
                augmented_selected = augmented_positive_all[
                    selected_points, selected_classes]
            else:
                augmented_selected = augmented_positive_all[selected_points]
            clean_regression_target = pred_offsets
            if self.temporal_consistency_teacher_detach:
                clean_regression_target = clean_regression_target.detach()
            temporal_consistency_reg_loss = F.smooth_l1_loss(
                torch.log1p(augmented_selected),
                torch.log1p(clean_regression_target),
                reduction="sum",
            ) / (2.0 * self.loss_normalizer)

        high_iou_margin_loss = reg_loss.new_zeros(())
        endpoint_precision_loss = reg_loss.new_zeros(())
        if self.high_iou_boundary_enabled and num_pos > 0:
            intersection = (
                torch.minimum(pred_offsets[:, 0], gt_offsets_for_reg[:, 0])
                + torch.minimum(
                    pred_offsets[:, 1], gt_offsets_for_reg[:, 1]))
            union = (
                torch.maximum(pred_offsets[:, 0], gt_offsets_for_reg[:, 0])
                + torch.maximum(
                    pred_offsets[:, 1], gt_offsets_for_reg[:, 1])
            ).clamp_min(1e-6)
            temporal_iou = (intersection / union).clamp(0.0, 1.0)
            thresholds = temporal_iou.new_tensor(
                self.high_iou_thresholds)
            shortfall = F.relu(
                thresholds[None] - temporal_iou[:, None])
            high_iou_margin_loss = shortfall.square().sum()
            high_iou_margin_loss /= (
                len(self.high_iou_thresholds) * self.loss_normalizer)

            target_duration = gt_offsets_for_reg.sum(
                dim=-1, keepdim=True).clamp_min(1e-6)
            normalized_prediction = pred_offsets / target_duration
            normalized_target = gt_offsets_for_reg / target_duration
            endpoint_precision_loss = F.smooth_l1_loss(
                normalized_prediction,
                normalized_target,
                reduction="sum",
                beta=self.endpoint_precision_beta,
            ) / (2.0 * self.loss_normalizer)

        diffusion_loss = reg_loss.new_zeros(())
        if self.diffusion_enabled:
            if points is None or reg_features is None:
                raise ValueError("Diffusion training requires points and regression features")
            concat_points = torch.cat(points, dim=0)
            point_center = concat_points[:, 0][None].expand(valid_mask.shape[0], -1)
            point_stride = concat_points[:, 3][None].expand(valid_mask.shape[0], -1)
            positive_centers = point_center[pos_mask][selected_points]
            positive_strides = point_stride[pos_mask][selected_points]
            coarse = torch.stack(
                (positive_centers - pred_offsets[:, 0] * positive_strides,
                 positive_centers + pred_offsets[:, 1] * positive_strides),
                dim=-1
            )
            target = torch.stack(
                (positive_centers
                 - gt_offsets_for_reg[:, 0] * positive_strides,
                 positive_centers
                 + gt_offsets_for_reg[:, 1] * positive_strides), dim=-1
            )
            positive_features = torch.cat(
                reg_features, dim=1)[pos_mask][selected_points]
            level_template = torch.cat([
                torch.full((point.shape[0],), level, device=point.device, dtype=torch.long)
                for level, point in enumerate(points)
            ])
            level_ids = level_template[None].expand(
                valid_mask.shape[0], -1)[pos_mask][selected_points]
            positive_batch_ids = pos_mask.nonzero(
                as_tuple=True)[0][selected_points]
            boundary_context = self._sample_boundary_context(
                reg_features, fpn_masks, coarse,
                positive_batch_ids, level_ids,
            )
            if self.diffusion_refiner.v15c1_enabled:
                (refiner_loss, diffusion_loss, boundary_loss,
                 final_ddim_loss, quality_gate_loss, v15b_tiou_loss,
                 v15b_endpoint_loss, v15c1_identity_loss,
                 v15c1_duration_asym_loss) = \
                    self.diffusion_refiner.training_loss(
                        coarse, target, positive_features, level_ids,
                        return_v15c1_components=True,
                        boundary_context=boundary_context,
                    )
            elif self.diffusion_refiner.v15b_enabled:
                (refiner_loss, diffusion_loss, boundary_loss,
                 final_ddim_loss, quality_gate_loss, v15b_tiou_loss,
                 v15b_endpoint_loss) = \
                    self.diffusion_refiner.training_loss(
                        coarse, target, positive_features, level_ids,
                        return_v15b_components=True,
                        boundary_context=boundary_context,
                    )
            else:
                (refiner_loss, diffusion_loss, boundary_loss,
                 final_ddim_loss, quality_gate_loss) = \
                    self.diffusion_refiner.training_loss(
                        coarse, target, positive_features, level_ids,
                        return_components=True,
                        boundary_context=boundary_context,
                    )
                v15b_tiou_loss = diffusion_loss.new_zeros(())
                v15b_endpoint_loss = diffusion_loss.new_zeros(())
                v15c1_identity_loss = diffusion_loss.new_zeros(())
                v15c1_duration_asym_loss = diffusion_loss.new_zeros(())
            if not self.diffusion_refiner.v15c1_enabled:
                v15c1_identity_loss = diffusion_loss.new_zeros(())
                v15c1_duration_asym_loss = diffusion_loss.new_zeros(())
        else:
            refiner_loss = diffusion_loss
            boundary_loss = diffusion_loss.new_zeros(())
            final_ddim_loss = diffusion_loss.new_zeros(())
            quality_gate_loss = diffusion_loss.new_zeros(())
            v15b_tiou_loss = diffusion_loss.new_zeros(())
            v15b_endpoint_loss = diffusion_loss.new_zeros(())
            v15c1_identity_loss = diffusion_loss.new_zeros(())
            v15c1_duration_asym_loss = diffusion_loss.new_zeros(())

        final_loss = (
            cls_loss + reg_loss * loss_weight
            + self.diffusion_weight * refiner_loss
            + self.start_end_distribution_weight * distribution_loss
            + self.start_end_tiou_weight * start_end_tiou_loss
            + self.boundary_proposal_distribution_weight
            * proposal_distribution_loss
            + self.boundary_proposal_coverage_weight
            * proposal_coverage_loss
            + self.boundary_proposal_improvement_weight
            * proposal_improvement_loss
            + self.trident_aux_weight * trident_aux_loss
            + self.temporal_consistency_weight * (
                self.temporal_consistency_cls_weight
                * temporal_consistency_cls_loss
                + self.temporal_consistency_reg_weight
                * temporal_consistency_reg_loss)
            + self.high_iou_margin_weight * high_iou_margin_loss
            + self.endpoint_precision_weight * endpoint_precision_loss
            + self.dense_iou_quality_weight * dense_iou_quality_loss
            + self.dense_iou_calibration_weight
            * dense_iou_calibration_loss
            + self.dense_iou_pairwise_weight * dense_iou_pairwise_loss
            + self.dense_iou_listwise_weight * dense_iou_listwise_loss
            + self.task_alignment_pairwise_weight
            * task_alignment_pairwise_loss
            + self.residual_alignment_cls_consistency_weight
            * residual_cls_consistency_loss
            + self.residual_alignment_reg_consistency_weight
            * residual_reg_consistency_loss
            + self.residual_alignment_magnitude_weight
            * residual_magnitude_loss
            + self.residual_alignment_confidence_weight
            * residual_confidence_loss
            + self.cluster_gain_loss_weight * cluster_gain_loss
            + self.cluster_pairwise_loss_weight * cluster_pairwise_loss
            + self.cluster_abstention_loss_weight
            * cluster_abstention_loss
            + self.cluster_score_amount_loss_weight
            * cluster_score_amount_loss
            + self.cluster_score_confidence_loss_weight
            * cluster_score_confidence_loss
        )
        output = {'cls_loss': cls_loss, 'reg_loss': reg_loss,
                  'final_loss': final_loss}
        if self.start_end_distribution_enabled:
            output['start_end_distribution_loss'] = distribution_loss
            output['start_end_tiou_loss'] = start_end_tiou_loss
        if self.boundary_proposal_enabled:
            output['proposal_distribution_loss'] = \
                proposal_distribution_loss
            output['proposal_coverage_loss'] = proposal_coverage_loss
            output['proposal_improvement_loss'] = \
                proposal_improvement_loss
        if self.trident_enabled:
            output['trident_aux_loss'] = trident_aux_loss
        if self.temporal_consistency_enabled:
            output['trc_cls_loss'] = temporal_consistency_cls_loss
            output['trc_reg_loss'] = temporal_consistency_reg_loss
        if self.high_iou_boundary_enabled:
            output['high_iou_margin_loss'] = high_iou_margin_loss
            output['endpoint_precision_loss'] = endpoint_precision_loss
        if self.dense_iou_quality_enabled:
            output['dense_iou_quality_loss'] = dense_iou_quality_loss
            output['dense_iou_target_mean'] = dense_iou_target_mean
            output['dense_iou_calibration_loss'] = \
                dense_iou_calibration_loss
            if self.dense_iou_ranking_enabled:
                output['dense_iou_pairwise_loss'] = \
                    dense_iou_pairwise_loss
                output['dense_iou_listwise_loss'] = \
                    dense_iou_listwise_loss
        if self.task_alignment_enabled:
            output['task_alignment_pairwise_loss'] = \
                task_alignment_pairwise_loss
            output['task_alignment_mean'] = task_alignment_mean
            output['task_alignment_quality_mean'] = \
                task_alignment_quality_mean
        if self.residual_alignment_enabled:
            output['residual_cls_consistency_loss'] = \
                residual_cls_consistency_loss
            output['residual_reg_consistency_loss'] = \
                residual_reg_consistency_loss
            output['residual_magnitude_loss'] = \
                residual_magnitude_loss
            output['residual_confidence_loss'] = \
                residual_confidence_loss
            output['residual_cls_positive_ratio'] = \
                residual_cls_positive_ratio
            output['residual_reg_positive_ratio'] = \
                residual_reg_positive_ratio
        if self.cluster_selector_enabled:
            output['cluster_selector_gain_loss'] = cluster_gain_loss
            output['cluster_selector_pairwise_loss'] = \
                cluster_pairwise_loss
            output['cluster_selector_abstention_loss'] = \
                cluster_abstention_loss
            if self.cluster_score_suppression_enabled:
                output['cluster_score_suppression_amount_loss'] = \
                    cluster_score_amount_loss
                output['cluster_score_suppression_confidence_loss'] = \
                    cluster_score_confidence_loss
            output.update(cluster_training_stats)
        if self.diffusion_enabled:
            output['diffusion_loss'] = diffusion_loss
            output['boundary_loss'] = boundary_loss
            output['final_ddim_loss'] = final_ddim_loss
            output['quality_gate_loss'] = quality_gate_loss
            output['refiner_loss'] = refiner_loss
            if self.diffusion_refiner.v15b_enabled:
                output['v15b_tiou_loss'] = v15b_tiou_loss
                output['v15b_endpoint_loss'] = v15b_endpoint_loss
            if self.diffusion_refiner.v15c1_enabled:
                output['v15c1_identity_loss'] = v15c1_identity_loss
                output['v15c1_duration_asym_loss'] = \
                    v15c1_duration_asym_loss
        return output

    @torch.no_grad()
    def inference(
            self,
            video_list,
            points, fpn_masks,
            out_cls_logits, out_offsets,
            reg_features=None,
            boundary_proposal_candidates=None,
            boundary_proposal_improvement=None,
            quality_logits=None,
            residual_cls_values=None,
            residual_reg_values=None,
            residual_cls_confidence=None,
            residual_reg_confidence=None,
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
                zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            reg_features_per_vid = (
                [x[idx] for x in reg_features] if reg_features is not None else None
            )
            proposal_candidates_per_vid = (
                [x[idx] for x in boundary_proposal_candidates]
                if boundary_proposal_candidates is not None else None)
            proposal_improvement_per_vid = (
                [x[idx] for x in boundary_proposal_improvement]
                if boundary_proposal_improvement is not None else None)
            quality_logits_per_vid = (
                [x[idx] for x in quality_logits]
                if quality_logits is not None else None)
            residual_cls_values_per_vid = (
                [x[idx] for x in residual_cls_values]
                if residual_cls_values is not None else None)
            residual_reg_values_per_vid = (
                [x[idx] for x in residual_reg_values]
                if residual_reg_values is not None else None)
            residual_cls_confidence_per_vid = (
                [x[idx] for x in residual_cls_confidence]
                if residual_cls_confidence is not None else None)
            residual_reg_confidence_per_vid = (
                [x[idx] for x in residual_reg_confidence]
                if residual_reg_confidence is not None else None)
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]

            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid, reg_features_per_vid,
                proposal_candidates_per_vid,
                proposal_improvement_per_vid,
                quality_logits_per_vid,
                residual_cls_values_per_vid,
                residual_reg_values_per_vid,
                residual_cls_confidence_per_vid,
                residual_reg_confidence_per_vid,
            )
            if self.diffusion_enabled:
                refine_indices = results_per_vid['_refine_indices']
                coarse = results_per_vid['coarse_segments'][refine_indices]
                refined = results_per_vid['segments'][refine_indices]
                if coarse.shape[0] > 0:
                    inter = (torch.minimum(coarse[:, 1], refined[:, 1])
                             - torch.maximum(coarse[:, 0], refined[:, 0])).clamp_min(0)
                    union = ((coarse[:, 1] - coarse[:, 0])
                             + (refined[:, 1] - refined[:, 0]) - inter).clamp_min(1e-6)
                    results_per_vid['refinement_stats']['mean_coarse_refined_tiou'] = float((inter / union).mean().item())
                else:
                    results_per_vid['refinement_stats']['mean_coarse_refined_tiou'] = 0.0
                if self.oracle_diagnostic_enabled:
                    self._attach_oracle_boundaries(
                        results_per_vid, video_list[idx])
            if self.v13_phase_a_enabled:
                gt_segments = video_list[idx].get('segments')
                gt_labels = video_list[idx].get('labels')
                if gt_segments is None or gt_labels is None:
                    raise ValueError(
                        "V13 Phase A requires ground-truth segments and labels")
                results_per_vid['_v13_gt_segments'] = gt_segments
                results_per_vid['_v13_gt_labels'] = gt_labels
            if self.v14_probe_enabled:
                gt_segments = video_list[idx].get('segments')
                gt_labels = video_list[idx].get('labels')
                if gt_segments is None or gt_labels is None:
                    raise ValueError(
                        "V14 learnability probes require ground-truth "
                        "segments and labels")
                results_per_vid['_v14_gt_segments'] = gt_segments
                results_per_vid['_v14_gt_labels'] = gt_labels
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        # step 3: postprocssing
        results = self.postprocessing(results)

        return results

    @staticmethod
    def _attach_oracle_boundaries(results, sample, tolerance=1e-6):
        """Select refined boundaries using class-matched GT for diagnostics.

        This is an analysis upper bound only. It is guarded by
        ``oracle_diagnostic_enabled`` and the normal inference path never
        calls it.
        """
        gt_segments = sample.get('segments')
        gt_labels = sample.get('labels')
        if gt_segments is None or gt_labels is None:
            raise ValueError(
                "Oracle diagnostic requires ground-truth segments and labels")
        coarse = results['coarse_segments']
        refined_candidates = results.get(
            'oracle_candidate_segments', results['segments'][None])
        if refined_candidates.ndim != 3:
            raise ValueError(
                "oracle_candidate_segments must have shape [K, N, 2]")
        refine_indices = results['_refine_indices']
        oracle = coarse.clone()
        accepts = torch.zeros(
            refine_indices.shape[0], dtype=torch.bool, device=coarse.device)
        improvements = coarse.new_zeros(refine_indices.shape[0])
        if refine_indices.numel() > 0 and gt_segments.numel() > 0:
            candidate_labels = results['labels'][refine_indices]
            for position, (candidate_index, candidate_label) in enumerate(
                    zip(refine_indices, candidate_labels)):
                matches = gt_labels == candidate_label
                if not bool(matches.any()):
                    continue
                targets = gt_segments[matches]

                def best_tiou(segment):
                    intersection = (
                        torch.minimum(segment[1], targets[:, 1])
                        - torch.maximum(segment[0], targets[:, 0])
                    ).clamp_min(0.0)
                    union = (
                        (segment[1] - segment[0]).clamp_min(0.0)
                        + (targets[:, 1] - targets[:, 0]).clamp_min(0.0)
                        - intersection
                    ).clamp_min(1e-6)
                    return (intersection / union).max()

                candidate_qualities = torch.stack([
                    best_tiou(candidate[candidate_index])
                    for candidate in refined_candidates
                ])
                coarse_quality = best_tiou(coarse[candidate_index])
                best_quality, best_index = candidate_qualities.max(dim=0)
                improvement = best_quality - coarse_quality
                improvements[position] = improvement
                accepts[position] = improvement > tolerance
                if accepts[position]:
                    oracle[candidate_index] = refined_candidates[
                        best_index, candidate_index]
        accepted_improvements = improvements[accepts]
        results['oracle_segments'] = oracle
        results['oracle_scores'] = results['scores'].clone()
        results['oracle_labels'] = results['labels'].clone()
        stats = results['refinement_stats']
        stats['oracle_candidates'] = int(refine_indices.numel())
        stats['oracle_num_candidates_per_proposal'] = int(
            refined_candidates.shape[0])
        stats['oracle_generated_candidates'] = int(
            refine_indices.numel() * refined_candidates.shape[0])
        stats['oracle_accepts'] = int(accepts.sum().item())
        stats['oracle_accept_ratio'] = float(accepts.float().mean().item()) \
            if accepts.numel() else 0.0
        stats['oracle_mean_accepted_delta_tiou'] = float(
            accepted_improvements.mean().item()) \
            if accepted_improvements.numel() else 0.0
        stats['oracle_accepted_delta_tiou_sum'] = float(
            accepted_improvements.sum().item())
        selected_candidates = refined_candidates[:, refine_indices]
        pair_indices = torch.triu_indices(
            refined_candidates.shape[0], refined_candidates.shape[0],
            offset=1, device=refined_candidates.device)
        pairwise_tious = None
        if pair_indices.shape[1] > 0 and selected_candidates.shape[1] > 0:
            first_segments = selected_candidates[pair_indices[0]]
            second_segments = selected_candidates[pair_indices[1]]
            intersection = (
                torch.minimum(first_segments[..., 1], second_segments[..., 1])
                - torch.maximum(first_segments[..., 0], second_segments[..., 0])
            ).clamp_min(0.0)
            union = (
                (first_segments[..., 1] - first_segments[..., 0]).clamp_min(0.0)
                + (second_segments[..., 1] - second_segments[..., 0]).clamp_min(0.0)
                - intersection
            ).clamp_min(1e-6)
            pairwise_tious = intersection / union
        stats['oracle_mean_pairwise_candidate_tiou'] = float(
            pairwise_tious.mean().item()) \
            if pairwise_tious is not None and pairwise_tious.numel() else 1.0
        stats['oracle_uses_ground_truth'] = 1.0

    @torch.no_grad()
    def _conservative_select_existing_boundaries(
            self, segments, scores, labels, quality_logits):
        """Replace score-ranked anchors only with safer existing boundaries.

        The original anchor score and label are never changed. Each anchor is
        also an implicit candidate, so a replacement is accepted only when an
        overlapping, same-class proposal predicts a sufficient quality gain.
        """
        count = int(segments.shape[0])
        stats = {
            "candidate_pool_size": count,
            "selection_anchors": 0,
            "selection_replacements": 0,
            "selection_replacement_ratio": 0.0,
            "selection_mean_predicted_gain": 0.0,
        }
        if count == 0:
            return segments, stats
        if quality_logits.shape != scores.shape:
            raise ValueError(
                "candidate selection quality logits must match scores")
        order = scores.argsort(descending=True)
        anchor_count = min(
            self.conservative_selection_anchor_topk, count)
        candidate_count = min(
            self.conservative_selection_candidate_topk, count)
        anchors = order[:anchor_count]
        candidates = order[:candidate_count]
        stats["selection_anchors"] = anchor_count

        anchor_segments = segments[anchors]
        candidate_segments = segments[candidates]
        overlap = self._segment_tiou(
            anchor_segments, candidate_segments)
        same_class = (
            labels[anchors, None] == labels[candidates][None])
        score_floor = (
            scores[anchors, None]
            * self.conservative_selection_min_score_ratio)
        score_safe = scores[candidates][None] >= score_floor
        gains = (
            quality_logits[candidates][None]
            - quality_logits[anchors, None])
        eligible = (
            (overlap >= self.conservative_selection_overlap)
            & same_class & score_safe
            & (gains > self.conservative_selection_min_gain))
        masked_gains = gains.masked_fill(~eligible, float("-inf"))
        best_gain, best_position = masked_gains.max(dim=1)
        replace = torch.isfinite(best_gain)
        selected = segments.clone()
        if bool(replace.any()):
            selected[anchors[replace]] = candidate_segments[
                best_position[replace]]
        accepted_gain = best_gain[replace]
        replacement_count = int(replace.sum().item())
        stats.update({
            "selection_replacements": replacement_count,
            "selection_replacement_ratio": (
                replacement_count / max(anchor_count, 1)),
            "selection_mean_predicted_gain": float(
                accepted_gain.mean().item())
            if accepted_gain.numel() else 0.0,
        })
        return selected, stats

    @torch.no_grad()
    def _cluster_select_existing_boundaries(
            self, segments, scores, labels, proposal_features,
            proposal_levels, regression_features, fpn_masks):
        """Select at most one existing boundary from every local cluster."""
        candidate_count = int(segments.shape[0])
        stats = {
            "p2d1_candidate_pool_size": candidate_count,
            "p2d1_cluster_count": 0,
            "p2d1_non_singleton_clusters": 0,
            "p2d1_eligible_replacements": 0,
            "p2d1_replacements": 0,
            "p2d1_replacement_ratio": 0.0,
            "p2d1_mean_predicted_gain": 0.0,
            "p2d1_mean_replace_confidence": 0.0,
            "p2d1_cluster_threshold": self.cluster_overlap_threshold,
            "p2d1_max_replacement_fraction":
                self.cluster_max_replacement_fraction,
            "p5_ap_listwise_enabled":
                self.cluster_selector_type == "ap_listwise",
            "p5_mean_predicted_ap_utility": 0.0,
            "p4a_score_suppression_enabled":
                self.cluster_score_suppression_enabled,
            "p4a_score_suppression_eligible": 0,
            "p4a_score_suppressions": 0,
            "p4a_score_suppression_ratio": 0.0,
            "p4a_mean_score_logit_suppression": 0.0,
            "p4a_mean_score_suppression_confidence": 0.0,
            "p4a_max_score_logit_suppression":
                self.cluster_score_max_logit_suppression,
            "p4a_max_score_suppression_fraction":
                self.cluster_score_max_suppression_fraction,
        }
        if candidate_count == 0 or not self.cluster_selection_enabled:
            return (
                segments, scores, labels, stats,
                segments.clone(),
                torch.zeros(
                    candidate_count, dtype=torch.bool,
                    device=segments.device),
            )
        clusters = self._greedy_cluster_indices(
            segments, scores, labels)
        anchors = torch.stack([members[0] for members in clusters])
        anchor_segments = segments[anchors].clone()
        selected_segments = anchor_segments.clone()
        selected_scores = scores[anchors].clone()
        selected_labels = labels[anchors].clone()
        replacement_mask = torch.zeros(
            len(clusters), dtype=torch.bool, device=segments.device)
        stats["p2d1_cluster_count"] = len(clusters)
        stats["p2d1_non_singleton_clusters"] = sum(
            int(members.numel() > 1) for members in clusters)
        replacement_active = (
            self.cluster_replacement_enabled
            and self.cluster_max_replacement_fraction > 0.0)
        suppression_active = (
            self.cluster_score_suppression_enabled
            and self.cluster_score_max_suppression_fraction > 0.0)
        if not replacement_active and not suppression_active:
            return (
                selected_segments, selected_scores, selected_labels,
                stats, anchor_segments, replacement_mask)

        descriptors = self._cluster_boundary_descriptors(
            regression_features, fpn_masks, segments, proposal_levels)
        decisions = []
        if replacement_active:
            for cluster_index, members in enumerate(clusters):
                if members.numel() < 2:
                    continue
                members = members[:self.cluster_max_members]
                anchor = int(members[0].item())
                if self.cluster_selector_type == "ap_listwise":
                    evaluated = members
                else:
                    evaluated = members[1:]
                repeated_anchor = torch.full_like(evaluated, anchor)
                geometry = self._cluster_pair_geometry(
                    segments[repeated_anchor], segments[evaluated],
                    scores[repeated_anchor], scores[evaluated])
                predicted_gain, confidence_logit = \
                    self.cluster_selector_head(
                        proposal_features[repeated_anchor],
                        proposal_features[evaluated],
                        descriptors[repeated_anchor],
                        descriptors[evaluated],
                        geometry,
                        proposal_levels[repeated_anchor],
                        proposal_levels[evaluated])
                if self.cluster_selector_type == "ap_listwise":
                    candidates = evaluated[1:]
                    predicted_gain = (
                        predicted_gain[1:] - predicted_gain[0])
                    confidence_logit = confidence_logit[1:]
                else:
                    candidates = evaluated
                confidence = confidence_logit.sigmoid()
                eligible = (
                    (predicted_gain > self.cluster_min_predicted_gain)
                    & (confidence >= self.cluster_min_confidence))
                if not bool(eligible.any()):
                    continue
                eligible_positions = torch.nonzero(
                    eligible, as_tuple=True)[0]
                best_local = eligible_positions[
                    (predicted_gain[eligible_positions]
                     * confidence[eligible_positions]).argmax()]
                priority = (
                    predicted_gain[best_local]
                    * confidence[best_local]
                    if self.cluster_selector_type == "ap_listwise"
                    else predicted_gain[best_local])
                decisions.append((
                    float(priority.item()),
                    float(predicted_gain[best_local].item()),
                    float(confidence[best_local].item()),
                    cluster_index,
                    int(candidates[best_local].item()),
                ))
            stats["p2d1_eligible_replacements"] = len(decisions)
            budget = int(math.floor(
                len(clusters) * self.cluster_max_replacement_fraction))
            decisions.sort(key=lambda item: item[0], reverse=True)
            accepted = decisions[:budget]
            for (_, gain, confidence, cluster_index,
                 candidate_index) in accepted:
                selected_segments[cluster_index] = segments[candidate_index]
                replacement_mask[cluster_index] = True
            replacement_count = len(accepted)
            stats.update({
                "p2d1_replacements": replacement_count,
                "p2d1_replacement_ratio": (
                    replacement_count / max(len(clusters), 1)),
                "p2d1_mean_predicted_gain": (
                    sum(item[1] for item in accepted)
                    / replacement_count if replacement_count else 0.0),
                "p2d1_mean_replace_confidence": (
                    sum(item[2] for item in accepted)
                    / replacement_count if replacement_count else 0.0),
                "p5_mean_predicted_ap_utility": (
                    sum(item[1] for item in accepted)
                    / replacement_count
                    if (replacement_count
                        and self.cluster_selector_type == "ap_listwise")
                    else 0.0),
            })

        if suppression_active:
            anchor_durations = (
                segments[anchors, 1] - segments[anchors, 0]
            ).clamp_min(1e-6)
            amount, confidence_logit = \
                self.cluster_score_suppression_head(
                    proposal_features[anchors], descriptors[anchors],
                    scores[anchors], anchor_durations,
                    proposal_levels[anchors])
            confidence = confidence_logit.sigmoid()
            eligible = (
                (amount >= self.cluster_score_min_suppression)
                & (confidence >= self.cluster_score_min_confidence))
            eligible_indices = torch.nonzero(
                eligible, as_tuple=True)[0]
            stats["p4a_score_suppression_eligible"] = int(
                eligible_indices.numel())
            suppression_budget = int(math.floor(
                len(clusters)
                * self.cluster_score_max_suppression_fraction))
            if eligible_indices.numel() and suppression_budget > 0:
                priority = (
                    amount[eligible_indices]
                    * confidence[eligible_indices])
                selected_local = eligible_indices[
                    priority.argsort(descending=True, stable=True)
                    [:suppression_budget]]
                original = selected_scores[selected_local].clamp(
                    min=1e-6, max=1.0 - 1e-6)
                selected_scores[selected_local] = torch.sigmoid(
                    torch.logit(original) - amount[selected_local])
                suppression_count = int(selected_local.numel())
                stats.update({
                    "p4a_score_suppressions": suppression_count,
                    "p4a_score_suppression_ratio": (
                        suppression_count / max(len(clusters), 1)),
                    "p4a_mean_score_logit_suppression": float(
                        amount[selected_local].mean().item()),
                    "p4a_mean_score_suppression_confidence": float(
                        confidence[selected_local].mean().item()),
                })
        return (
            selected_segments, selected_scores, selected_labels,
            stats, anchor_segments, replacement_mask)

    @torch.no_grad()
    def inference_single_video(
            self,
            points,
            fpn_masks,
            out_cls_logits,
            out_offsets,
            reg_features=None,
            boundary_proposal_candidates=None,
            boundary_proposal_improvement=None,
            quality_logits=None,
            residual_cls_values=None,
            residual_reg_values=None,
            residual_cls_confidence=None,
            residual_reg_confidence=None,
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []
        reg_features_all = []
        level_ids_all = []
        score_features_all = []
        proposal_segs_all = []
        proposal_scores_all = []
        proposal_labels_all = []
        proposal_candidates_considered = 0
        proposal_candidates_accepted = 0
        selection_quality_all = []
        residual_alignment_stats = None
        if self.residual_alignment_enabled:
            if any(value is None for value in (
                    residual_cls_values, residual_reg_values,
                    residual_cls_confidence,
                    residual_reg_confidence)):
                raise ValueError(
                    "residual alignment inference tensors are missing")
            valid_class_residual = torch.cat([
                value[mask] for value, mask in zip(
                    residual_cls_values, fpn_masks)], dim=0)
            valid_reg_residual = torch.cat([
                value[mask] for value, mask in zip(
                    residual_reg_values, fpn_masks)], dim=0)
            valid_class_confidence = torch.cat([
                value[mask] for value, mask in zip(
                    residual_cls_confidence, fpn_masks)], dim=0)
            valid_reg_confidence = torch.cat([
                value[mask] for value, mask in zip(
                    residual_reg_confidence, fpn_masks)], dim=0)
            minimum = self.residual_alignment_head.minimum_confidence
            class_active = (
                valid_class_confidence.sigmoid() >= minimum)
            reg_active = (
                valid_reg_confidence.sigmoid() >= minimum)
            if self.residual_alignment_head.residual_weight == 0.0:
                class_active.zero_()
                reg_active.zero_()
            residual_alignment_stats = {
                "p4b_residual_weight": float(
                    self.residual_alignment_head.residual_weight),
                "p4b_minimum_confidence": float(minimum),
                "p4b_valid_class_predictions": int(
                    valid_class_residual.numel()),
                "p4b_valid_regression_predictions": int(
                    valid_reg_residual.shape[0]),
                "p4b_active_class_residuals": int(
                    class_active.sum().item()),
                "p4b_active_regression_residuals": int(
                    reg_active.sum().item()),
                "p4b_class_active_ratio": float(
                    class_active.to(torch.float32).mean().item())
                    if class_active.numel() else 0.0,
                "p4b_regression_active_ratio": float(
                    reg_active.to(torch.float32).mean().item())
                    if reg_active.numel() else 0.0,
                "p4b_mean_absolute_class_residual": float(
                    valid_class_residual.abs().mean().item())
                    if valid_class_residual.numel() else 0.0,
                "p4b_mean_absolute_regression_residual": float(
                    valid_reg_residual.abs().mean().item())
                    if valid_reg_residual.numel() else 0.0,
                "p4b_uses_ground_truth_during_inference": False,
            }

        # loop over fpn levels
        for level, (cls_i, offsets_i, pts_i, mask_i) in enumerate(zip(
                out_cls_logits, out_offsets, points, fpn_masks)):
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]
            
            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs = torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)

            # 3. For efficiency, pad the boarder head with num_bins zeros (Pad left for start branch and Pad right
            # for end branch). Then we re-arrange the output of boundary branch to [class_num, T, num_bins + 1 (the
            # neighbour bin for each instant)]. In this way, the output can be directly added to the center offset
            # later.
            
            # pick topk output from the prediction
            if offsets_i.ndim == 3:
                # V11 Trident offsets are class-aware: [T, C, 2].
                offsets = offsets_i[pt_idxs, cls_idxs]
            else:
                offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]
            
            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]

            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh
            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])
            if quality_logits is not None:
                selected_quality = quality_logits[
                    level][pt_idxs, cls_idxs]
                selection_quality_all.append(
                    selected_quality[keep_idxs2])
            if self.v14_probe_enabled or self.cluster_selector_enabled:
                if reg_features is None:
                    raise ValueError(
                        "proposal diagnostics/selectors require regression "
                        "features")
                selected_features = reg_features[level][pt_idxs]
                reg_features_all.append(selected_features[keep_idxs2])
                level_ids_all.append(torch.full(
                    (int(keep_idxs2.sum().item()),), level,
                    dtype=torch.long, device=pred_segs.device))
                if self.v14_probe_enabled:
                    point_probabilities = cls_i.sigmoid()[pt_idxs]
                    selected_probability = point_probabilities.gather(
                        1, cls_idxs[:, None]).squeeze(1)
                    if self.num_classes > 1:
                        competitors = point_probabilities.clone()
                        competitors.scatter_(1, cls_idxs[:, None], -1.0)
                        competitor_probability = \
                            competitors.max(dim=1).values
                    else:
                        competitor_probability = torch.zeros_like(
                            selected_probability)
                    binary_entropy = -(
                        point_probabilities.clamp_min(1e-6).log()
                        * point_probabilities
                        + (1.0 - point_probabilities).clamp_min(1e-6).log()
                        * (1.0 - point_probabilities)
                    ).mean(dim=1)
                    score_features_all.append(torch.stack((
                        selected_probability,
                        competitor_probability,
                        selected_probability - competitor_probability,
                        binary_entropy,
                    ), dim=1)[keep_idxs2])
            if boundary_proposal_candidates is not None:
                candidate_offsets = boundary_proposal_candidates[
                    level][pt_idxs][keep_idxs2]
                candidate_quality = boundary_proposal_improvement[
                    level][pt_idxs][keep_idxs2].sigmoid()
                kept_points = pts[keep_idxs2]
                kept_scores = pred_prob[keep_idxs2]
                kept_labels = cls_idxs[keep_idxs2]
                candidate_left = (
                    kept_points[:, None, 0]
                    - candidate_offsets[..., 0]
                    * kept_points[:, None, 3])
                candidate_right = (
                    kept_points[:, None, 0]
                    + candidate_offsets[..., 1]
                    * kept_points[:, None, 3])
                candidate_segments = torch.stack(
                    (candidate_left, candidate_right), dim=-1)
                valid_candidate = (
                    candidate_right - candidate_left
                    > self.test_duration_thresh)
                accepted = (
                    candidate_quality
                    >= self.boundary_proposal_accept_threshold
                ) & valid_candidate
                proposal_candidates_considered += int(
                    accepted.numel())
                proposal_candidates_accepted += int(
                    accepted.sum().item())
                if bool(accepted.any()):
                    normalized_advantage = (
                        candidate_quality
                        - self.boundary_proposal_accept_threshold
                    ) / (1.0 - self.boundary_proposal_accept_threshold)
                    score_factor = 1.0 + (
                        self.boundary_proposal_max_score_boost
                        * normalized_advantage.clamp(min=0.0, max=1.0))
                    proposal_segs_all.append(
                        candidate_segments[accepted])
                    proposal_scores_all.append(
                        (kept_scores[:, None] * score_factor)[accepted]
                        .clamp(max=1.0))
                    proposal_labels_all.append(
                        kept_labels[:, None].expand_as(
                            candidate_quality)[accepted])
            if self.diffusion_enabled:
                if reg_features is None:
                    raise ValueError("Diffusion inference requires regression features")
                if (not self.v14_probe_enabled
                        and not self.cluster_selector_enabled):
                    selected_features = reg_features[level][pt_idxs]
                    reg_features_all.append(selected_features[keep_idxs2])
                    level_ids_all.append(torch.full(
                        (int(keep_idxs2.sum().item()),), level,
                        dtype=torch.long, device=pred_segs.device
                    ))

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        coarse_segs = segs_all.clone()
        coarse_scores = scores_all.clone()
        candidate_selection_stats = None
        p2d1_anchor_segments = None
        p2d1_replacement_mask = None
        if self.cluster_selector_enabled:
            if not reg_features_all:
                raise ValueError(
                    "cluster selector requires collected proposal features")
            candidate_features = torch.cat(reg_features_all)
            candidate_levels = torch.cat(level_ids_all)
            (segs_all, scores_all, cls_idxs_all,
             candidate_selection_stats, p2d1_anchor_segments,
             p2d1_replacement_mask) = \
                self._cluster_select_existing_boundaries(
                    segs_all, scores_all, cls_idxs_all,
                    candidate_features, candidate_levels,
                    reg_features, fpn_masks)
        if self.conservative_selection_enabled:
            if not selection_quality_all:
                raise ValueError(
                    "conservative selection requires quality logits")
            selection_quality = torch.cat(selection_quality_all)
            segs_all, candidate_selection_stats = \
                self._conservative_select_existing_boundaries(
                    segs_all, scores_all, cls_idxs_all,
                    selection_quality)
        if self.v14_probe_enabled:
            candidate_features = torch.cat(reg_features_all)
            candidate_levels = torch.cat(level_ids_all)
            candidate_score_features = torch.cat(score_features_all)
            maximum = min(
                int(self.v14_probe_max_proposals), coarse_scores.numel())
            selected = coarse_scores.argsort(descending=True)[:maximum]
            v14_probe_values = {
                '_v14_all_segments': coarse_segs.clone(),
                '_v14_all_scores': coarse_scores.clone(),
                '_v14_all_labels': cls_idxs_all.clone(),
                '_v14_segments': coarse_segs[selected].clone(),
                '_v14_scores': coarse_scores[selected].clone(),
                '_v14_labels': cls_idxs_all[selected].clone(),
                '_v14_reg_features': candidate_features[selected].clone(),
                '_v14_level_ids': candidate_levels[selected].clone(),
                '_v14_score_features':
                    candidate_score_features[selected].clone(),
            }
        if proposal_segs_all:
            segs_all = torch.cat((segs_all, *proposal_segs_all), dim=0)
            scores_all = torch.cat(
                (scores_all, *proposal_scores_all), dim=0)
            cls_idxs_all = torch.cat(
                (cls_idxs_all, *proposal_labels_all), dim=0)
        illegal_count = 0
        fallback_count = 0
        refined_count = 0
        multi_candidate_sets = []
        ranker_stats = {}
        ranker_coarse_context = None
        ranker_candidate_context = None
        ranker_synthetic_candidates = None
        ranker_synthetic_context = None
        refine_indices = torch.empty(0, dtype=torch.long, device=segs_all.device)
        if self.diffusion_enabled and segs_all.shape[0] > 0:
            candidate_features = torch.cat(reg_features_all)
            candidate_levels = torch.cat(level_ids_all)
            effective_refine_topk = self.refine_topk
            if (self.candidate_ranker_training_data
                    and not self.candidate_ranker_profile_discovery
                    and self.diffusion_refiner.candidate_ranker_target_type
                    == "oracle_distilled_utility"):
                effective_refine_topk = self.candidate_ranker_train_topk
            refined_count = min(effective_refine_topk, segs_all.shape[0])
            refine_indices = scores_all.argsort(descending=True)[:refined_count]
            valid_end = float(fpn_masks[0].sum().item())
            boundary_context = self._sample_boundary_context(
                [features[None] for features in reg_features],
                [mask[None] for mask in fpn_masks],
                segs_all[refine_indices],
                torch.zeros_like(candidate_levels[refine_indices]),
                candidate_levels[refine_indices],
            )
            refined, illegal_count, gate_stats = self.diffusion_refiner.refine(
                segs_all[refine_indices], candidate_features[refine_indices],
                candidate_levels[refine_indices], valid_range=(0.0, valid_end),
                return_stats=True,
                boundary_context=boundary_context,
            )
            primary_segments = segs_all.clone()
            primary_segments[refine_indices] = refined
            candidate_count = 1
            active_profiles = []
            if self.oracle_diagnostic_enabled and self.oracle_candidate_profiles:
                active_profiles = self.oracle_candidate_profiles
            elif (self.diffusion_refiner.candidate_ranker_enabled
                  and self.diffusion_refiner.candidate_ranker_target_type
                  in ("profiled_relative_iou",
                      "oracle_distilled_utility")):
                active_profiles = (
                    self.diffusion_refiner.candidate_ranker_profile_pool
                    if (self.candidate_ranker_training_data
                        and self.candidate_ranker_profile_discovery)
                    else self.diffusion_refiner.selected_candidate_profiles())
            if active_profiles:
                for profile in active_profiles:
                    candidate, _ = self.diffusion_refiner.refine(
                        segs_all[refine_indices],
                        candidate_features[refine_indices],
                        candidate_levels[refine_indices],
                        valid_range=(0.0, valid_end),
                        boundary_context=boundary_context,
                        inference_seed=profile['seed'],
                        num_inference_steps=profile['steps'],
                        residual_blend_alpha=profile['alpha'],
                        max_relative_shift=profile.get('max_relative_shift'),
                    )
                    candidate_segments = segs_all.clone()
                    candidate_segments[refine_indices] = candidate
                    multi_candidate_sets.append(candidate_segments)
            elif self.oracle_diagnostic_enabled:
                candidate_count = max(
                    candidate_count, self.oracle_num_candidates)
            if self.diffusion_refiner.candidate_ranker_enabled:
                candidate_count = max(
                    candidate_count,
                    self.diffusion_refiner.candidate_ranker_num_candidates)
            if not multi_candidate_sets and candidate_count > 1:
                multi_candidate_sets.append(primary_segments)
                for candidate_number in range(1, candidate_count):
                    additional, _ = self.diffusion_refiner.refine(
                        segs_all[refine_indices],
                        candidate_features[refine_indices],
                        candidate_levels[refine_indices],
                        valid_range=(0.0, valid_end),
                        boundary_context=boundary_context,
                        inference_seed=(
                            self.diffusion_refiner.inference_seed
                            + candidate_number),
                    )
                    candidate_segments = segs_all.clone()
                    candidate_segments[refine_indices] = additional
                    multi_candidate_sets.append(candidate_segments)
            elif (not multi_candidate_sets and (
                  self.oracle_diagnostic_enabled
                  or self.candidate_ranker_training_data
                  or self.diffusion_refiner.candidate_ranker_enabled)):
                multi_candidate_sets.append(primary_segments)
            candidate_tensor = None
            if multi_candidate_sets:
                candidate_tensor = torch.stack(
                    multi_candidate_sets, dim=0
                )[:, refine_indices].permute(1, 0, 2)
            if (self.diffusion_refiner.candidate_ranker_enabled
                    and self.diffusion_refiner.candidate_ranker_context_enabled
                    and not self.candidate_ranker_profile_discovery
                    and candidate_tensor is not None):
                context_offsets = self.diffusion_refiner.\
                    candidate_ranker_context_offsets
                context_features = [
                    features[None] for features in reg_features]
                context_masks = [mask[None] for mask in fpn_masks]
                selected_levels = candidate_levels[refine_indices]
                coarse_samples = self._sample_boundary_context(
                    context_features, context_masks,
                    segs_all[refine_indices],
                    torch.zeros_like(selected_levels), selected_levels,
                    offset_values=context_offsets)
                ranker_coarse_context = \
                    self._boundary_contrast_descriptor(coarse_samples)
                flat_candidates = candidate_tensor.reshape(-1, 2)
                repeated_levels = selected_levels[:, None].expand(
                    -1, candidate_tensor.shape[1]).reshape(-1)
                candidate_samples = self._sample_boundary_context(
                    context_features, context_masks, flat_candidates,
                    torch.zeros_like(repeated_levels), repeated_levels,
                    offset_values=context_offsets)
                ranker_candidate_context = \
                    self._boundary_contrast_descriptor(
                        candidate_samples).reshape(
                            candidate_tensor.shape[0],
                            candidate_tensor.shape[1], -1)
                if (self.candidate_ranker_training_data
                        and self.diffusion_refiner.
                        candidate_ranker_recompute_synthetic_context):
                    pattern_values = candidate_tensor.new_tensor((
                        (-1.0, 1.0), (1.0, -1.0),
                        (-1.0, -1.0), (1.0, 1.0)))
                    pattern_index = torch.arange(
                        candidate_tensor.shape[1],
                        device=candidate_tensor.device) % 4
                    patterns = pattern_values[pattern_index][None]
                    coarse_duration = (
                        segs_all[refine_indices, 1]
                        - segs_all[refine_indices, 0]
                    ).clamp_min(self.diffusion_refiner.min_duration)
                    jitter = (
                        coarse_duration[:, None, None]
                        * self.diffusion_refiner.
                        candidate_ranker_synthetic_jitter_std)
                    synthetic = candidate_tensor + patterns * jitter
                    repeated_coarse = segs_all[refine_indices, None].expand(
                        -1, candidate_tensor.shape[1], -1).reshape(-1, 2)
                    synthetic, _ = self.diffusion_refiner.sanitize(
                        synthetic.reshape(-1, 2), repeated_coarse,
                        valid_range=(0.0, valid_end))
                    ranker_synthetic_candidates = synthetic.reshape_as(
                        candidate_tensor)
                    synthetic_samples = self._sample_boundary_context(
                        context_features, context_masks, synthetic,
                        torch.zeros_like(repeated_levels), repeated_levels,
                        offset_values=context_offsets)
                    ranker_synthetic_context = \
                        self._boundary_contrast_descriptor(
                            synthetic_samples).reshape(
                                candidate_tensor.shape[0],
                                candidate_tensor.shape[1], -1)
            if (self.diffusion_refiner.candidate_ranker_enabled
                    and not self.candidate_ranker_training_data):
                if (self.diffusion_refiner.candidate_ranker_target_type
                        == "oracle_distilled_utility"):
                    selected, score_factors, ranker_stats = \
                        self.diffusion_refiner.\
                        select_oracle_distilled_candidates(
                            segs_all[refine_indices], candidate_tensor,
                            candidate_features[refine_indices],
                            candidate_levels[refine_indices],
                            scores_all[refine_indices],
                            cls_idxs_all[refine_indices],
                            ranker_coarse_context,
                            ranker_candidate_context)
                    scores_all = scores_all.clone()
                    scores_all[refine_indices] = (
                        scores_all[refine_indices] * score_factors
                    ).clamp(max=1.0)
                elif (self.diffusion_refiner.candidate_ranker_target_type
                        == "profiled_relative_iou"):
                    selected, score_factors, ranker_stats = \
                        self.diffusion_refiner.select_profile_ranked_candidates(
                            segs_all[refine_indices], candidate_tensor,
                            candidate_features[refine_indices],
                            candidate_levels[refine_indices],
                            scores_all[refine_indices], ranker_coarse_context,
                            ranker_candidate_context)
                    scores_all = scores_all.clone()
                    scores_all[refine_indices] = (
                        scores_all[refine_indices] * score_factors
                    ).clamp(max=1.0)
                elif self.diffusion_refiner.candidate_ranker_quality_enabled:
                    selected, score_factors, ranker_stats = \
                        self.diffusion_refiner.select_quality_ranked_candidates(
                            segs_all[refine_indices], candidate_tensor,
                            candidate_features[refine_indices],
                            candidate_levels[refine_indices],
                            scores_all[refine_indices])
                    scores_all = scores_all.clone()
                    scores_all[refine_indices] = (
                        scores_all[refine_indices] * score_factors
                    ).clamp(max=1.0)
                else:
                    selected, ranker_stats = \
                        self.diffusion_refiner.select_ranked_candidates(
                            segs_all[refine_indices], candidate_tensor,
                            candidate_features[refine_indices],
                            candidate_levels[refine_indices],
                            scores_all[refine_indices], ranker_coarse_context,
                            ranker_candidate_context)
                primary_segments = segs_all.clone()
                primary_segments[refine_indices] = selected
            fallback_count = int(torch.isclose(
                primary_segments[refine_indices], segs_all[refine_indices],
                rtol=0.0, atol=1e-7
            ).all(dim=-1).sum().item())
            segs_all = primary_segments
        results = {'segments': segs_all,
                   'scores': scores_all,
                   'labels': cls_idxs_all}
        if residual_alignment_stats is not None:
            results['refinement_stats'] = residual_alignment_stats
        if p2d1_anchor_segments is not None:
            results.update({
                '_p2d1_anchor_segments': p2d1_anchor_segments,
                '_p2d1_selected_segments': segs_all.clone(),
                '_p2d1_selected_labels': cls_idxs_all.clone(),
                '_p2d1_replacement_mask': p2d1_replacement_mask,
            })
        if candidate_selection_stats is not None:
            results['candidate_selection_stats'] = \
                candidate_selection_stats
        if self.boundary_proposal_enabled:
            gates = torch.tanh(
                self.boundary_proposal_generator.gates.detach())
            results['refinement_stats'] = {
                'v13c_coarse_proposals': int(coarse_segs.shape[0]),
                'v13c_candidates_considered':
                    proposal_candidates_considered,
                'v13c_candidates_accepted':
                    proposal_candidates_accepted,
                'v13c_accept_ratio': (
                    proposal_candidates_accepted
                    / max(proposal_candidates_considered, 1)),
                'v13c_mean_absolute_gate': float(
                    gates.abs().mean().item()),
                'v13c_max_absolute_gate': float(
                    gates.abs().max().item()),
            }
        if self.v13_phase_a_enabled:
            results.update({
                '_v13_candidate_segments': coarse_segs.clone(),
                '_v13_candidate_scores': coarse_scores.clone(),
                '_v13_candidate_labels': cls_idxs_all.clone(),
            })
        if self.v14_probe_enabled:
            results.update(v14_probe_values)
        if self.diffusion_enabled:
            delta = segs_all - coarse_segs
            refined_delta = delta[refine_indices]
            coarse_duration = (
                coarse_segs[refine_indices, 1]
                - coarse_segs[refine_indices, 0]
            ).clamp_min(self.diffusion_refiner.min_duration)
            relative_adjustment = (
                refined_delta.abs() / coarse_duration[:, None]
                if refined_delta.numel() else refined_delta
            )
            results.update({
                'coarse_segments': coarse_segs,
                'coarse_scores': coarse_scores,
                'coarse_labels': cls_idxs_all.clone(),
                'refinement_stats': {
                    'refined_proposals': refined_count,
                    'mean_start_offset': float(refined_delta[:, 0].mean().item()) if refined_delta.numel() else 0.0,
                    'mean_end_offset': float(refined_delta[:, 1].mean().item()) if refined_delta.numel() else 0.0,
                    'mean_boundary_adjustment': float(refined_delta.abs().mean().item()) if refined_delta.numel() else 0.0,
                    'mean_relative_boundary_adjustment': float(relative_adjustment.mean().item()) if relative_adjustment.numel() else 0.0,
                    'illegal_boundary_fixes': illegal_count,
                    'coarse_fallbacks': fallback_count,
                    **gate_stats,
                    **ranker_stats,
                },
                '_refine_indices': refine_indices,
            })
            if self.oracle_diagnostic_enabled and multi_candidate_sets:
                results['oracle_candidate_segments'] = torch.stack(
                    multi_candidate_sets[:self.oracle_num_candidates], dim=0)
            if self.candidate_ranker_training_data and multi_candidate_sets:
                results.update({
                    '_ranker_coarse_segments': coarse_segs[refine_indices],
                    '_ranker_candidate_segments': torch.stack(
                        multi_candidate_sets, dim=0
                    )[:, refine_indices].permute(1, 0, 2),
                    '_ranker_reg_features': candidate_features[refine_indices],
                    '_ranker_level_ids': candidate_levels[refine_indices],
                    '_ranker_proposal_scores': scores_all[refine_indices],
                    '_ranker_labels': cls_idxs_all[refine_indices],
                    '_ranker_candidate_profiles': active_profiles,
                })
                if ranker_coarse_context is not None:
                    results.update({
                        '_ranker_coarse_context': ranker_coarse_context,
                        '_ranker_candidate_context': ranker_candidate_context,
                    })
                if ranker_synthetic_candidates is not None:
                    results.update({
                        '_ranker_synthetic_candidate_segments':
                            ranker_synthetic_candidates,
                        '_ranker_synthetic_candidate_context':
                            ranker_synthetic_context,
                    })

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            coarse = None
            oracle = None
            p2d1_diagnostic = None
            if '_p2d1_anchor_segments' in results_per_vid:
                p2d1_diagnostic = (
                    results_per_vid[
                        '_p2d1_anchor_segments'].detach().cpu(),
                    results_per_vid[
                        '_p2d1_selected_segments'].detach().cpu(),
                    results_per_vid[
                        '_p2d1_selected_labels'].detach().cpu(),
                    results_per_vid[
                        '_p2d1_replacement_mask'].detach().cpu(),
                )
            if self.diffusion_enabled:
                coarse = (
                    results_per_vid['coarse_segments'].detach().cpu(),
                    results_per_vid['coarse_scores'].detach().cpu(),
                    results_per_vid['coarse_labels'].detach().cpu(),
                )
                if 'oracle_segments' in results_per_vid:
                    oracle = (
                        results_per_vid['oracle_segments'].detach().cpu(),
                        results_per_vid['oracle_scores'].detach().cpu(),
                        results_per_vid['oracle_labels'].detach().cpu(),
                    )
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms=(self.test_nms_method == 'soft'),
                    multiclass=self.test_multiclass_nms,
                    sigma=self.test_nms_sigma,
                    voting_thresh=self.test_voting_thresh
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs <= 0.0] *= 0.0
                segs[segs >= vlen] = segs[segs >= vlen] * 0.0 + vlen
            if coarse is not None:
                coarse_segs, coarse_scores, coarse_labels = coarse
                if self.test_nms_method != 'none':
                    coarse_segs, coarse_scores, coarse_labels = batched_nms(
                        coarse_segs, coarse_scores, coarse_labels,
                        self.test_iou_threshold, self.test_min_score,
                        self.test_max_seg_num,
                        use_soft_nms=(self.test_nms_method == 'soft'),
                        multiclass=self.test_multiclass_nms,
                        sigma=self.test_nms_sigma,
                        voting_thresh=self.test_voting_thresh,
                    )
                if coarse_segs.shape[0] > 0:
                    coarse_segs = (coarse_segs * stride + 0.5 * nframes) / fps
                    coarse_segs.clamp_(0.0, vlen)
            if oracle is not None:
                oracle_segs, oracle_scores, oracle_labels = oracle
                if self.test_nms_method != 'none':
                    oracle_segs, oracle_scores, oracle_labels = batched_nms(
                        oracle_segs, oracle_scores, oracle_labels,
                        self.test_iou_threshold, self.test_min_score,
                        self.test_max_seg_num,
                        use_soft_nms=(self.test_nms_method == 'soft'),
                        multiclass=self.test_multiclass_nms,
                        sigma=self.test_nms_sigma,
                        voting_thresh=self.test_voting_thresh,
                    )
                if oracle_segs.shape[0] > 0:
                    oracle_segs = (
                        oracle_segs * stride + 0.5 * nframes) / fps
                    oracle_segs.clamp_(0.0, vlen)
            # 4: repack the results
            processed = {'video_id': vidx, 'segments': segs,
                         'scores': scores, 'labels': labels}
            if p2d1_diagnostic is not None:
                (p2d1_anchor, p2d1_selected, p2d1_labels,
                 p2d1_mask) = p2d1_diagnostic
                if p2d1_anchor.shape[0] > 0:
                    p2d1_anchor = (
                        p2d1_anchor * stride + 0.5 * nframes) / fps
                    p2d1_selected = (
                        p2d1_selected * stride + 0.5 * nframes) / fps
                    p2d1_anchor.clamp_(0.0, vlen)
                    p2d1_selected.clamp_(0.0, vlen)
                processed.update({
                    '_p2d1_anchor_segments': p2d1_anchor,
                    '_p2d1_selected_segments': p2d1_selected,
                    '_p2d1_selected_labels': p2d1_labels,
                    '_p2d1_replacement_mask': p2d1_mask,
                })
            if self.v13_phase_a_enabled:
                candidate_segments = results_per_vid[
                    '_v13_candidate_segments'].detach().cpu()
                candidate_scores = results_per_vid[
                    '_v13_candidate_scores'].detach().cpu()
                candidate_labels = results_per_vid[
                    '_v13_candidate_labels'].detach().cpu()
                gt_segments = results_per_vid[
                    '_v13_gt_segments'].detach().cpu()
                gt_labels = results_per_vid[
                    '_v13_gt_labels'].detach().cpu()
                if candidate_segments.shape[0] > 0:
                    candidate_segments = (
                        candidate_segments * stride + 0.5 * nframes) / fps
                    candidate_segments.clamp_(0.0, vlen)
                if gt_segments.shape[0] > 0:
                    gt_segments = (
                        gt_segments * stride + 0.5 * nframes) / fps
                    gt_segments.clamp_(0.0, vlen)
                processed.update({
                    '_v13_candidate_segments': candidate_segments,
                    '_v13_candidate_scores': candidate_scores,
                    '_v13_candidate_labels': candidate_labels,
                    '_v13_gt_segments': gt_segments,
                    '_v13_gt_labels': gt_labels,
                    '_v13_nms_config': {
                        'method': self.test_nms_method,
                        'iou_threshold': self.test_iou_threshold,
                        'min_score': self.test_min_score,
                        'max_seg_num': self.test_max_seg_num,
                        'multiclass': self.test_multiclass_nms,
                        'sigma': self.test_nms_sigma,
                        'voting_thresh': self.test_voting_thresh,
                    },
                    '_v13_uses_ground_truth': True,
                })
            if self.v14_probe_enabled:
                all_probe_segments = results_per_vid[
                    '_v14_all_segments'].detach().cpu()
                probe_segments = results_per_vid[
                    '_v14_segments'].detach().cpu()
                probe_gt_segments = results_per_vid[
                    '_v14_gt_segments'].detach().cpu()
                if all_probe_segments.shape[0] > 0:
                    all_probe_segments = (
                        all_probe_segments * stride + 0.5 * nframes) / fps
                    all_probe_segments.clamp_(0.0, vlen)
                if probe_segments.shape[0] > 0:
                    probe_segments = (
                        probe_segments * stride + 0.5 * nframes) / fps
                    probe_segments.clamp_(0.0, vlen)
                if probe_gt_segments.shape[0] > 0:
                    probe_gt_segments = (
                        probe_gt_segments * stride + 0.5 * nframes) / fps
                    probe_gt_segments.clamp_(0.0, vlen)
                processed.update({
                    '_v14_all_segments': all_probe_segments,
                    '_v14_all_scores': results_per_vid[
                        '_v14_all_scores'].detach().cpu(),
                    '_v14_all_labels': results_per_vid[
                        '_v14_all_labels'].detach().cpu(),
                    '_v14_segments': probe_segments,
                    '_v14_scores': results_per_vid[
                        '_v14_scores'].detach().cpu(),
                    '_v14_labels': results_per_vid[
                        '_v14_labels'].detach().cpu(),
                    '_v14_reg_features': results_per_vid[
                        '_v14_reg_features'].detach().cpu(),
                    '_v14_level_ids': results_per_vid[
                        '_v14_level_ids'].detach().cpu(),
                    '_v14_score_features': results_per_vid[
                        '_v14_score_features'].detach().cpu(),
                    '_v14_gt_segments': probe_gt_segments,
                    '_v14_gt_labels': results_per_vid[
                        '_v14_gt_labels'].detach().cpu(),
                    '_v14_duration': float(vlen),
                })
            if coarse is not None:
                diagnostic_indices = results_per_vid['_refine_indices'].detach().cpu()
                diagnostic_coarse = results_per_vid['coarse_segments'][diagnostic_indices].detach().cpu()
                diagnostic_refined = results_per_vid['segments'][diagnostic_indices].detach().cpu()
                if diagnostic_coarse.shape[0] > 0:
                    diagnostic_coarse = (diagnostic_coarse * stride + 0.5 * nframes) / fps
                    diagnostic_refined = (diagnostic_refined * stride + 0.5 * nframes) / fps
                    diagnostic_coarse.clamp_(0.0, vlen)
                    diagnostic_refined.clamp_(0.0, vlen)
                processed.update({
                    'coarse_segments': coarse_segs,
                    'coarse_scores': coarse_scores,
                    'coarse_labels': coarse_labels,
                    'refinement_stats': results_per_vid['refinement_stats'],
                    '_diagnostic_coarse_segments': diagnostic_coarse,
                    '_diagnostic_refined_segments': diagnostic_refined,
                })
            elif 'refinement_stats' in results_per_vid:
                processed['refinement_stats'] = \
                    results_per_vid['refinement_stats']
            if 'candidate_selection_stats' in results_per_vid:
                processed['candidate_selection_stats'] = \
                    results_per_vid['candidate_selection_stats']
            if oracle is not None:
                processed.update({
                    'oracle_segments': oracle_segs,
                    'oracle_scores': oracle_scores,
                    'oracle_labels': oracle_labels,
                    '_oracle_uses_ground_truth': True,
                })
            if '_ranker_candidate_segments' in results_per_vid:
                for key in (
                        '_ranker_coarse_segments',
                        '_ranker_candidate_segments',
                        '_ranker_reg_features', '_ranker_level_ids',
                        '_ranker_proposal_scores', '_ranker_labels'):
                    processed[key] = results_per_vid[key].detach().cpu()
                for key in ('_ranker_coarse_context',
                            '_ranker_candidate_context',
                            '_ranker_synthetic_candidate_segments',
                            '_ranker_synthetic_candidate_context'):
                    if key in results_per_vid:
                        processed[key] = results_per_vid[key].detach().cpu()
                processed['_ranker_candidate_profiles'] = \
                    results_per_vid.get('_ranker_candidate_profiles', [])
            processed_results.append(processed)

        return processed_results
