import copy
import yaml

DEFAULTS = {
    # random seed for reproducibility, a large number is preferred
    "init_rand_seed": 1234567891,
    # dataset loader, specify the dataset here
    "dataset_name": "epic",
    "devices": ['cuda:0'],  # default: single gpu
    "train_split": ('training',),
    "val_split": ('validation',),
    "model_name": "DyFADet",
    "dataset": {
        # temporal stride of the feats
        "feat_stride": 16,
        # number of frames for each feat
        "num_frames": 32,
        # default fps, may vary across datasets; Set to none for read from json file
        "default_fps": None,
        # input feat dim
        "input_dim": 2304,
        # number of classes
        "num_classes": 97,
        # downsampling rate of features, 1 to use original resolution
        "downsample_rate": 1,
        # max sequence length during training
        "max_seq_len": 2304,
        # threshold for truncating an action
        "trunc_thresh": 0.5,
        # set to a tuple (e.g., (0.9, 1.0)) to enable random feature cropping
        # might not be implemented by the dataloader
        "crop_ratio": None,
        # if true, force upsampling of the input features into a fixed size
        # only used for ActivityNet
        "force_upsampling": False,
    },
    "loader": {
        "batch_size": 8,
        "num_workers": 4,
    },
    # network architecture
    "model": {
        # type of backbone (DynE | conv)
        "backbone_type": 'DynE',
        # type of FPN (fpn | identity)
        "fpn_type": "identity",
        "backbone_arch": (2, 2, 5),
        # scale factor between pyramid levels
        "scale_factor": 2,
        # regression range for pyramid levels
        "regression_range": [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        # window size for self attention; <=1 to use full seq (ie global attention)
        "n_encoder_win_size": -1,
        # kernel size for embedding network
        "embd_kernel_size": 3,
        # (output) feature dim for embedding network
        "embd_dim": 512,
        # if attach group norm to embedding network
        "embd_with_ln": True,
        # feat dim for FPN
        "fpn_dim": 512,
        # if add ln at the end of fpn outputs
        "fpn_with_ln": True,
        # optional identity-preserving boundary residuals for regression only
        "regression_boundary_pyramid": {
            "enabled": False,
            "hidden_dim": 128,
            "dropout": 0.0,
            "gate_init": 0.0,
        },
        # feat dim for head
        "head_dim": 512,
        # kernel size for reg/cls/center heads
        "head_kernel_size": 3,
        # number of layers in the head (including the final one)
        "head_num_layers": 3,
        # defines the max length of the buffered points
        "max_buffer_len_factor": 6.0,
        # disable abs position encoding (added to input embedding)
        "use_abs_pe": False,
        # the K to generate large kernel
        "k": 1.5,
        # initialization of gaussian variance for the weight in SGP
        "init_conv_vars": 0,
        # add gaussian noise with the variance, play a similar role to position embedding
        "input_noise": 0,

        "diffusion_refiner": {
            "enabled": False,
            "num_train_timesteps": 1000,
            "num_inference_steps": 10,
            "beta_schedule": "cosine",
            "prediction_type": "epsilon",
            "state_space": "endpoint",
            "hidden_dim": None,
            "num_decoder_layers": 2,
            "dropout": 0.1,
            "detach_coarse_boundary": True,
            "condition_on_reg_feature": True,
            "boundary_context_enabled": False,
            "boundary_context_offsets": [-2, -1, 0, 1, 2],
            "refine_topk": 200,
            "min_duration": 0.01,
            "clamp_boundary": True,
            "use_ddim": True,
            "inference_seed": 0,
            "residual_blend_alpha": 1.0,
            "max_relative_shift": None,
            "boundary_loss_enabled": False,
            "boundary_loss_weight": 0.0,
            "boundary_loss_snr_cap": 5.0,
            "boundary_loss_max_residual": 2.0,
            # V15-B: x0-specific timestep balancing and boundary geometry.
            # Disabled by default so every historical experiment is unchanged.
            "v15b_enabled": False,
            "v15b_min_snr_gamma": 5.0,
            "v15b_tiou_loss_weight": 0.0,
            "v15b_endpoint_loss_weight": 0.0,
            "v15b_thresholds": [0.6, 0.7],
            "v15b_threshold_sigma": 0.05,
            "v15b_threshold_gain": 1.0,
            "v15b_endpoint_beta": 0.1,
            # V15-C1: an intrinsic identity/no-change prior for high-quality
            # coarse proposals. It keeps the V15-B sampler and architecture.
            "v15c1_enabled": False,
            "v15c1_identity_start_tiou": 0.7,
            "v15c1_identity_full_tiou": 0.8,
            "v15c1_identity_max_weight": 2.0,
            "v15c1_identity_loss_weight": 0.0,
            "v15c1_duration_asym_loss_weight": 0.0,
            "v15c1_duration_beta": 0.05,
            "final_ddim_loss_enabled": False,
            "final_ddim_loss_weight": 0.0,
            "final_ddim_train_steps": 3,
            "final_ddim_max_samples": 128,
            "final_ddim_max_residual": 2.0,
            "candidate_ranker_enabled": False,
            "candidate_ranker_num_candidates": 8,
            "candidate_ranker_hidden_dim": 128,
            "candidate_ranker_dropout": 0.1,
            "candidate_ranker_delta_margin": 0.0,
            "candidate_ranker_improve_threshold": 0.5,
            "candidate_ranker_regression_weight": 1.0,
            "candidate_ranker_bce_weight": 1.0,
            "candidate_ranker_listwise_weight": 1.0,
            "candidate_ranker_delta_scale": 1.0,
            "candidate_ranker_soft_listwise": False,
            "candidate_ranker_softmax_temperature": 0.01,
            "candidate_ranker_zero_init_delta": False,
            "candidate_ranker_ensemble_enabled": False,
            "candidate_ranker_ensemble_size": 5,
            "candidate_ranker_ensemble_min_agreement": 3,
            "candidate_ranker_low_tiou_safety_enabled": False,
            "candidate_ranker_low_tiou_safety_thresholds": [0.3, 0.4],
            "candidate_ranker_context_enabled": False,
            "candidate_ranker_context_offsets": [-2, -1, 0, 1, 2],
            "candidate_ranker_target_type": "delta_tiou",
            "candidate_ranker_tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
            "candidate_ranker_threshold_weights": [1, 1, 1, 1, 1],
            "candidate_ranker_score_weight_power": 0.0,
            "candidate_ranker_negative_utility_weight": 1.0,
            "candidate_ranker_proximity_sigma": 0.05,
            "candidate_ranker_proximity_gain": 2.0,
            "candidate_ranker_crossing_bonus": 0.05,
            "candidate_ranker_harmful_crossing_penalty": 0.1,
            "candidate_ranker_quality_enabled": False,
            "candidate_ranker_quality_focal_gamma": 2.0,
            "candidate_ranker_quality_bce_weight": 1.0,
            "candidate_ranker_quality_pairwise_weight": 1.0,
            "candidate_ranker_quality_listwise_weight": 1.0,
            "candidate_ranker_quality_pairwise_temperature": 0.1,
            "candidate_ranker_quality_margin": 0.02,
            "candidate_ranker_quality_harmful_weight": 2.0,
            "candidate_ranker_quality_rescore_enabled": False,
            "candidate_ranker_quality_rescore_gamma": 0.5,
            "candidate_ranker_quality_rescore_max_change": 0.1,
            "quality_gate_enabled": False,
            "quality_gate_hidden_dim": 128,
            "quality_gate_dropout": 0.1,
            "quality_gate_threshold": 0.5,
            "quality_gate_margin": 0.0,
            "quality_gate_loss_weight": 1.0,
            "quality_gate_noise_std": 0.25,
        },
        "start_end_distribution": {
            "enabled": False,
            "hidden_dim": 128,
            "num_bins": 17,
            "max_residual": 2.0,
            "dropout": 0.0,
            "gate_init": 0.0,
            "distribution_loss_weight": 1.0,
            "tiou_loss_weight": 0.25,
        },
        # V11-A: class-aware TriDet-style relative boundary decoding.
        "trident_head": {
            "enabled": False,
            "hidden_dim": 128,
            "num_bins": 16,
            "num_layers": 2,
            "kernel_size": 3,
            "dropout": 0.0,
            "prior_prob": 0.01,
            "detach_boundary_features": True,
            "gate_init": 0.0,
            "aux_loss_weight": 1.0,
            "iou_weight_power": 0.2,
        },
        # V11-B: SGP-style temporal aggregation in regression only.
        "regression_sgp_adapter": {
            "enabled": False,
            "hidden_dim": 256,
            "kernel_size": 3,
            "expansion": 1.5,
            "dropout": 0.0,
            "gate_init": 0.0,
        },
        # V11-C: training-only feature perturbation and consistency.
        "temporal_consistency": {
            "enabled": False,
            "drop_probability": 0.1,
            "noise_std": 0.01,
            "loss_weight": 1.0,
            "classification_weight": 0.0,
            "regression_weight": 1.0,
            "teacher_detach": True,
        },
        # V11-D: direct strict-boundary optimization at high tIoU.
        "high_iou_boundary": {
            "enabled": False,
            "thresholds": [0.6, 0.7],
            "margin_loss_weight": 0.5,
            "endpoint_loss_weight": 0.25,
            "endpoint_beta": 0.05,
        },
        # V12-A: dense class-aware localization quality score calibration.
        "dense_iou_quality": {
            "enabled": False,
            "hidden_dim": 128,
            "num_layers": 2,
            "kernel_size": 3,
            "dropout": 0.0,
            "prior_prob": 0.01,
            "gate_init": 0.0,
            "max_logit_shift": 0.5,
            "score_fusion": "bounded_logit_residual",
            "score_fusion_beta": 0.0,
            "focal_beta": 2.0,
            "quality_loss_weight": 1.0,
            "calibration_loss_weight": 0.25,
            "proposal_target_enabled": False,
            "proposal_target_topk": 2000,
            "ranking_enabled": False,
            "pairwise_loss_weight": 1.0,
            "listwise_loss_weight": 0.5,
            "ranking_topk": 32,
            "ranking_overlap_threshold": 0.3,
            "ranking_quality_margin": 0.02,
            "pairwise_temperature": 0.25,
            "listwise_temperature": 0.25,
            "listwise_target_temperature": 0.1,
            "ranking_tiou_thresholds": [0.5, 0.6, 0.7],
            "threshold_proximity_sigma": 0.05,
            "threshold_proximity_gain": 2.0,
            "threshold_crossing_gain": 2.0,
        },
        # V13-A: training-time alignment of the original classification
        # confidence with the localization quality of the original boundary.
        # This adds no inference-time module or checkpoint parameters.
        "task_alignment": {
            "enabled": False,
            "alpha": 1.0,
            "beta": 2.0,
            "soft_label_mix": 0.5,
            "minimum_positive_weight": 0.5,
            "maximum_positive_weight": 2.0,
            "pairwise_loss_weight": 0.25,
            "ranking_topk": 32,
            "ranking_overlap_threshold": 0.3,
            "ranking_quality_margin": 0.05,
            "pairwise_temperature": 0.25,
            "ranking_tiou_thresholds": [0.5, 0.6, 0.7],
            "threshold_proximity_sigma": 0.05,
            "threshold_proximity_gain": 2.0,
        },
        # P4-B: zero-initialized residual corrections over frozen A0 heads.
        # A fixed confidence gate retains the original A0 output whenever a
        # learned correction is not trusted.
        "residual_alignment": {
            "enabled": False,
            "hidden_dim": 128,
            "num_layers": 2,
            "kernel_size": 3,
            "dropout": 0.1,
            "maximum_class_logit_residual": 0.5,
            "maximum_regression_residual": 0.5,
            "residual_weight": 1.0,
            "minimum_confidence": 0.5,
            "confidence_prior": 0.1,
            "classification_consistency_loss_weight": 1.0,
            "regression_consistency_loss_weight": 1.0,
            "residual_magnitude_loss_weight": 0.1,
            "confidence_loss_weight": 0.5,
            "confidence_gain_margin": 0.005,
            "confidence_harm_margin": 0.005,
        },
        # V13-C: boundary-aware multi-hypothesis proposal generation.
        "boundary_proposal_generator": {
            "enabled": False,
            "hidden_dim": 128,
            "num_bins": 17,
            "max_residual": 4.0,
            "num_candidates": 4,
            "dropout": 0.0,
            "gate_init": 0.0,
            "distribution_loss_weight": 1.0,
            "coverage_loss_weight": 1.0,
            "improvement_loss_weight": 0.5,
            "accept_threshold": 0.55,
            "max_score_boost": 0.05,
        },
        # P7: residual long-range temporal modeling. The dependency is loaded
        # lazily, so the original A0 baseline remains runnable without Mamba.
        "mamba_adapter": {
            "enabled": False,
            "position": "stem_branch",
            "backend": "dbm",
            "d_state": 16,
            "d_conv": 4,
            "expand": 1,
            "drop_path_rate": 0.3,
            "layer_scale_init": 0.0001,
            "residual_weight": 1.0,
            "use_fast_path": True,
        },
        # Optional selective replacement of the legacy DynamicScale modules.
        # The disabled identity path constructs the exact historical heads.
        "selective_dynamic_scale": {
            "enabled": False,
            "variant": "identity",
            "reduction": 4,
            "min_hidden_dim": 32,
            "channel_wise": True,
            "temperature": 1.0,
            "selection_gate_init": 0.0,
            "selection_gate_mode": "tanh",
            "selection_gate_bound": 0.25,
            "selection_gate_lr_scale": 1.0,
            "preserve_parent_rng": True,
            # Zero-based DynamicScale blocks in the classification head.
            # The historical DyHead has exactly two blocks: [0, 1].
            "active_layers": [0, 1],
        },
        # P8-Fusion: task-specific access to the historical neck-only Mamba.
        # shared reproduces the exact 286c26f path. mtg1 learns two bounded
        # residual gates initialized at one. td1 sends baseline features to
        # classification and Mamba features to regression.
        "mamba_task_routing": {
            "enabled": False,
            "variant": "shared",
            "cls_gate_init": 1.0,
            "reg_gate_init": 1.0,
            "gate_bound": 0.5,
            "gate_lr_scale": 1.0,
            "gradient_diagnostics": False,
        },

    },
    "loss": {"diffusion_weight": 1.0},
    "training": {
        "refiner_only": False,
        "freeze_backbone": False,
        "freeze_encoder": False,
        "freeze_cls_branch": False,
        "freeze_reg_branch": False,
        "freeze_trident_head": False,
        "freeze_regression_sgp_adapter": False,
        "freeze_dense_iou_quality_head": False,
        "freeze_residual_alignment_head": False,
        "quality_gate_only": False,
        "candidate_ranker_only": False,
    },
    "train_cfg": {
        # radius | none (if to use center sampling)
        "center_sample": "radius",
        "center_sample_radius": 1.5,
        "loss_weight": 1.0,  # on reg_loss, use -1 to enable auto balancing
        "cls_prior_prob": 0.01,
        "init_loss_norm": 2000,
        # gradient cliping, not needed for pre-LN transformer
        "clip_grad_l2norm": -1,
        # cls head without data (a fix to epic-kitchens / thumos)
        "head_empty_cls": [],
        # dropout ratios for tranformers
        "dropout": 0.0,
        # ratio for drop path
        "droppath": 0.1,
        # if to use label smoothing (>0.0)
        "label_smoothing": 0.0,
    },
    "test_cfg": {
        "pre_nms_thresh": 0.001,
        "pre_nms_topk": 5000,
        "iou_threshold": 0.1,
        "min_score": 0.01,
        "max_seg_num": 1000,
        "nms_method": 'soft',  # soft | hard | none
        "nms_sigma": 0.5,
        "duration_thresh": 0.05,
        "multiclass_nms": True,
        "ext_score_file": None,
        "voting_thresh": 0.75,
    },
    # optimizer (for training)
    "opt": {
        # solver
        "type": "AdamW",  # SGD or AdamW
        # solver params
        "momentum": 0.9,
        "weight_decay": 0.0,
        "learning_rate": 1e-3,
        # excluding the warmup epochs
        "epochs": 30,
        # lr scheduler: cosine / multistep
        "warmup": True,
        "warmup_epochs": 5,
        "schedule_type": "cosine",
        # Minimum learning rate
        "eta_min": 1e-8,
        # in #epochs excluding warmup
        "schedule_steps": [],
        "schedule_gamma": 0.1,
    }
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_default_config():
    return copy.deepcopy(DEFAULTS)


def _update_config(config):
    # fill in derived fields
    config["model"]["input_dim"] = config["dataset"]["input_dim"]
    config["model"]["num_classes"] = config["dataset"]["num_classes"]
    config["model"]["max_seq_len"] = config["dataset"]["max_seq_len"]
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]
    config["model"]["loss_cfg"] = config["loss"]
    config["model"]["training_cfg"] = config["training"]
    return config


def load_config(config_file, defaults=DEFAULTS):

    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    defaults = copy.deepcopy(defaults)
    _merge(defaults, config)
    config = _update_config(config)
    return config
