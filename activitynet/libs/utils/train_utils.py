import os
import json
import pickle
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from .p2d0_diagnostics import (
    build_cluster_oracle_record,
    full_oracle_masks,
    global_replacement_mask_sweep,
    high_tiou_crossings,
    render_cluster_predictions,
    summarize_cluster_records,
)
from .p3_score_diagnostics import (
    build_bounded_score_record,
    global_score_adjustment_mask_sweep,
    render_bounded_score_predictions,
    summarize_score_adjustments,
)
from .p4_joint_diagnostics import (
    build_joint_score_records,
    summarize_boundary_replacements,
)
from .v13_diagnostics import (
    build_v13_oracle_variants,
    merge_recall_statistics,
    proposal_recall,
)
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm


################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    explicit_no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if not p.requires_grad:
                continue
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if getattr(p, '_no_weight_decay', False):
                # Mamba state parameters such as A_log and D explicitly opt
                # out of AdamW decay; changing their continuous-time dynamics
                # through L2 shrinkage is not equivalent to regularization.
                no_decay.add(fpn)
                explicit_no_decay.add(fpn)
            elif pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)
            elif pn.endswith('scale_weight'):
                # corner case for relative position encoding
                no_decay.add(fpn)
            elif pn.endswith('output_weight'):
                # corner case for relative position encoding
                no_decay.add(fpn)
            elif 'odc_norm' in pn or 'odp_norm' in pn:
                # norm in MaskedDC1D
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    # Cover standard Transformer parameters while retaining the legacy rules.
    for pn, parameter in param_dict.items():
        if pn not in decay and pn not in no_decay:
            (decay if parameter.ndim >= 2 else no_decay).add(pn)
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    optimizer.explicit_no_weight_decay_names = tuple(
        sorted(explicit_no_decay))
    if explicit_no_decay:
        print(
            "Optimizer explicit no-weight-decay parameters ({}): {}".format(
                len(explicit_no_decay),
                ", ".join(sorted(explicit_no_decay))))

    return optimizer


def make_scheduler(
        optimizer,
        optimizer_config,
        num_iters_per_epoch,
        last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # get eta min
        eta_min = optimizer_config["eta_min"]

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                eta_min=eta_min,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get eta min
        eta_min = optimizer_config["eta_min"]

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                eta_min=eta_min,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


################################################################################
def train_one_epoch(
        train_loader,
        model,
        optimizer,
        scheduler,
        curr_epoch,
        model_ema=None,
        clip_grad_l2norm=-1,
        print_freq=20,
        auto_cast=None,
        loss_scaler=None,
        metrics_file=None,
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()

    # main training loop
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    epoch_start = start
    for iter_idx, video_list in enumerate(train_loader, 0):

        # zero out optim
        optimizer.zero_grad(set_to_none=True)
        # forward / backward the model
        losses = model(video_list)

        gradient_diagnostics = None
        core_model = model.module if hasattr(model, "module") else model
        if iter_idx == 0 and getattr(
                core_model, "fusion_gradient_diagnostics", False):
            gradient_diagnostics = _compute_fusion_gradient_diagnostics(
                core_model, losses)

        losses['final_loss'].backward()

        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler

        # loss_scaler.step(optimizer)
        # loss_scaler.update()


        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        # Track every iteration. The original terminal print frequency remains unchanged.
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())
        if metrics_file is not None:
            record = {
                "phase": "train",
                "epoch": curr_epoch,
                "iteration": iter_idx,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - epoch_start,
                "gpu_peak_memory_mb": (
                    torch.cuda.max_memory_allocated() / (1024 ** 2)
                    if torch.cuda.is_available() else 0.0
                ),
            }
            record.update({key: value.item() for key, value in losses.items()})
            if gradient_diagnostics is not None:
                record["gradient_diagnostics"] = gradient_diagnostics
            with open(metrics_file, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx

            # print to terminal
            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = 'Loss {:.2f} ({:.2f})\n'.format(
                losses_tracker['final_loss'].val,
                losses_tracker['final_loss'].avg
            )
            block4 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4 += '\t{:s} {:.2f} ({:.2f})'.format(
                        key, value.val, value.avg
                    )

            print('\t'.join([block1, block2, block3, block4]))

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return {key: value.avg for key, value in losses_tracker.items()}


def _compute_fusion_gradient_diagnostics(model, losses):
    """Measure cls/reg gradient agreement without populating ``.grad``."""
    selected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("backbone.", "neck.", "mamba_neck_adapter.")):
            selected.append((name, parameter))
    parameters = [parameter for _, parameter in selected]
    cls_grads = torch.autograd.grad(
        losses["cls_loss"], parameters, retain_graph=True,
        allow_unused=True)
    reg_grads = torch.autograd.grad(
        losses["reg_loss"], parameters, retain_graph=True,
        allow_unused=True)

    def summarize(indices):
        dot = cls_norm = reg_norm = 0.0
        parameter_cosines = []
        used = 0
        for index in indices:
            cls_grad = cls_grads[index]
            reg_grad = reg_grads[index]
            if cls_grad is None or reg_grad is None:
                continue
            cls_value = cls_grad.detach().float()
            reg_value = reg_grad.detach().float()
            cls_sq = float(torch.sum(cls_value * cls_value).cpu())
            reg_sq = float(torch.sum(reg_value * reg_value).cpu())
            pair_dot = float(torch.sum(cls_value * reg_value).cpu())
            dot += pair_dot
            cls_norm += cls_sq
            reg_norm += reg_sq
            if cls_sq > 0.0 and reg_sq > 0.0:
                parameter_cosines.append(
                    pair_dot / ((cls_sq * reg_sq) ** 0.5))
            used += 1
        cosine = None
        if cls_norm > 0.0 and reg_norm > 0.0:
            cosine = dot / ((cls_norm * reg_norm) ** 0.5)
        return {
            "cosine": cosine,
            "classification_gradient_norm": cls_norm ** 0.5,
            "regression_gradient_norm": reg_norm ** 0.5,
            "negative_parameter_fraction": (
                sum(value < 0.0 for value in parameter_cosines)
                / len(parameter_cosines)
                if parameter_cosines else None),
            "parameter_tensors_used": used,
        }

    overall = summarize(range(len(selected)))
    levels = {}
    for level in range(6):
        prefix = "mamba_neck_adapter.blocks.{}.".format(level)
        indices = [index for index, (name, _) in enumerate(selected)
                   if name.startswith(prefix)]
        levels[str(level)] = summarize(indices)
    return {"shared": overall, "mamba_levels": levels}


def valid_one_epoch(
        val_loader,
        model,
        curr_epoch,
        ext_score_file=None,
        evaluator=None,
        output_file=None,
        tb_writer=None,
        print_freq=100,
        oracle_report_file=None,
        v13_report_file=None,
        p2d0_report_file=None,
        p2d0_cluster_thresholds=(0.7, 0.8, 0.9),
        p2d0_replacement_budgets=(0.01, 0.02, 0.05, 0.10),
        p3_score_report_file=None,
        p3_score_logit_caps=(0.25, 0.5, 1.0),
        p3_score_budgets=(0.01, 0.02, 0.05),
        p4_joint_report_file=None,
        p4_cluster_threshold=0.9,
        p4_boundary_budgets=(0.0, 0.01, 0.02, 0.05),
        p4_score_budgets=(0.0, 0.01, 0.02, 0.05),
        p4_score_logit_cap=1.0,
        p2d1_report_file=None,
        p2d1_calibration_only=False,
        diagnostic_dataset_name=None,
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)
    if oracle_report_file is not None and evaluator is None:
        raise ValueError("Oracle report requires an evaluator")
    if v13_report_file is not None and evaluator is None:
        raise ValueError("V13 Phase A report requires an evaluator")
    if p2d0_report_file is not None and evaluator is None:
        raise ValueError("P2-D0 report requires an evaluator")
    if p3_score_report_file is not None and evaluator is None:
        raise ValueError("P3-0 score report requires an evaluator")
    if p4_joint_report_file is not None and evaluator is None:
        raise ValueError("P4-0 joint report requires an evaluator")
    if (p2d1_report_file is not None and evaluator is None
            and not p2d1_calibration_only):
        raise ValueError("P2-D1 report requires an evaluator")
    if p2d1_calibration_only and p2d1_report_file is None:
        raise ValueError(
            "P2-D1 calibration-only mode requires a report file")
    p2d0_cluster_thresholds = tuple(
        float(value) for value in p2d0_cluster_thresholds)
    p2d0_replacement_budgets = tuple(
        float(value) for value in p2d0_replacement_budgets)
    p3_score_logit_caps = tuple(
        float(value) for value in p3_score_logit_caps)
    p3_score_budgets = tuple(
        float(value) for value in p3_score_budgets)
    p4_cluster_threshold = float(p4_cluster_threshold)
    p4_boundary_budgets = tuple(
        float(value) for value in p4_boundary_budgets)
    p4_score_budgets = tuple(
        float(value) for value in p4_score_budgets)
    p4_score_logit_cap = float(p4_score_logit_cap)
    if p2d0_report_file is not None:
        if diagnostic_dataset_name != "anet":
            raise ValueError("P2-D0 diagnostic supports ActivityNet only")
        if any(not 0.0 < value <= 1.0
               for value in p2d0_cluster_thresholds):
            raise ValueError(
                "P2-D0 cluster thresholds must be within (0, 1]")
        if len(p2d0_cluster_thresholds) != len(
                set(p2d0_cluster_thresholds)):
            raise ValueError("P2-D0 cluster thresholds must be unique")
        if any(not 0.0 <= value <= 1.0
               for value in p2d0_replacement_budgets):
            raise ValueError(
                "P2-D0 replacement budgets must be within [0, 1]")
        if len(p2d0_replacement_budgets) != len(
                set(p2d0_replacement_budgets)):
            raise ValueError("P2-D0 replacement budgets must be unique")
    if p3_score_report_file is not None:
        if diagnostic_dataset_name != "anet":
            raise ValueError("P3-0 score diagnostic supports ActivityNet only")
        if any(value <= 0.0 for value in p3_score_logit_caps):
            raise ValueError("P3-0 logit caps must be positive")
        if len(p3_score_logit_caps) != len(set(p3_score_logit_caps)):
            raise ValueError("P3-0 logit caps must be unique")
        if any(not 0.0 < value <= 1.0 for value in p3_score_budgets):
            raise ValueError("P3-0 score budgets must be within (0, 1]")
        if len(p3_score_budgets) != len(set(p3_score_budgets)):
            raise ValueError("P3-0 score budgets must be unique")
    if p4_joint_report_file is not None:
        if diagnostic_dataset_name != "anet":
            raise ValueError("P4-0 joint diagnostic supports ActivityNet only")
        if not 0.0 < p4_cluster_threshold <= 1.0:
            raise ValueError("P4-0 cluster threshold must be within (0, 1]")
        for name, budgets in (
                ("boundary", p4_boundary_budgets),
                ("score", p4_score_budgets)):
            if any(not 0.0 <= value <= 1.0 for value in budgets):
                raise ValueError(
                    "P4-0 {} budgets must be within [0, 1]".format(name))
            if len(budgets) != len(set(budgets)):
                raise ValueError(
                    "P4-0 {} budgets must be unique".format(name))
            if 0.0 not in budgets:
                raise ValueError(
                    "P4-0 {} budgets must include identity 0".format(name))
        if p4_score_logit_cap <= 0.0:
            raise ValueError("P4-0 score logit cap must be positive")

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }
    coarse_results = {key: [] for key in results}
    oracle_results = {key: [] for key in results}
    v13_score_results = {key: [] for key in results}
    v13_boundary_results = {key: [] for key in results}
    v13_joint_results = {key: [] for key in results}
    v13_pre_nms_recall = []
    v13_post_nms_recall = []
    v13_candidate_count = 0
    v13_replacement_count = 0
    p2d0_records = {
        threshold: [] for threshold in p2d0_cluster_thresholds}
    p3_score_records = []
    p4_joint_records = []
    refinement_stats = []

    def append_predictions(raw, video_id, segments, scores, labels):
        count = segments.shape[0]
        if not count:
            return
        raw['video-id'].extend([video_id] * count)
        raw['t-start'].append(segments[:, 0])
        raw['t-end'].append(segments[:, 1])
        raw['label'].append(labels)
        raw['score'].append(scores)

    # loop over validation set
    start = time.time()
    eval_start = start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def add_gt_diagnostics(prediction, sample):
        """Compare aligned coarse/refined proposals to GT outside model inference."""
        coarse = prediction.get('_diagnostic_coarse_segments')
        refined = prediction.get('_diagnostic_refined_segments')
        gt = sample.get('segments')
        stats = prediction.get('refinement_stats')
        if stats is None or coarse is None or refined is None:
            return
        if gt is None or not gt.numel() or not coarse.numel():
            stats['tiou_improved_ratio'] = 0.0
            stats['tiou_decreased_ratio'] = 0.0
            return
        gt = gt.detach().cpu()
        gt = (gt * float(sample['feat_stride'])
              + 0.5 * float(sample['feat_num_frames'])) / float(sample['fps'])
        gt.clamp_(0.0, float(sample['duration']))

        def best_tiou(segments):
            left = torch.maximum(segments[:, None, 0], gt[None, :, 0])
            right = torch.minimum(segments[:, None, 1], gt[None, :, 1])
            intersection = (right - left).clamp_min(0)
            union = ((segments[:, None, 1] - segments[:, None, 0])
                     + (gt[None, :, 1] - gt[None, :, 0])
                     - intersection).clamp_min(1e-6)
            return (intersection / union).max(dim=1).values

        change = best_tiou(refined) - best_tiou(coarse)
        stats['tiou_improved_ratio'] = float((change > 1e-6).float().mean())
        stats['tiou_decreased_ratio'] = float((change < -1e-6).float().mean())

    def add_p2d1_gt_diagnostics(prediction, sample):
        """Measure selector outcomes with validation GT outside inference."""
        anchors = prediction.get('_p2d1_anchor_segments')
        selected = prediction.get('_p2d1_selected_segments')
        labels = prediction.get('_p2d1_selected_labels')
        replacement = prediction.get('_p2d1_replacement_mask')
        stats = prediction.get('candidate_selection_stats')
        gt = sample.get('segments')
        gt_labels = sample.get('labels')
        if any(value is None for value in (
                anchors, selected, labels, replacement, stats)):
            return
        stats.update({
            'p2d1_true_improvements': 0,
            'p2d1_true_harms': 0,
            'p2d1_true_neutral': 0,
            'p2d1_true_gain_sum': 0.0,
            'p5_true_ap_utility_sum': 0.0,
        })
        for threshold in (0.85, 0.90, 0.95):
            key = str(threshold).replace('.', '_')
            stats['p2d1_tiou_' + key + '_improved'] = 0
            stats['p2d1_tiou_' + key + '_harmed'] = 0
        if gt is None or gt_labels is None:
            return
        gt = gt.detach().cpu()
        gt_labels = gt_labels.detach().cpu()
        gt = (gt * float(sample['feat_stride'])
              + 0.5 * float(sample['feat_num_frames'])) / \
            float(sample['fps'])
        gt.clamp_(0.0, float(sample['duration']))

        def pairwise(first, second):
            if not first.shape[0] or not second.shape[0]:
                return first.new_zeros((first.shape[0], second.shape[0]))
            intersection = (
                torch.minimum(first[:, None, 1], second[None, :, 1])
                - torch.maximum(first[:, None, 0], second[None, :, 0])
            ).clamp_min(0.0)
            duration_first = first[:, 1] - first[:, 0]
            duration_second = second[:, 1] - second[:, 0]
            union = (
                duration_first[:, None] + duration_second[None]
                - intersection).clamp_min(1e-6)
            return intersection / union

        gains = anchors.new_zeros(anchors.shape[0])
        ap_utility = anchors.new_zeros(anchors.shape[0])
        if gt.shape[0] and anchors.shape[0]:
            anchor_overlap = pairwise(anchors, gt)
            selected_overlap = pairwise(selected, gt)
            same_class = labels[:, None] == gt_labels[None]
            anchor_overlap = anchor_overlap.masked_fill(~same_class, -1.0)
            selected_overlap = selected_overlap.masked_fill(
                ~same_class, -1.0)
            anchor_quality, assignment = anchor_overlap.max(dim=1)
            valid = anchor_quality > 0.0
            rows = torch.nonzero(valid, as_tuple=True)[0]
            if rows.numel():
                selected_quality = selected_overlap[
                    rows, assignment[rows]].clamp_min(0.0)
                gains[rows] = selected_quality - anchor_quality[rows]
                thresholds = anchors.new_tensor([
                    0.50, 0.55, 0.60, 0.65, 0.70,
                    0.75, 0.80, 0.85, 0.90, 0.95])
                anchor_credit = (
                    anchor_quality[rows, None]
                    >= thresholds[None]).to(anchors.dtype).mean(dim=1)
                selected_credit = (
                    selected_quality[:, None]
                    >= thresholds[None]).to(anchors.dtype).mean(dim=1)
                ap_utility[rows] = selected_credit - anchor_credit
        replaced_gain = gains[replacement]
        stats['p2d1_true_improvements'] = int(
            (replaced_gain > 1e-6).sum().item())
        stats['p2d1_true_harms'] = int(
            (replaced_gain < -1e-6).sum().item())
        stats['p2d1_true_neutral'] = int(
            (replaced_gain.abs() <= 1e-6).sum().item())
        stats['p2d1_true_gain_sum'] = float(replaced_gain.sum().item())
        stats['p5_true_ap_utility_sum'] = float(
            ap_utility[replacement].sum().item())

        if gt.shape[0]:
            anchor_gt = pairwise(anchors, gt)
            selected_gt = pairwise(selected, gt)
            same_class = labels[:, None] == gt_labels[None]
            anchor_best = anchor_gt.masked_fill(
                ~same_class, 0.0).max(dim=0).values
            selected_best = selected_gt.masked_fill(
                ~same_class, 0.0).max(dim=0).values
            for threshold in (0.85, 0.90, 0.95):
                key = str(threshold).replace('.', '_')
                before = anchor_best >= threshold
                after = selected_best >= threshold
                stats['p2d1_tiou_' + key + '_improved'] = int(
                    (~before & after).sum().item())
                stats['p2d1_tiou_' + key + '_harmed'] = int(
                    (before & ~after).sum().item())

    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        
        with torch.no_grad():
            output = model(video_list)

            # upack the results into ANet format
            num_vids = len(output)
            for vid_idx in range(num_vids):
                add_gt_diagnostics(output[vid_idx], video_list[vid_idx])
                add_p2d1_gt_diagnostics(
                    output[vid_idx], video_list[vid_idx])
                if output[vid_idx]['segments'].shape[0] > 0:
                    append_predictions(
                        results, output[vid_idx]['video_id'],
                        output[vid_idx]['segments'],
                        output[vid_idx]['scores'],
                        output[vid_idx]['labels'])
                if 'coarse_segments' in output[vid_idx]:
                    count = output[vid_idx]['coarse_segments'].shape[0]
                    if count > 0:
                        coarse_results['video-id'].extend([output[vid_idx]['video_id']] * count)
                        coarse_results['t-start'].append(output[vid_idx]['coarse_segments'][:, 0])
                        coarse_results['t-end'].append(output[vid_idx]['coarse_segments'][:, 1])
                        coarse_results['label'].append(output[vid_idx]['coarse_labels'])
                        coarse_results['score'].append(output[vid_idx]['coarse_scores'])
                combined_stats = {}
                if 'refinement_stats' in output[vid_idx]:
                    combined_stats.update(
                        output[vid_idx]['refinement_stats'])
                if 'candidate_selection_stats' in output[vid_idx]:
                    combined_stats.update(
                        output[vid_idx]['candidate_selection_stats'])
                if combined_stats:
                    refinement_stats.append(combined_stats)
                if 'oracle_segments' in output[vid_idx]:
                    if not output[vid_idx].get('_oracle_uses_ground_truth'):
                        raise RuntimeError(
                            "Oracle output is missing its GT-use watermark")
                    count = output[vid_idx]['oracle_segments'].shape[0]
                    if count > 0:
                        oracle_results['video-id'].extend(
                            [output[vid_idx]['video_id']] * count)
                        oracle_results['t-start'].append(
                            output[vid_idx]['oracle_segments'][:, 0])
                        oracle_results['t-end'].append(
                            output[vid_idx]['oracle_segments'][:, 1])
                        oracle_results['label'].append(
                            output[vid_idx]['oracle_labels'])
                        oracle_results['score'].append(
                            output[vid_idx]['oracle_scores'])
                if '_v13_candidate_segments' in output[vid_idx]:
                    if not output[vid_idx].get('_v13_uses_ground_truth'):
                        raise RuntimeError(
                            "V13 output is missing its GT-use watermark")
                    if v13_report_file is not None:
                        variants = build_v13_oracle_variants(
                            output[vid_idx]['_v13_candidate_segments'],
                            output[vid_idx]['_v13_candidate_scores'],
                            output[vid_idx]['_v13_candidate_labels'],
                            output[vid_idx]['_v13_gt_segments'],
                            output[vid_idx]['_v13_gt_labels'],
                            output[vid_idx]['_v13_nms_config'],
                            recall_thresholds=evaluator.tiou_thresholds,
                        )
                        for key, destination in (
                                ('score', v13_score_results),
                                ('boundary', v13_boundary_results),
                                ('joint', v13_joint_results)):
                            segments, scores, labels = variants[key]
                            append_predictions(
                                destination, output[vid_idx]['video_id'],
                                segments, scores, labels)
                        v13_pre_nms_recall.append(
                            variants['pre_nms_recall'])
                        v13_post_nms_recall.append(proposal_recall(
                            output[vid_idx]['segments'],
                            output[vid_idx]['labels'],
                            output[vid_idx]['_v13_gt_segments'],
                            output[vid_idx]['_v13_gt_labels'],
                            thresholds=evaluator.tiou_thresholds,
                        ))
                        v13_candidate_count += variants['candidate_count']
                        v13_replacement_count += \
                            variants['replacement_count']
                    if p2d0_report_file is not None:
                        for threshold in p2d0_cluster_thresholds:
                            p2d0_records[threshold].append(
                                build_cluster_oracle_record(
                                    output[vid_idx]['video_id'],
                                    output[vid_idx][
                                        '_v13_candidate_segments'],
                                    output[vid_idx]['_v13_candidate_scores'],
                                    output[vid_idx]['_v13_candidate_labels'],
                                    output[vid_idx]['_v13_gt_segments'],
                                    output[vid_idx]['_v13_gt_labels'],
                                    output[vid_idx]['_v13_nms_config'],
                                    threshold,
                                ))
                    if p3_score_report_file is not None:
                        p3_score_records.append(
                            build_bounded_score_record(
                                output[vid_idx]['video_id'],
                                output[vid_idx][
                                    '_v13_candidate_segments'],
                                output[vid_idx]['_v13_candidate_scores'],
                                output[vid_idx]['_v13_candidate_labels'],
                                output[vid_idx]['_v13_gt_segments'],
                                output[vid_idx]['_v13_gt_labels'],
                                output[vid_idx]['_v13_nms_config'],
                            ))
                    if p4_joint_report_file is not None:
                        p4_joint_records.append(
                            build_cluster_oracle_record(
                                output[vid_idx]['video_id'],
                                output[vid_idx][
                                    '_v13_candidate_segments'],
                                output[vid_idx]['_v13_candidate_scores'],
                                output[vid_idx]['_v13_candidate_labels'],
                                output[vid_idx]['_v13_gt_segments'],
                                output[vid_idx]['_v13_gt_labels'],
                                output[vid_idx]['_v13_nms_config'],
                                p4_cluster_threshold,
                            ))

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    def finalize(raw):
        raw['t-start'] = torch.cat(raw['t-start']).numpy() if raw['t-start'] else np.empty(0)
        raw['t-end'] = torch.cat(raw['t-end']).numpy() if raw['t-end'] else np.empty(0)
        raw['label'] = torch.cat(raw['label']).numpy() if raw['label'] else np.empty(0, dtype=np.int64)
        raw['score'] = torch.cat(raw['score']).numpy() if raw['score'] else np.empty(0)
        return raw
    results = finalize(results)
    has_coarse = bool(coarse_results['t-start'])
    if has_coarse:
        coarse_results = finalize(coarse_results)
    has_oracle = bool(oracle_results['t-start'])
    if has_oracle:
        oracle_results = finalize(oracle_results)
    has_v13 = bool(v13_pre_nms_recall)
    if has_v13:
        v13_score_results = finalize(v13_score_results)
        v13_boundary_results = finalize(v13_boundary_results)
        v13_joint_results = finalize(v13_joint_results)
    if oracle_report_file is not None and not has_oracle:
        raise RuntimeError(
            "Oracle report requested, but model produced no oracle predictions")
    if v13_report_file is not None and not has_v13:
        raise RuntimeError(
            "V13 report requested, but model produced no V13 candidates")
    has_p2d0 = any(p2d0_records.values())
    if p2d0_report_file is not None and not has_p2d0:
        raise RuntimeError(
            "P2-D0 report requested, but model produced no V13 candidates")
    if p3_score_report_file is not None and not p3_score_records:
        raise RuntimeError(
            "P3-0 score report requested, but model produced no V13 "
            "candidates")
    if p4_joint_report_file is not None and not p4_joint_records:
        raise RuntimeError(
            "P4-0 joint report requested, but model produced no V13 "
            "candidates")

    if evaluator is not None:
        if (ext_score_file is not None) and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
            if has_coarse:
                coarse_results = postprocess_results(coarse_results, ext_score_file)
            if has_oracle:
                oracle_results = postprocess_results(oracle_results, ext_score_file)
            if has_v13:
                v13_score_results = postprocess_results(
                    v13_score_results, ext_score_file)
                v13_boundary_results = postprocess_results(
                    v13_boundary_results, ext_score_file)
                v13_joint_results = postprocess_results(
                    v13_joint_results, ext_score_file)
        if has_v13:
            print("\nV13 Phase A baseline:")
            baseline_curve, mAP = evaluator.evaluate(results, verbose=True)
            print("\nScore Oracle "
                  "(USES GROUND TRUTH; NOT A MODEL RESULT):")
            score_curve, score_mAP = evaluator.evaluate(
                v13_score_results, verbose=True)
            print("\nBoundary/Proposal Oracle "
                  "(USES GROUND TRUTH; NOT A MODEL RESULT):")
            boundary_curve, boundary_mAP = evaluator.evaluate(
                v13_boundary_results, verbose=True)
            print("\nJoint Oracle "
                  "(USES GROUND TRUTH; NOT A MODEL RESULT):")
            joint_curve, joint_mAP = evaluator.evaluate(
                v13_joint_results, verbose=True)
            print("\nV13 Phase A gains over baseline:")
            print("  Score: {:+.2f} mAP".format(
                (score_mAP - mAP) * 100))
            print("  Boundary/Proposal: {:+.2f} mAP".format(
                (boundary_mAP - mAP) * 100))
            print("  Joint: {:+.2f} mAP".format(
                (joint_mAP - mAP) * 100))
        elif has_coarse:
            print("\nBaseline coarse:")
            coarse_curve, coarse_mAP = evaluator.evaluate(coarse_results, verbose=True)
            print("\nDiffusion refined:")
            refined_curve, mAP = evaluator.evaluate(results, verbose=True)
            print("\nDelta:")
            print(" / ".join("{:+.2f}".format(value * 100)
                             for value in refined_curve - coarse_curve)
                  + " / Avg. {:+.2f}".format((mAP - coarse_mAP) * 100))
            if has_oracle:
                print("\nOracle diagnostic (USES GROUND TRUTH; NOT A MODEL RESULT):")
                oracle_curve, oracle_mAP = evaluator.evaluate(
                    oracle_results, verbose=True)
                print("\nOracle gain over coarse:")
                print(" / ".join("{:+.2f}".format(value * 100)
                                 for value in oracle_curve - coarse_curve)
                      + " / Avg. {:+.2f}".format(
                          (oracle_mAP - coarse_mAP) * 100))
        else:
            baseline_curve, mAP = evaluator.evaluate(results, verbose=True)
    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0

    if output_file is not None and evaluator is not None:
        root, ext = os.path.splitext(output_file)
        with open(root + '.refined' + ext, 'wb') as f:
            pickle.dump(results, f)
        if has_coarse:
            with open(root + '.coarse' + ext, 'wb') as f:
                pickle.dump(coarse_results, f)
        if has_oracle:
            with open(root + '.oracle' + ext, 'wb') as f:
                pickle.dump(oracle_results, f)
        if has_v13:
            for suffix, variant in (
                    ('score_oracle', v13_score_results),
                    ('boundary_oracle', v13_boundary_results),
                    ('joint_oracle', v13_joint_results)):
                with open(root + '.v13_' + suffix + ext, 'wb') as f:
                    pickle.dump(variant, f)
    if p2d0_report_file is not None:
        def evaluate_p2d0_variant(records, masks):
            raw = {
                'video-id': [], 't-start': [], 't-end': [],
                'label': [], 'score': [],
            }
            for record, mask in zip(records, masks):
                segments, scores, labels = render_cluster_predictions(
                    record, mask)
                append_predictions(
                    raw, record["video_id"], segments, scores, labels)
            predictions = finalize(raw)
            if (ext_score_file is not None
                    and isinstance(ext_score_file, str)):
                predictions = postprocess_results(
                    predictions, ext_score_file)
            return evaluator.evaluate(predictions, verbose=False)

        threshold_reports = {}
        best_full_gain = float("-inf")
        best_threshold = None
        print("\nP2-D0 constrained cluster oracle "
              "(USES GROUND TRUTH; NOT A MODEL RESULT):")
        for threshold in p2d0_cluster_thresholds:
            records = p2d0_records[threshold]
            empty_masks = [
                torch.zeros_like(
                    record["improvements"], dtype=torch.bool)
                for record in records
            ]
            all_masks = full_oracle_masks(records)
            anchor_curve, anchor_mAP = evaluate_p2d0_variant(
                records, empty_masks)
            full_curve, full_mAP = evaluate_p2d0_variant(
                records, all_masks)
            full_gain = float((full_mAP - mAP) * 100)
            best_full_gain = max(best_full_gain, full_gain)
            if best_threshold is None or full_gain == best_full_gain:
                best_threshold = threshold

            budget_reports = {}
            budget_masks = global_replacement_mask_sweep(
                records, p2d0_replacement_budgets)
            for budget in p2d0_replacement_budgets:
                masks, budget_stats = budget_masks[budget]
                curve, budget_mAP = evaluate_p2d0_variant(records, masks)
                budget_key = "{:g}_percent".format(budget * 100)
                budget_reports[budget_key] = {
                    **budget_stats,
                    "average_mAP_percent": float(budget_mAP * 100),
                    "gain_over_a0_baseline_percent": float(
                        (budget_mAP - mAP) * 100),
                    "gain_over_cluster_anchor_percent": float(
                        (budget_mAP - anchor_mAP) * 100),
                    "mAP_percent": [
                        float(value * 100) for value in curve],
                    "high_tiou_gt_coverage": high_tiou_crossings(
                        records, masks),
                }

            statistics = summarize_cluster_records(records)
            threshold_key = "{:.10g}".format(threshold)
            threshold_reports[threshold_key] = {
                "cluster_threshold": float(threshold),
                "statistics": statistics,
                "cluster_anchor_baseline": {
                    "average_mAP_percent": float(anchor_mAP * 100),
                    "gain_over_a0_baseline_percent": float(
                        (anchor_mAP - mAP) * 100),
                    "mAP_percent": [
                        float(value * 100) for value in anchor_curve],
                },
                "full_constrained_oracle": {
                    "average_mAP_percent": float(full_mAP * 100),
                    "gain_over_a0_baseline_percent": full_gain,
                    "gain_over_cluster_anchor_percent": float(
                        (full_mAP - anchor_mAP) * 100),
                    "replacement_count": int(sum(
                        int(mask.sum().item()) for mask in all_masks)),
                    "mAP_percent": [
                        float(value * 100) for value in full_curve],
                    "high_tiou_gt_coverage":
                        statistics["high_tiou_gt_coverage"],
                },
                "replacement_budgets": budget_reports,
            }
            print("  cluster_tIoU={:.1f}: anchor={:.2f}, "
                  "full_oracle={:.2f}, gain_vs_A0={:+.2f}".format(
                      threshold, anchor_mAP * 100, full_mAP * 100,
                      full_gain))

        if best_full_gain >= 3.0:
            route = "ENTER_P2D_TRAINING"
            explanation = (
                "Best deployability-constrained oracle gain is at least "
                "+3 mAP; train a cluster-level selector.")
        elif best_full_gain >= 2.0:
            route = "ONLY_CONSERVATIVE_HIGH_CONFIDENCE_SELECTION"
            explanation = (
                "Constrained headroom is between +2 and +3 mAP; only a "
                "high-confidence, low-budget selector is justified.")
        else:
            route = "STOP_CANDIDATE_SELECTION_ROUTE"
            explanation = (
                "Constrained headroom is below +2 mAP; the earlier per-anchor "
                "oracle does not provide deployable candidate-selection "
                "headroom.")
        report = {
            "p2d0_deployable_constrained_oracle": True,
            "diagnostic_protocol": "activitynet_p2d0_cluster_oracle",
            "dataset_name": diagnostic_dataset_name,
            "diagnostic_only": True,
            "uses_ground_truth": True,
            "must_not_be_reported_as_model_performance": True,
            "trains_or_changes_model": False,
            "uses_only_existing_candidate_boundaries": True,
            "keeps_cluster_anchor_a0_score_and_label": True,
            "nms_external_score_and_max_proposals_unchanged": True,
            "cluster_definition": (
                "Greedy NMS-style same-label local clusters: the highest A0 "
                "score unassigned proposal is the anchor; unassigned proposals "
                "with anchor tIoU >= threshold join that cluster. No "
                "transitive closure and no per-anchor independent replacement."
            ),
            "replacement_budget_denominator": (
                "all deployable clusters across the complete validation set; "
                "only positive-GT-gain replacements can consume the quota"
            ),
            "a0_baseline": {
                "average_mAP_percent": float(mAP * 100),
                "mAP_percent": [
                    float(value * 100) for value in baseline_curve],
            },
            "cluster_thresholds": threshold_reports,
            "decision": {
                "recommended_route": route,
                "best_cluster_threshold": float(best_threshold),
                "best_full_constrained_oracle_gain_percent": float(
                    best_full_gain),
                "train_gate_percent": 3.0,
                "conservative_only_lower_bound_percent": 2.0,
                "explanation": explanation,
            },
            "tiou_thresholds": [
                float(value) for value in evaluator.tiou_thresholds],
        }
        with open(p2d0_report_file, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    if p3_score_report_file is not None:
        def evaluate_p3_score_variant(records, masks, logit_cap):
            raw = {
                'video-id': [], 't-start': [], 't-end': [],
                'label': [], 'score': [],
            }
            rendered = []
            for record, mask in zip(records, masks):
                prediction = render_bounded_score_predictions(
                    record, mask, logit_cap)
                rendered.append(prediction)
                segments, scores, labels = prediction
                append_predictions(
                    raw, record["video_id"], segments, scores, labels)
            statistics = summarize_score_adjustments(
                records, masks, logit_cap,
                rendered_predictions=rendered)
            predictions = finalize(raw)
            if (ext_score_file is not None
                    and isinstance(ext_score_file, str)):
                predictions = postprocess_results(
                    predictions, ext_score_file)
            curve, average = evaluator.evaluate(
                predictions, verbose=False)
            return curve, average, statistics

        verified_a0 = 38.37534459002466
        empty_masks = [
            torch.zeros_like(record["scores"], dtype=torch.bool)
            for record in p3_score_records
        ]
        identity_curve, identity_mAP, identity_statistics = \
            evaluate_p3_score_variant(
                p3_score_records, empty_masks, 0.0)
        baseline_percent = float(mAP * 100)
        identity_percent = float(identity_mAP * 100)
        tolerance = 0.02
        identity_passed = (
            abs(baseline_percent - verified_a0) <= tolerance
            and abs(identity_percent - baseline_percent) <= tolerance)

        budget_masks = global_score_adjustment_mask_sweep(
            p3_score_records, p3_score_budgets)
        cap_reports = {}
        best = None
        print("\nP3-0 bounded score oracle "
              "(USES GROUND TRUTH; NOT A MODEL RESULT):")
        for cap in p3_score_logit_caps:
            budget_reports = {}
            for budget in p3_score_budgets:
                masks, budget_statistics = budget_masks[budget]
                curve, average, adjustment_statistics = \
                    evaluate_p3_score_variant(
                        p3_score_records, masks, cap)
                result = {
                    **budget_statistics,
                    **adjustment_statistics,
                    "logit_adjustment_cap": float(cap),
                    "average_mAP_percent": float(average * 100),
                    "gain_over_verified_a0_percent": float(
                        average * 100 - verified_a0),
                    "gain_over_runtime_a0_percent": float(
                        (average - mAP) * 100),
                    "gain_over_identity_percent": float(
                        (average - identity_mAP) * 100),
                    "mAP_percent": [
                        float(value * 100) for value in curve],
                }
                budget_key = "{:g}_percent".format(budget * 100)
                budget_reports[budget_key] = result
                if (best is None or result["average_mAP_percent"]
                        > best["average_mAP_percent"]):
                    best = dict(result)
                    best["budget_key"] = budget_key
                print("  cap={:.2f}, budget={:g}%: mAP={:.2f}, "
                      "gain_vs_A0={:+.2f}".format(
                          cap, budget * 100,
                          result["average_mAP_percent"],
                          result["gain_over_verified_a0_percent"]))
            cap_reports["{:.10g}".format(cap)] = {
                "logit_adjustment_cap": float(cap),
                "budgets": budget_reports,
            }

        best_gain = best["gain_over_verified_a0_percent"]
        if not identity_passed:
            route = "INVALID_P3_IDENTITY"
            explanation = (
                "The A0 or no-adjustment identity gate failed; do not "
                "interpret the score oracle.")
        elif best_gain >= 3.0:
            route = "ENTER_P3A_TASK_ALIGNED_TRAINING"
            explanation = (
                "Bounded score headroom is at least +3 mAP; train the "
                "original proposal-score branch with localization-aware "
                "targets.")
        elif best_gain >= 2.0:
            route = "ONLY_CONSERVATIVE_P3A"
            explanation = (
                "Bounded score headroom is between +2 and +3 mAP; only a "
                "conservative score-alignment fine-tune is justified.")
        else:
            route = "STOP_SCORE_ALIGNMENT_ROUTE"
            explanation = (
                "Bounded score headroom is below +2 mAP; do not train or "
                "expand P3-A score alignment.")
        report = {
            "p3_bounded_score_oracle": True,
            "diagnostic_protocol": "activitynet_p3_bounded_score_oracle",
            "dataset_name": diagnostic_dataset_name,
            "diagnostic_only": True,
            "uses_ground_truth": True,
            "must_not_be_reported_as_model_performance": True,
            "trains_or_changes_model": False,
            "fixed_candidate_boundaries_and_labels": True,
            "nms_external_score_and_max_proposals_unchanged": True,
            "score_adjustment": (
                "For the globally selected budget only, move each A0 score "
                "logit toward its class-matched true-tIoU logit by no more "
                "than the configured absolute cap. Unselected proposal "
                "scores are bitwise unchanged."
            ),
            "score_adjustment_selection": (
                "GT-only upper bound: rank proposals globally by absolute "
                "mismatch between A0 score logit and class-matched true-tIoU "
                "logit, with deterministic original-order tie breaking."
            ),
            "score_budget_denominator": (
                "all pre-NMS candidate proposals across the complete "
                "ActivityNet validation set"
            ),
            "verified_a0_average_mAP_percent": verified_a0,
            "runtime_a0": {
                "average_mAP_percent": baseline_percent,
                "mAP_percent": [
                    float(value * 100) for value in baseline_curve],
            },
            "no_adjustment_identity": {
                "average_mAP_percent": identity_percent,
                "mAP_percent": [
                    float(value * 100) for value in identity_curve],
                "statistics": identity_statistics,
                "absolute_tolerance_percent": tolerance,
                "passed": bool(identity_passed),
            },
            "logit_caps": cap_reports,
            "best_bounded_score_result": best,
            "decision": {
                "recommended_route": route,
                "best_gain_over_verified_a0_percent": float(best_gain),
                "train_gate_percent": 3.0,
                "conservative_only_lower_bound_percent": 2.0,
                "target_model_gain_percent": 2.0,
                "explanation": explanation,
            },
            "tiou_thresholds": [
                float(value) for value in evaluator.tiou_thresholds],
        }
        with open(p3_score_report_file, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        if not identity_passed:
            raise RuntimeError("P3-0 no-adjustment identity gate failed")
    if p4_joint_report_file is not None:
        def evaluate_p4_joint_variant(records, masks):
            raw = {
                'video-id': [], 't-start': [], 't-end': [],
                'label': [], 'score': [],
            }
            rendered = []
            for record, mask in zip(records, masks):
                prediction = render_bounded_score_predictions(
                    record, mask, p4_score_logit_cap)
                rendered.append(prediction)
                segments, scores, labels = prediction
                append_predictions(
                    raw, record["video_id"], segments, scores, labels)
            score_statistics = summarize_score_adjustments(
                records, masks, p4_score_logit_cap,
                rendered_predictions=rendered)
            predictions = finalize(raw)
            if (ext_score_file is not None
                    and isinstance(ext_score_file, str)):
                predictions = postprocess_results(
                    predictions, ext_score_file)
            curve, average = evaluator.evaluate(
                predictions, verbose=False)
            return curve, average, score_statistics

        def budget_key(value):
            return "{:g}_percent".format(float(value) * 100)

        verified_a0 = 38.37534459002466
        expected_anchor = 38.64203731665842
        tolerance = 0.02
        boundary_sweeps = global_replacement_mask_sweep(
            p4_joint_records, p4_boundary_budgets)
        matrix = {}
        flat_results = {}
        print("\nP4-0 constrained joint oracle "
              "(USES GROUND TRUTH; NOT A MODEL RESULT):")
        for boundary_budget in p4_boundary_budgets:
            boundary_masks, boundary_budget_statistics = \
                boundary_sweeps[boundary_budget]
            score_records = build_joint_score_records(
                p4_joint_records, boundary_masks)
            boundary_statistics = summarize_boundary_replacements(
                score_records)
            score_sweeps = global_score_adjustment_mask_sweep(
                score_records, p4_score_budgets)
            score_reports = {}
            for score_budget in p4_score_budgets:
                score_masks, score_budget_statistics = \
                    score_sweeps[score_budget]
                curve, average, score_statistics = \
                    evaluate_p4_joint_variant(
                        score_records, score_masks)
                result = {
                    "cluster_threshold": float(p4_cluster_threshold),
                    "boundary_budget_fraction": float(boundary_budget),
                    "score_budget_fraction": float(score_budget),
                    "score_logit_adjustment_cap": float(
                        p4_score_logit_cap),
                    "average_mAP_percent": float(average * 100),
                    "gain_over_verified_a0_percent": float(
                        average * 100 - verified_a0),
                    "gain_over_runtime_a0_percent": float(
                        (average - mAP) * 100),
                    "mAP_percent": [
                        float(value * 100) for value in curve],
                    "boundary_budget_statistics":
                        boundary_budget_statistics,
                    "score_budget_statistics": score_budget_statistics,
                    "boundary_statistics": boundary_statistics,
                    "score_statistics": score_statistics,
                }
                flat_results[(boundary_budget, score_budget)] = result
                score_reports[budget_key(score_budget)] = result
                print("  boundary={:g}%, score={:g}%: mAP={:.2f}, "
                      "gain_vs_A0={:+.2f}".format(
                          boundary_budget * 100, score_budget * 100,
                          result["average_mAP_percent"],
                          result["gain_over_verified_a0_percent"]))
            matrix[budget_key(boundary_budget)] = {
                "boundary_budget_fraction": float(boundary_budget),
                "score_budgets": score_reports,
            }

        anchor = flat_results[(0.0, 0.0)]
        anchor_percent = anchor["average_mAP_percent"]
        identity_passed = (
            abs(float(mAP * 100) - verified_a0) <= tolerance
            and abs(anchor_percent - expected_anchor) <= tolerance)
        joint_results = []
        for boundary_budget in p4_boundary_budgets:
            for score_budget in p4_score_budgets:
                result = flat_results[(boundary_budget, score_budget)]
                boundary_only = flat_results[(boundary_budget, 0.0)][
                    "average_mAP_percent"]
                score_only = flat_results[(0.0, score_budget)][
                    "average_mAP_percent"]
                result["gain_over_cluster_anchor_percent"] = float(
                    result["average_mAP_percent"] - anchor_percent)
                result["increment_over_boundary_only_percent"] = float(
                    result["average_mAP_percent"] - boundary_only)
                result["increment_over_score_only_percent"] = float(
                    result["average_mAP_percent"] - score_only)
                result["joint_synergy_percent"] = float(
                    result["average_mAP_percent"]
                    - boundary_only - score_only + anchor_percent)
                if boundary_budget > 0.0 and score_budget > 0.0:
                    joint_results.append(result)
        best_joint = max(
            joint_results,
            key=lambda item: item["average_mAP_percent"])
        best_gain = best_joint["gain_over_verified_a0_percent"]
        score_increment = best_joint[
            "increment_over_boundary_only_percent"]
        if not identity_passed:
            route = "INVALID_P4_IDENTITY"
            explanation = (
                "The A0 or cluster-anchor identity gate failed; do not "
                "interpret the joint oracle.")
        elif best_gain >= 3.0:
            route = "ENTER_P4_TASK_ALIGNED_PROPOSAL_TRAINING"
            explanation = (
                "The best constrained joint boundary-and-score result has "
                "at least +3 mAP headroom; an end-to-end P4 training stage "
                "is justified.")
        elif best_gain >= 2.0:
            route = "ONLY_CONSERVATIVE_P4_TRAINING"
            explanation = (
                "Joint constrained headroom is between +2 and +3 mAP; only "
                "a conservative end-to-end P4 experiment is justified.")
        else:
            route = "STOP_JOINT_PROPOSAL_SCORE_ROUTE"
            explanation = (
                "Joint constrained headroom is below +2 mAP; stop proposal "
                "and score coupling experiments.")
        report = {
            "p4_constrained_joint_oracle": True,
            "diagnostic_protocol": (
                "activitynet_p4_constrained_joint_boundary_score_oracle"),
            "dataset_name": diagnostic_dataset_name,
            "diagnostic_only": True,
            "uses_ground_truth": True,
            "must_not_be_reported_as_model_performance": True,
            "trains_or_changes_model": False,
            "uses_only_existing_candidate_boundaries": True,
            "one_candidate_per_cluster": True,
            "keeps_cluster_anchor_label": True,
            "nms_external_score_and_max_proposals_unchanged": True,
            "cluster_threshold": float(p4_cluster_threshold),
            "score_logit_adjustment_cap": float(p4_score_logit_cap),
            "verified_a0_average_mAP_percent": verified_a0,
            "runtime_a0": {
                "average_mAP_percent": float(mAP * 100),
                "mAP_percent": [
                    float(value * 100) for value in baseline_curve],
            },
            "cluster_anchor_identity": {
                "expected_average_mAP_percent": expected_anchor,
                "average_mAP_percent": anchor_percent,
                "mAP_percent": anchor["mAP_percent"],
                "absolute_tolerance_percent": tolerance,
                "passed": bool(identity_passed),
            },
            "matrix": matrix,
            "best_joint_result": best_joint,
            "decision": {
                "recommended_route": route,
                "best_joint_gain_over_verified_a0_percent": float(
                    best_gain),
                "best_joint_increment_over_boundary_only_percent": float(
                    score_increment),
                "best_joint_synergy_percent": float(
                    best_joint["joint_synergy_percent"]),
                "score_coupling_is_material": bool(
                    score_increment >= 0.3),
                "train_gate_percent": 3.0,
                "conservative_only_lower_bound_percent": 2.0,
                "material_score_increment_percent": 0.3,
                "target_model_gain_percent": 2.0,
                "explanation": explanation,
            },
            "tiou_thresholds": [
                float(value) for value in evaluator.tiou_thresholds],
        }
        with open(p4_joint_report_file, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        if not identity_passed:
            raise RuntimeError("P4-0 cluster-anchor identity gate failed")
    if p2d1_report_file is not None:
        if not any(
                'p2d1_cluster_count' in item
                for item in refinement_stats):
            raise RuntimeError(
                "P2-D1 report requested, but selector stats are missing")
        selector_stats = [
            item for item in refinement_stats
            if 'p2d1_cluster_count' in item]

        def total(key):
            return sum(float(item.get(key, 0.0))
                       for item in selector_stats)

        clusters = int(total('p2d1_cluster_count'))
        replacements = int(total('p2d1_replacements'))
        improvements = int(total('p2d1_true_improvements'))
        harms = int(total('p2d1_true_harms'))
        neutral = int(total('p2d1_true_neutral'))
        true_gain_sum = total('p2d1_true_gain_sum')
        predicted_gain_sum = sum(
            float(item.get('p2d1_mean_predicted_gain', 0.0))
            * int(item.get('p2d1_replacements', 0))
            for item in selector_stats)
        confidence_sum = sum(
            float(item.get('p2d1_mean_replace_confidence', 0.0))
            * int(item.get('p2d1_replacements', 0))
            for item in selector_stats)
        true_gain_mean = (
            true_gain_sum / replacements if replacements else 0.0)
        predicted_gain_mean = (
            predicted_gain_sum / replacements if replacements else 0.0)
        module = model.module if hasattr(model, 'module') else model
        ap_listwise = getattr(
            module, 'cluster_selector_type', 'pointwise_gain') == \
            'ap_listwise'
        true_ap_utility_mean = (
            total('p5_true_ap_utility_sum') / replacements
            if replacements else 0.0)
        score_suppression_enabled = bool(getattr(
            module, 'cluster_score_suppression_enabled', False))
        score_suppressions = int(total('p4a_score_suppressions'))
        score_suppression_amount_sum = sum(
            float(item.get(
                'p4a_mean_score_logit_suppression', 0.0))
            * int(item.get('p4a_score_suppressions', 0))
            for item in selector_stats)
        score_suppression_confidence_sum = sum(
            float(item.get(
                'p4a_mean_score_suppression_confidence', 0.0))
            * int(item.get('p4a_score_suppressions', 0))
            for item in selector_stats)
        high_tiou = {}
        for threshold in (0.85, 0.90, 0.95):
            key = str(threshold).replace('.', '_')
            high_tiou["{:.2f}".format(threshold)] = {
                "improved_count": int(total(
                    'p2d1_tiou_' + key + '_improved')),
                "harmed_count": int(total(
                    'p2d1_tiou_' + key + '_harmed')),
            }
        report = {
            "p2d1_cluster_selector": True,
            "p5_ap_listwise_joint_selector": ap_listwise,
            "p4a_joint_alignment": score_suppression_enabled,
            "calibration_only": bool(p2d1_calibration_only),
            "dataset_name": diagnostic_dataset_name,
            "uses_ground_truth_during_inference": False,
            "existing_candidate_boundaries_only": True,
            "keeps_anchor_score_and_label":
                not score_suppression_enabled,
            "keeps_anchor_label": True,
            "score_adjustment_is_downward_only":
                score_suppression_enabled,
            "cluster_threshold": float(
                module.cluster_overlap_threshold),
            "minimum_predicted_gain": float(
                module.cluster_min_predicted_gain),
            "minimum_replace_confidence": float(
                module.cluster_min_confidence),
            "max_replacement_fraction": float(
                module.cluster_max_replacement_fraction),
            "selection": {
                "video_count": len(selector_stats),
                "candidate_pool_size": int(total(
                    'p2d1_candidate_pool_size')),
                "cluster_count": clusters,
                "non_singleton_cluster_count": int(total(
                    'p2d1_non_singleton_clusters')),
                "eligible_replacement_count": int(total(
                    'p2d1_eligible_replacements')),
                "replacement_count": replacements,
                "fallback_count": max(clusters - replacements, 0),
                "replacement_ratio": (
                    replacements / clusters if clusters else 0.0),
                "true_improvement_count": improvements,
                "true_harm_count": harms,
                "true_neutral_count": neutral,
                "true_improvement_precision": (
                    improvements / replacements
                    if replacements else 0.0),
                "mean_true_gain": true_gain_mean,
                "mean_predicted_gain": predicted_gain_mean,
                "prediction_target": (
                    "activitynet_ap_threshold_utility"
                    if ap_listwise else "fixed_gt_tiou_gain"),
                "mean_true_ap_utility": true_ap_utility_mean,
                "mean_predicted_ap_utility": (
                    predicted_gain_mean if ap_listwise else 0.0),
                "mean_replace_confidence": (
                    confidence_sum / replacements
                    if replacements else 0.0),
                "mean_gain_calibration_error": (
                    predicted_gain_mean
                    - (true_ap_utility_mean
                       if ap_listwise else true_gain_mean)),
                "mean_gain_calibration_absolute_error": abs(
                    predicted_gain_mean
                    - (true_ap_utility_mean
                       if ap_listwise else true_gain_mean)),
            },
            "high_tiou_gt_coverage": high_tiou,
        }
        residual_stats = [
            item for item in refinement_stats
            if 'p4b_valid_class_predictions' in item]
        if residual_stats:
            valid_class_predictions = sum(
                int(item.get('p4b_valid_class_predictions', 0))
                for item in residual_stats)
            valid_regression_predictions = sum(
                int(item.get('p4b_valid_regression_predictions', 0))
                for item in residual_stats)
            active_class_residuals = sum(
                int(item.get('p4b_active_class_residuals', 0))
                for item in residual_stats)
            active_regression_residuals = sum(
                int(item.get('p4b_active_regression_residuals', 0))
                for item in residual_stats)
            class_residual_sum = sum(
                float(item.get(
                    'p4b_mean_absolute_class_residual', 0.0))
                * int(item.get('p4b_valid_class_predictions', 0))
                for item in residual_stats)
            regression_residual_sum = sum(
                float(item.get(
                    'p4b_mean_absolute_regression_residual', 0.0))
                * int(item.get('p4b_valid_regression_predictions', 0))
                for item in residual_stats)
            uses_ground_truth = any(bool(item.get(
                'p4b_uses_ground_truth_during_inference', False))
                for item in residual_stats)
            report["residual_alignment"] = {
                "video_count": len(residual_stats),
                "residual_weight": float(residual_stats[0].get(
                    'p4b_residual_weight', 0.0)),
                "minimum_confidence": float(residual_stats[0].get(
                    'p4b_minimum_confidence', 0.0)),
                "valid_class_predictions": valid_class_predictions,
                "valid_regression_predictions":
                    valid_regression_predictions,
                "active_class_residuals": active_class_residuals,
                "active_regression_residuals":
                    active_regression_residuals,
                "class_active_ratio": (
                    active_class_residuals / valid_class_predictions
                    if valid_class_predictions else 0.0),
                "regression_active_ratio": (
                    active_regression_residuals
                    / valid_regression_predictions
                    if valid_regression_predictions else 0.0),
                "mean_absolute_class_residual": (
                    class_residual_sum / valid_class_predictions
                    if valid_class_predictions else 0.0),
                "mean_absolute_regression_residual": (
                    regression_residual_sum
                    / valid_regression_predictions
                    if valid_regression_predictions else 0.0),
                "uses_ground_truth_during_inference": uses_ground_truth,
            }
        if score_suppression_enabled:
            report["score_suppression"] = {
                "eligible_count": int(total(
                    'p4a_score_suppression_eligible')),
                "suppression_count": score_suppressions,
                "suppression_ratio": (
                    score_suppressions / clusters if clusters else 0.0),
                "fallback_count": max(
                    clusters - score_suppressions, 0),
                "mean_predicted_logit_suppression": (
                    score_suppression_amount_sum / score_suppressions
                    if score_suppressions else 0.0),
                "mean_suppression_confidence": (
                    score_suppression_confidence_sum / score_suppressions
                    if score_suppressions else 0.0),
                "maximum_logit_suppression": float(
                    module.cluster_score_max_logit_suppression),
                "max_suppression_fraction": float(
                    module.cluster_score_max_suppression_fraction),
            }
        if p2d1_calibration_only:
            report["calibration_protocol"] = {
                "subset": "p2d1_calibration",
                "external_classification_fusion_used": False,
                "official_mAP_computed": False,
                "purpose": (
                    "Choose an inference confidence threshold using only "
                    "selector replacement rate and fixed-GT gain statistics."
                ),
            }
        else:
            report["official"] = {
                "average_mAP_percent": float(mAP * 100),
                "gain_over_verified_a0_percent": float(
                    mAP * 100 - 38.37534459002466),
                "mAP_percent": [
                    float(value * 100) for value in baseline_curve],
            }
            report["decision"] = {
                "target_average_mAP_percent": 40.37534459002466,
                "target_achieved": bool(
                    mAP * 100 >= 40.37534459002466),
                "continue_gate_gain_percent": 0.5,
                "continue_gate_passed": bool(
                    mAP * 100 >= 38.87534459002466),
                "stop_below_gain_percent": 0.3,
                "stop_gate_triggered": bool(
                    mAP * 100 < 38.67534459002466),
            }
        with open(p2d1_report_file, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    if oracle_report_file is not None:
        total_candidates = sum(
            int(item.get('oracle_candidates', 0))
            for item in refinement_stats)
        total_accepts = sum(
            int(item.get('oracle_accepts', 0))
            for item in refinement_stats)
        accepted_delta_sum = sum(
            float(item.get('oracle_accepted_delta_tiou_sum', 0.0))
            for item in refinement_stats)
        generated_candidates = sum(
            int(item.get('oracle_generated_candidates', 0))
            for item in refinement_stats)
        weighted_pairwise_tiou = sum(
            float(item.get('oracle_mean_pairwise_candidate_tiou', 1.0))
            * int(item.get('oracle_candidates', 0))
            for item in refinement_stats)
        num_candidates_per_proposal = max((
            int(item.get('oracle_num_candidates_per_proposal', 1))
            for item in refinement_stats), default=1)
        report = {
            "diagnostic_only": True,
            "uses_ground_truth": True,
            "must_not_be_reported_as_model_performance": True,
            "coarse_average_mAP_percent": float(coarse_mAP * 100),
            "refined_average_mAP_percent": float(mAP * 100),
            "oracle_average_mAP_percent": float(oracle_mAP * 100),
            "oracle_gain_over_coarse_percent": float(
                (oracle_mAP - coarse_mAP) * 100),
            "tiou_thresholds": [float(value) for value in evaluator.tiou_thresholds],
            "coarse_mAP_percent": [float(value * 100) for value in coarse_curve],
            "refined_mAP_percent": [float(value * 100) for value in refined_curve],
            "oracle_mAP_percent": [float(value * 100) for value in oracle_curve],
            "oracle_candidates": total_candidates,
            "oracle_num_candidates_per_proposal": num_candidates_per_proposal,
            "oracle_generated_candidates": generated_candidates,
            "oracle_accepts": total_accepts,
            "oracle_accept_ratio": (
                float(total_accepts / total_candidates)
                if total_candidates else 0.0),
            "oracle_mean_accepted_delta_tiou": (
                float(accepted_delta_sum / total_accepts)
                if total_accepts else 0.0),
            "oracle_mean_pairwise_candidate_tiou": (
                float(weighted_pairwise_tiou / total_candidates)
                if total_candidates else 1.0),
        }
        with open(oracle_report_file, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    if v13_report_file is not None:
        pre_nms_recall = merge_recall_statistics(v13_pre_nms_recall)
        post_nms_recall = merge_recall_statistics(v13_post_nms_recall)
        gains = {
            "score": float((score_mAP - mAP) * 100),
            "boundary_proposal": float((boundary_mAP - mAP) * 100),
            "joint": float((joint_mAP - mAP) * 100),
        }
        dataset_name = diagnostic_dataset_name or "unspecified"
        route_margin = 0.25
        if dataset_name == "anet":
            if gains["boundary_proposal"] >= 3.0:
                recommended_route = "P2_boundary_refinement"
            elif gains["joint"] >= 5.0:
                recommended_route = "P2_geometry_quality_score_coupling"
            elif gains["joint"] < 3.0:
                recommended_route = "STOP_current_proposal_pool_insufficient"
            else:
                recommended_route = "REVIEW_mixed_or_marginal_headroom"
        elif gains["score"] >= 1.5 and gains["boundary_proposal"] >= 1.5:
            recommended_route = "V13-A_then_V13-C"
        elif gains["score"] > gains["boundary_proposal"] + route_margin:
            recommended_route = "V13-A_Task-Aligned_DyFADet"
        elif gains["boundary_proposal"] > gains["score"] + route_margin:
            recommended_route = "V13-C_boundary-aware_proposals"
        else:
            recommended_route = "V13-A_then_V13-C"
        report = {
            "v13_phase_a": True,
            "diagnostic_protocol": (
                "activitynet_p1_headroom"
                if dataset_name == "anet" else "v13_phase_a"),
            "dataset_name": dataset_name,
            "diagnostic_only": True,
            "uses_ground_truth": True,
            "must_not_be_reported_as_model_performance": True,
            "uses_only_existing_candidate_boundaries": True,
            "baseline": {
                "average_mAP_percent": float(mAP * 100),
                "mAP_percent": [
                    float(value * 100) for value in baseline_curve],
            },
            "score_oracle": {
                "average_mAP_percent": float(score_mAP * 100),
                "gain_over_baseline_percent": gains["score"],
                "mAP_percent": [
                    float(value * 100) for value in score_curve],
                "fixed_boundaries": True,
                "ground_truth_tiou_replaces_score": True,
            },
            "boundary_proposal_oracle": {
                "average_mAP_percent": float(boundary_mAP * 100),
                "gain_over_baseline_percent": gains[
                    "boundary_proposal"],
                "mAP_percent": [
                    float(value * 100) for value in boundary_curve],
                "fixed_original_scores_and_labels": True,
                "existing_boundaries_only": True,
                "replacement_count": int(v13_replacement_count),
            },
            "joint_oracle": {
                "average_mAP_percent": float(joint_mAP * 100),
                "gain_over_baseline_percent": gains["joint"],
                "mAP_percent": [
                    float(value * 100) for value in joint_curve],
            },
            "proposal_recall": {
                "pre_nms": pre_nms_recall,
                "post_nms": post_nms_recall,
                "candidate_count": int(v13_candidate_count),
            },
            "decision": {
                "recommended_first_route": recommended_route,
                "score_boundary_clear_margin_percent": route_margin,
                "preferred_joint_oracle_gain_percent": 3.0,
                "joint_has_preferred_space": gains["joint"] >= 3.0,
                "activitynet_boundary_gate_percent": 3.0,
                "activitynet_joint_coupling_gate_percent": 5.0,
                "activitynet_stop_below_joint_gain_percent": 3.0,
                "target_absolute_gain_percent": 2.0,
                "target_final_mAP_percent": float(mAP * 100 + 2.0),
                "note": (
                    "Oracle gain is an upper bound, not an expected model "
                    "gain. Train no V13 module until this report is reviewed."
                ),
            },
            "tiou_thresholds": [
                float(value) for value in evaluator.tiou_thresholds],
        }
        with open(v13_report_file, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    if refinement_stats:
        keys = sorted({key for item in refinement_stats for key in item})
        print("Refinement diagnostics:")
        derived_keys = {
            'v92_accepted_agreement_sum',
            'v92_accepted_uncertainty_sum',
        }
        total_v92_accepts = sum(
            float(item.get('v9_accepts', 0.0))
            for item in refinement_stats
            if 'v92_accepted_agreement_sum' in item)
        if total_v92_accepts > 0.0:
            agreement_sum = sum(
                float(item.get('v92_accepted_agreement_sum', 0.0))
                for item in refinement_stats)
            uncertainty_sum = sum(
                float(item.get('v92_accepted_uncertainty_sum', 0.0))
                for item in refinement_stats)
            print("  v92_mean_accepted_agreement: {:.6f}".format(
                agreement_sum / total_v92_accepts))
            print("  v92_mean_accepted_uncertainty: {:.6f}".format(
                uncertainty_sum / total_v92_accepts))
            derived_keys.update({
                'v92_mean_accepted_agreement',
                'v92_mean_accepted_uncertainty',
            })
        for key in keys:
            if key in derived_keys:
                continue
            values = [item[key] for item in refinement_stats if key in item]
            print("  {}: {:.6f}".format(key, float(np.mean(values))))
        print("  inference_latency_s_per_video: {:.6f}".format(
            (time.time() - eval_start) / max(len(val_loader), 1)))
        if torch.cuda.is_available():
            print("  gpu_peak_memory_mb: {:.2f}".format(
                torch.cuda.max_memory_allocated() / (1024 ** 2)))

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)

    return mAP
