# python imports
import argparse
import itertools
import json
import os
import glob
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed


def _parse_csv_values(raw, cast, name):
    try:
        values = [cast(value.strip()) for value in raw.split(',')
                  if value.strip()]
    except ValueError as error:
        raise ValueError("{} contains an invalid value".format(name)) from error
    if not values:
        raise ValueError("{} must not be empty".format(name))
    return values


def build_oracle_candidate_profiles(args, num_train_timesteps):
    requested = any((args.oracle_seeds, args.oracle_steps,
                     args.oracle_alphas))
    if not requested:
        return []
    if not args.oracle_diagnostic:
        raise ValueError(
            "V8 oracle profile options require --oracle-diagnostic")
    if not all((args.oracle_seeds, args.oracle_steps, args.oracle_alphas)):
        raise ValueError(
            "V8 oracle profiles require seeds, steps, and alphas together")
    seeds = _parse_csv_values(args.oracle_seeds, int, "--oracle-seeds")
    steps = _parse_csv_values(args.oracle_steps, int, "--oracle-steps")
    alphas = _parse_csv_values(args.oracle_alphas, float, "--oracle-alphas")
    if any(value < 0 for value in seeds):
        raise ValueError("--oracle-seeds values must be >= 0")
    if any(not 1 <= value <= num_train_timesteps for value in steps):
        raise ValueError(
            "--oracle-steps values must be within the training schedule")
    if any(not 0.0 < value <= 1.0 for value in alphas):
        raise ValueError("--oracle-alphas values must be within (0, 1]")
    if args.oracle_max_relative_shift is not None \
            and not 0.0 < args.oracle_max_relative_shift <= 1.0:
        raise ValueError(
            "--oracle-max-relative-shift must be within (0, 1]")
    profiles = [{
        'seed': seed,
        'steps': step,
        'alpha': alpha,
        'max_relative_shift': args.oracle_max_relative_shift,
    } for seed, step, alpha in itertools.product(seeds, steps, alphas)]
    if len(profiles) > args.oracle_max_profiles:
        raise ValueError(
            "V8 oracle grid contains {} profiles, exceeding the safety cap {}"
            .format(len(profiles), args.oracle_max_profiles))
    return profiles


################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if (cfg['dataset_name'] == 'anet' and not args.saveonly
            and not args.p2d1_calibration_only):
        external_score_file = cfg['test_cfg'].get('ext_score_file')
        if not external_score_file or not os.path.isfile(external_score_file):
            raise FileNotFoundError(
                "ActivityNet final multi-class evaluation requires an existing "
                "external score JSON: {}".format(
                    external_score_file or '<empty>'))
    if args.oracle_diagnostic and cfg['dataset_name'] != 'thumos':
        raise ValueError(
            "Oracle diagnostic currently supports THUMOS only")
    if (args.v13_phase_a
            and cfg['dataset_name'] not in ('thumos', 'anet')):
        raise ValueError(
            "V13 Phase A currently supports THUMOS and ActivityNet only")
    if args.v13_phase_a and args.oracle_diagnostic:
        raise ValueError(
            "V13 Phase A and diffusion oracle modes are mutually exclusive")
    if args.v13_phase_a and not (
            args.v13_report or args.p2d0_report or args.p3_score_report
            or args.p4_joint_report):
        raise ValueError(
            "V13 candidate export requires --v13-report, --p2d0-report, "
            "--p3-score-report, or --p4-joint-report")
    if (args.v13_report or args.p2d0_report or args.p3_score_report
            or args.p4_joint_report) and not args.v13_phase_a:
        raise ValueError(
            "V13/P2-D0/P3-0/P4-0 reports require --v13-phase-a")
    if args.p2d0_report and cfg['dataset_name'] != 'anet':
        raise ValueError("P2-D0 diagnostic supports ActivityNet only")
    if args.p2d0_report and args.saveonly:
        raise ValueError("P2-D0 diagnostic requires evaluation")
    if args.p3_score_report and cfg['dataset_name'] != 'anet':
        raise ValueError("P3-0 score diagnostic supports ActivityNet only")
    if args.p3_score_report and args.saveonly:
        raise ValueError("P3-0 score diagnostic requires evaluation")
    if args.p4_joint_report and cfg['dataset_name'] != 'anet':
        raise ValueError("P4-0 joint diagnostic supports ActivityNet only")
    if args.p4_joint_report and args.saveonly:
        raise ValueError("P4-0 joint diagnostic requires evaluation")
    if args.p2d1_report:
        if cfg['dataset_name'] != 'anet':
            raise ValueError("P2-D1 report supports ActivityNet only")
        if not (cfg.get('model', {}).get(
                'cluster_selector', {}).get('enabled', False)):
            raise ValueError(
                "P2-D1 report requires an enabled cluster selector")
        if args.saveonly:
            raise ValueError("P2-D1 report requires evaluation")
    if args.p2d1_calibration_only:
        if not args.p2d1_report:
            raise ValueError(
                "P2-D1 calibration-only mode requires --p2d1-report")
        if list(cfg.get('val_split', [])) != ['p2d1_calibration']:
            raise ValueError(
                "P2-D1 calibration-only mode requires the held-out split")
    p2d0_cluster_thresholds = _parse_csv_values(
        args.p2d0_cluster_thresholds, float,
        "--p2d0-cluster-thresholds")
    p2d0_replacement_budgets = _parse_csv_values(
        args.p2d0_replacement_budgets, float,
        "--p2d0-replacement-budgets")
    p3_score_logit_caps = _parse_csv_values(
        args.p3_score_logit_caps, float,
        "--p3-score-logit-caps")
    p3_score_budgets = _parse_csv_values(
        args.p3_score_budgets, float,
        "--p3-score-budgets")
    p4_boundary_budgets = _parse_csv_values(
        args.p4_boundary_budgets, float,
        "--p4-boundary-budgets")
    p4_score_budgets = _parse_csv_values(
        args.p4_score_budgets, float,
        "--p4-score-budgets")
    if any(not 0.0 < value <= 1.0
           for value in p2d0_cluster_thresholds):
        raise ValueError(
            "--p2d0-cluster-thresholds values must be within (0, 1]")
    if any(not 0.0 <= value <= 1.0
           for value in p2d0_replacement_budgets):
        raise ValueError(
            "--p2d0-replacement-budgets values must be within [0, 1]")
    if any(value <= 0.0 for value in p3_score_logit_caps):
        raise ValueError("--p3-score-logit-caps values must be positive")
    if any(not 0.0 < value <= 1.0 for value in p3_score_budgets):
        raise ValueError(
            "--p3-score-budgets values must be within (0, 1]")
    if not 0.0 < args.p4_cluster_threshold <= 1.0:
        raise ValueError("--p4-cluster-threshold must be within (0, 1]")
    for name, values in (
            ("--p4-boundary-budgets", p4_boundary_budgets),
            ("--p4-score-budgets", p4_score_budgets)):
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(name + " values must be within [0, 1]")
        if 0.0 not in values:
            raise ValueError(name + " must include identity 0")
    if args.p4_score_logit_cap <= 0.0:
        raise ValueError("--p4-score-logit-cap must be positive")
    if args.oracle_diagnostic and args.saveonly:
        raise ValueError(
            "Oracle diagnostic requires evaluation, not --saveonly")
    if args.oracle_candidates <= 0:
        raise ValueError("--oracle-candidates must be > 0")
    if not args.oracle_diagnostic and args.oracle_candidates != 1:
        raise ValueError(
            "--oracle-candidates is only valid with --oracle-diagnostic")
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
        ckpt_file = ckpt_file_list[-1]

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    # pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    oracle_profiles = build_oracle_candidate_profiles(
        args, model.diffusion_refiner.scheduler.num_train_timesteps
        if getattr(model, 'diffusion_enabled', False) else 0)
    if args.oracle_diagnostic:
        if not getattr(model, 'diffusion_enabled', False):
            raise ValueError(
                "Oracle diagnostic requires a diffusion-enabled model")
        model.oracle_diagnostic_enabled = True
        model.oracle_num_candidates = (
            len(oracle_profiles) if oracle_profiles else args.oracle_candidates)
        model.oracle_candidate_profiles = oracle_profiles
        print(
            "WARNING: ORACLE DIAGNOSTIC ENABLED. Ground truth selects "
            "boundaries; oracle metrics are not model results. "
            "candidates_per_proposal={}".format(model.oracle_num_candidates))
        if oracle_profiles:
            print("V8 oracle profile grid: seeds={} steps={} alphas={} "
                  "max_relative_shift={}".format(
                      args.oracle_seeds, args.oracle_steps,
                      args.oracle_alphas, args.oracle_max_relative_shift))
    if args.v13_phase_a:
        if getattr(model, 'diffusion_enabled', False):
            raise ValueError(
                "V13 Phase A must use a diffusion-disabled coarse detector")
        model.v13_phase_a_enabled = True
        print(
            "WARNING: V13 PHASE A DIAGNOSTIC ENABLED. Ground truth is used "
            "for oracle ranking/selection; oracle metrics are not model "
            "results and no model parameters are changed.")
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location='cuda:{}'.format(
            int(str(cfg['devices'][0]).split(':')[-1])),
        weights_only=False,
    )
    # load ema model instead
    print("Loading from EMA model ...")
    incompatible = model.load_state_dict(checkpoint['state_dict_ema'], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("Checkpoint compatibility: missing={}, unexpected={}".format(
            incompatible.missing_keys, incompatible.unexpected_keys))
    refiner = getattr(model.module, "diffusion_refiner", None)
    ensemble_members = checkpoint.get("candidate_ranker_ensemble")
    if ensemble_members is not None:
        if refiner is None:
            raise RuntimeError(
                "V9.2 checkpoint requires a diffusion refiner")
        refiner.install_candidate_ranker_ensemble(
            ensemble_members, checkpoint["v92_calibration"])
        ensemble_version = (
            "V9.3" if checkpoint["v92_calibration"].get(
                "low_tiou_safety_enabled", False) else "V9.2")
        print("Installed {} candidate-ranker ensemble: {} members".format(
            ensemble_version, len(ensemble_members)))
    elif (refiner is not None
          and refiner.candidate_ranker_ensemble_enabled):
        raise RuntimeError(
            "V9.2 config requires checkpoint ensemble metadata")
    del checkpoint

    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly and not args.p2d1_calibration_only:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds']
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')
    if args.output_file:
        output_file = args.output_file
    elif output_file is None:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=(
            None if args.p2d1_calibration_only
            else cfg['test_cfg']['ext_score_file']),
        tb_writer=None,
        print_freq=args.print_freq,
        oracle_report_file=args.oracle_report or None,
        v13_report_file=args.v13_report or None,
        p2d0_report_file=args.p2d0_report or None,
        p2d0_cluster_thresholds=p2d0_cluster_thresholds,
        p2d0_replacement_budgets=p2d0_replacement_budgets,
        p3_score_report_file=args.p3_score_report or None,
        p3_score_logit_caps=p3_score_logit_caps,
        p3_score_budgets=p3_score_budgets,
        p4_joint_report_file=args.p4_joint_report or None,
        p4_cluster_threshold=args.p4_cluster_threshold,
        p4_boundary_budgets=p4_boundary_budgets,
        p4_score_budgets=p4_score_budgets,
        p4_score_logit_cap=args.p4_score_logit_cap,
        p2d1_report_file=args.p2d1_report or None,
        p2d1_calibration_only=args.p2d1_calibration_only,
        diagnostic_dataset_name=cfg['dataset_name'],
    )
    if args.oracle_report and oracle_profiles:
        with open(args.oracle_report, encoding='utf-8') as stream:
            report = json.load(stream)
        report.update({
            'v8_oracle_gate': True,
            'oracle_candidate_profile_count': len(oracle_profiles),
            'oracle_candidate_profile_grid': {
                'seeds': _parse_csv_values(
                    args.oracle_seeds, int, '--oracle-seeds'),
                'steps': _parse_csv_values(
                    args.oracle_steps, int, '--oracle-steps'),
                'alphas': _parse_csv_values(
                    args.oracle_alphas, float, '--oracle-alphas'),
                'max_relative_shift': args.oracle_max_relative_shift,
            },
        })
        with open(args.oracle_report, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('--oracle-diagnostic', action='store_true',
                        help='GT-only upper-bound analysis; never a model result')
    parser.add_argument('--oracle-report', default='', type=str,
                        help='write structured oracle diagnostic JSON')
    parser.add_argument('--oracle-candidates', default=1, type=int,
                        help='different DDIM noise seeds per proposal in oracle mode')
    parser.add_argument('--oracle-seeds', default='', type=str,
                        help='V8 oracle seed grid, comma separated')
    parser.add_argument('--oracle-steps', default='', type=str,
                        help='V8 oracle DDIM-step grid, comma separated')
    parser.add_argument('--oracle-alphas', default='', type=str,
                        help='V8 oracle residual-alpha grid, comma separated')
    parser.add_argument('--oracle-max-relative-shift', default=None, type=float,
                        help='optional shared safety cap for V8 oracle profiles')
    parser.add_argument('--oracle-max-profiles', default=128, type=int,
                        help='safety cap for the V8 Cartesian profile grid')
    parser.add_argument('--v13-phase-a', action='store_true',
                        help='GT-only score/boundary/joint oracle decomposition')
    parser.add_argument('--v13-report', default='', type=str,
                        help='write the structured V13 Phase A report')
    parser.add_argument('--p2d0-report', default='', type=str,
                        help='write ActivityNet P2-D0 constrained oracle JSON')
    parser.add_argument(
        '--p2d0-cluster-thresholds', default='0.7,0.8,0.9', type=str,
        help='P2-D0 proposal-cluster tIoUs, comma separated')
    parser.add_argument(
        '--p2d0-replacement-budgets',
        default='0.01,0.02,0.05,0.10', type=str,
        help='P2-D0 global replacement budget fractions, comma separated')
    parser.add_argument('--p3-score-report', default='', type=str,
                        help='write ActivityNet P3-0 bounded score oracle JSON')
    parser.add_argument(
        '--p3-score-logit-caps', default='0.25,0.5,1.0', type=str,
        help='P3-0 absolute score-logit adjustment caps, comma separated')
    parser.add_argument(
        '--p3-score-budgets', default='0.01,0.02,0.05', type=str,
        help='P3-0 global score-adjustment fractions, comma separated')
    parser.add_argument('--p4-joint-report', default='', type=str,
                        help='write ActivityNet P4-0 constrained joint JSON')
    parser.add_argument('--p4-cluster-threshold', default=0.9, type=float,
                        help='P4-0 local proposal-cluster tIoU')
    parser.add_argument(
        '--p4-boundary-budgets', default='0,0.01,0.02,0.05', type=str,
        help='P4-0 existing-boundary replacement fractions')
    parser.add_argument(
        '--p4-score-budgets', default='0,0.01,0.02,0.05', type=str,
        help='P4-0 bounded score-adjustment fractions')
    parser.add_argument('--p4-score-logit-cap', default=1.0, type=float,
                        help='P4-0 absolute score-logit adjustment cap')
    parser.add_argument('--p2d1-report', default='', type=str,
                        help='write ActivityNet P2-D1 selector report JSON')
    parser.add_argument(
        '--p2d1-calibration-only', action='store_true',
        help='report selector statistics on held-out training data without '
             'external classification fusion or official mAP')
    parser.add_argument('--output-file', default='', type=str,
                        help='override prediction pickle output location')
    args = parser.parse_args()
    main(args)
