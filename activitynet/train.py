# python imports
import argparse
import os
import time
import datetime
import json
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)


################################################################################
def get_training_validation_settings(cfg):
    """Return official validation settings and enforce ActivityNet scores.

    ActivityNet trains a class-agnostic (``num_classes == 1``) proposal model.
    Its official 200-class mAP therefore needs external video classification
    scores. Refuse to start without that file so every baseline validation and
    best-checkpoint decision has the same meaning as the original DyFADet run.
    """
    ext_score_file = cfg.get('test_cfg', {}).get('ext_score_file')
    if cfg.get('dataset_name') != 'anet':
        return True, ext_score_file
    if ext_score_file and os.path.isfile(ext_score_file):
        return True, ext_score_file
    raise FileNotFoundError(
        "ActivityNet training requires an existing external 200-class score "
        "JSON for official validation: {}".format(
            ext_score_file or '<empty>'))


def append_fusion_diagnostics(metrics_file, epoch, model, source):
    core = model.module if hasattr(model, "module") else model
    if not hasattr(core, "fusion_diagnostic_report"):
        return
    report = core.fusion_diagnostic_report()
    if not report.get("mamba_task_gates") \
            and not report.get("selective_modules"):
        return
    payload = {
        "phase": "fusion_diagnostics",
        "epoch": epoch,
        "source": source,
        "timestamp_unix": time.time(),
        "report": report,
    }
    with open(metrics_file, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    # pprint(cfg)


    # prep for output folder (based on time stamp or exact experiment directory)
    if not os.path.exists(cfg['output_folder']):
        os.makedirs(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if args.output_dir:
        ckpt_folder = os.path.abspath(args.output_dir)
        if os.path.exists(ckpt_folder) and not args.resume:
            existing_training_files = [
                os.path.join(ckpt_folder, "config.txt"),
                os.path.join(ckpt_folder, "last.pth.tar"),
            ]
            if any(os.path.exists(path) for path in existing_training_files):
                raise FileExistsError(
                    "Refusing to overwrite existing experiment: {}".format(ckpt_folder)
                )
    elif len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    metrics_file = os.path.join(ckpt_folder, "metrics.jsonl")

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )

    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])


    # create val loader
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )
    _, validation_ext_score_file = \
        get_training_validation_settings(cfg)

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    if args.baseline_ckpt:
        checkpoint = torch.load(
            args.baseline_ckpt, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict_ema', checkpoint.get('state_dict', checkpoint))
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_new_modules = (
            'diffusion_refiner',
            'start_end_distribution_head',
            'trident_head',
            'regression_sgp_adapter',
            'dense_iou_quality_head',
            'residual_alignment_head',
            'boundary_proposal_generator',
            'cluster_selector_head',
            'cluster_score_suppression_head',
            'mamba_stem',
            'mamba_branch',
            'mamba_neck_adapter',
        )
        unexpected = [
            key for key in incompatible.unexpected_keys
            if not any(module in key for module in allowed_new_modules)
        ]
        missing = [
            key for key in incompatible.missing_keys
            if not any(module in key for module in allowed_new_modules)
        ]
        if unexpected or missing:
            raise RuntimeError("Baseline checkpoint mismatch: missing={}, unexpected={}".format(
                missing, unexpected))
        print(
            "Loaded baseline weights; newly initialized optional-module "
            "parameters: {}".format(len(incompatible.missing_keys)))
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            device_index = int(str(cfg['devices'][0]).split(':')[-1])
            checkpoint = torch.load(
                args.resume,
                map_location='cuda:{}'.format(device_index),
                weights_only=False,
            )
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    best_ep = 0
    best_mAP = 0

    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_metrics = train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            print_freq=args.print_freq,
            metrics_file=metrics_file,
        )
        append_fusion_diagnostics(
            metrics_file, epoch, model, "online_after_train")

        # Always keep a resumable last checkpoint. This does not change optimization.
        last_state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'state_dict_ema': model_ema.module.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_mAP': best_mAP,
            'best_epoch': best_ep,
        }
        save_checkpoint(
            last_state, False, file_folder=ckpt_folder,
            file_name='last.pth.tar'
        )

        if ((epoch >= (max_epochs - 7)) | (epoch%2 == 1) ) & (epoch > max_epochs//2) :
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }


            val_db_vars = val_dataset.get_attributes()
            det_eval = ANETdetection(
                val_dataset.json_file,
                val_dataset.split[0],
                tiou_thresholds = val_db_vars['tiou_thresholds'])
            
            mAP = valid_one_epoch(
                val_loader,
                model_ema.module,
                -1,
                evaluator=det_eval,
                output_file=None,
                ext_score_file=validation_ext_score_file,
                tb_writer=None,
                print_freq=args.print_freq)
            append_fusion_diagnostics(
                metrics_file, epoch, model_ema.module,
                "ema_after_validation")
            with open(metrics_file, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "phase": "validation",
                    "epoch": epoch,
                    "validation_mAP": mAP,
                    "best_mAP_before_update": best_mAP,
                    "best_epoch_before_update": best_ep,
                    "timestamp_unix": time.time(),
                    "gpu_peak_memory_mb": (
                        torch.cuda.max_memory_allocated() / (1024 ** 2)
                        if torch.cuda.is_available() else 0.0
                    ),
                    "train_epoch_metrics": train_metrics,
                }, sort_keys=True) + "\n")
            
            if (best_mAP-mAP) > 0.0015:
                break

            if mAP > best_mAP:
                best_mAP = mAP
                file_name=ckpt_folder+'/epoch_{:03d}bestmodel.pth.tar'.format(best_ep)
                if os.path.exists(file_name):
                    os.remove(file_name)

                best_ep = epoch + 1
                print('saving best model')
                # save ckpt once in a while
                save_states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'mAP': mAP
                }
                
                file_name='epoch_{:03d}bestmodel.pth.tar'.format(best_ep)
                save_states['state_dict_ema'] = model_ema.module.state_dict()
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name=file_name
                )

    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR', default= './configs/thumos_i3d.yaml',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default= 30, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--output-dir', default='', type=str,
                        help='exact experiment directory used by reproducible wrappers')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--baseline-ckpt', default='', type=str, metavar='PATH',
                        help='initialize baseline weights and leave the new refiner initialized')
    
    args = parser.parse_args()
    main(args)
