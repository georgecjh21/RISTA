# RISTA

Task-Asymmetric Scale Aggregation with Intra-Scale Mamba for Temporal Action Localization

Jiahao Chen and Qiuhong Tian, Zhejiang Sci-Tech University

**Initial source-code package. See the artifact status and validation limits below.**

This compact candidate follows the file categories of
[DyFADet](https://github.com/yangle15/DyFADet-pytorch). ActivityNet and FineAction
retain separate implementations because their entry points and supporting code
are not interchangeable. Do not combine their `libs/` directories or load both
as `libs` in the same Python process.

## Layout

```text
activitynet/
  configs/rista_activitynet_archived.yaml
  data/README.md
  libs/
  train.py
  eval.py
  requirements.txt
fineaction/
  configs/rista_fineaction.yaml
  data/README.md
  libs/
  train.py
  eval.py
  requirements.txt
```

Only main-model training/evaluation sources, their supporting libraries, two
configurations and brief setup notes are included. Standalone MAC-counting
scripts are omitted. Existing imported legacy modules remain to preserve
compatibility; their presence does not mean they are enabled by the supplied
configurations. No datasets, features, scores, weights, logs, ablation recipes,
paper files, figures or compiled extensions are bundled.

## Environment

The authors confirmed Python 3.9.25, PyTorch 2.7.1+cu128, CUDA 12.8 and an NVIDIA
GeForce RTX 5090 D v2 for both datasets. The historical requirements files are
not a complete environment lockfile. ActivityNet's list is copied from the
archive's `requirements-5090.txt`; FineAction retains its historical list,
including `setuptools==58.0.4`. A clean installation has not been validated.

The DBM implementation requires `mamba_ssm.modules.mamba_new.Mamba` from
[Video Mamba Suite](https://github.com/OpenGVLab/video-mamba-suite) at commit
`ec9108b72d5db59f6d634c94cd0e008228a7b918`, including its `mamba/` and
`causal-conv1d/` packages. Generic PyPI `mamba-ssm` is not a drop-in replacement.
Build the CUDA extensions for the target GPU; RTX 5090 requires appropriate
`sm_120` support. NMS source is provided in each implementation's
`libs/utils/` and must also be built for the selected environment.

## Training and evaluation

Prepare inputs at the paths described in each `data/README.md`, or edit
`dataset.json_file`, `dataset.feat_folder` and `test_cfg.ext_score_file` in the
corresponding configuration. Relative paths are resolved from the dataset
implementation's working directory.

For FineAction, starting from the repository root:

```bash
cd fineaction
CUDA_VISIBLE_DEVICES=0 python train.py configs/rista_fineaction.yaml --output rista_seed1234567891
CUDA_VISIBLE_DEVICES=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python eval.py configs/rista_fineaction.yaml /absolute/path/to/trusted_checkpoint.pth.tar
```

Training starts from scratch on fixed pre-extracted features. The historical
name FT-R875 does not indicate detector checkpoint fine-tuning. The command
above saves checkpoints to `fineaction/outputs/rista_fineaction_rista_seed1234567891/`
relative to the repository root.

For the archived ActivityNet candidate, start again from the repository root:

```bash
cd activitynet
CUDA_VISIBLE_DEVICES=0 python train.py configs/rista_activitynet_archived.yaml --output-dir outputs/rista_archived_seed0
CUDA_VISIBLE_DEVICES=0 python eval.py configs/rista_activitynet_archived.yaml /absolute/path/to/trusted_checkpoint.pth.tar
```

Both legacy evaluation paths allow pickled objects in checkpoints. **Only load
checkpoints you created or otherwise trust**; deserialization can execute code.

## Artifact status and attribution

The paper reports 38.80% and 24.23% average mAP on ActivityNet-1.3 and FineAction,
respectively. FineAction source and configuration are matched to the archived
24.23 run, but its checkpoint is not bundled or re-evaluated here. The supplied
ActivityNet configuration is an older archive, not a verified reproduction
recipe for the final 38.80 result. Matching the final ActivityNet artifacts
remains outstanding.

This implementation builds on DyFADet and uses the external Video Mamba Suite
DBM dependency. Source notices are retained. Upstream redistribution terms and
the authors' license choice remain to be confirmed; this draft does not apply
a blanket license to third-party code. No clean Linux installation, CUDA
forward/backward or benchmark evaluation has been performed for this candidate.
