# FineAction inputs

This directory contains instructions only, not videos, features, annotations,
classification scores or checkpoints.

Starting from `fineaction/`, the YAML expects:

```text
data/fineaction/
  annotations/annotations_gt_af.json
  fineaction_mae_g/*.npy
  classifiers/new_swinB_1x1x256_views2x3_max_label_avg_prob.json
```

Alternatively, set `dataset.json_file`, `dataset.feat_folder` and
`test_cfg.ext_score_file` to your input locations. Features have 1408 channels.
The processed annotation file must match the available features and preserve
the 106 zero-based action-category IDs; do not replace all labels with zero
because the detector head is class-agnostic. This compact candidate does not
bundle the annotation-preparation helper or the prepared input files.

Sources:

- [FineAction](https://github.com/Richard-61/FineAction)
- [VideoMAEv2 temporal-localization resources](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/TAD.md)
- [VideoMAEv2 feature repository](https://huggingface.co/datasets/OpenGVLab/VideoMAEv2-TAL-Features)
- [DyFADet data instructions and external scores](https://github.com/yangle15/DyFADet-pytorch#data-preparation)

Obtain inputs under their respective terms. The external classification scores
are required to assign action categories to class-agnostic detector proposals.
