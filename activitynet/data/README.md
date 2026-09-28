# ActivityNet inputs

This directory contains instructions only, not videos, features, annotations,
classification scores or checkpoints.

Starting from `activitynet/`, the archived YAML expects:

```text
data/activitynet/
  annotations/anet1.3_tsp_filtered.json
  tsp_features/*.npy
  classifiers/new_3ensemble_uniformerv2_large_only_global_anet_16x10x3.json
```

Alternatively, set `dataset.json_file`, `dataset.feat_folder` and
`test_cfg.ext_score_file` to your input locations. The TSP features have 512
channels. Training validation and evaluation require the external 200-class
video scores even though the detector predicts class-agnostic action scores.

Refer to the [ActivityNet project](http://activity-net.org/download.html) and
the [DyFADet data instructions](https://github.com/yangle15/DyFADet-pytorch#data-preparation)
for the original resources. Obtain files under their respective terms.

The final 38.80 run configuration and input mapping still need verification;
these paths describe only the supplied archived candidate.
