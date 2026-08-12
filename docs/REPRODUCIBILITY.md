# Reproducibility

## Environment

```bash
conda env create -f environment.yml
conda activate hcbf
python -m pip install -e .
```

## Frozen protocol

`configs/protocol.yaml` records the six-model portfolio, MRL subject-disjoint split, RT-BENE participant-safe target-domain protocol, deterministic corruption catalogue, numerical-parity tolerances, operational screening criteria, RISE configuration, and retrospective HCS scope.

The RT-BENE matched cohort contains 11 participants and 77,119 frames. Three additional participants are reserved for validation; the outer test folds contain 4/4/3 participants with no participant overlap among training, validation, and test roles within a fold.

## Tests

```bash
pytest
```

## Numerical verification

```bash
python scripts/reproduce.py verify
```

## Numerical figures

```bash
python scripts/reproduce.py figures
```

## Complete numerical workflow

```bash
python scripts/reproduce.py all
```

## Qualitative RISE figures

Required local resources:

- MRL Eye images
- `rise_full_curve_metrics.csv`
- `rise_full_saliency_shard_registry.csv`
- frozen RISE saliency shards

Main figure:

```bash
python scripts/rise_figures.py main --root /path/to/local/HCBF-project
```

Supplementary galleries:

```bash
python scripts/rise_figures.py galleries --root /path/to/local/HCBF-project
```

## Numerical parity and timing

All 12 TensorRT FP32 engines must satisfy the frozen parity thresholds before latency measurement: P99 absolute probability difference <= 1e-4, maximum absolute probability difference <= 5e-4, zero class disagreements, and maximum repeated-run absolute logit difference <= 1e-7 on 16 frozen validation inputs per engine.

## Frozen artifacts

`results/frozen/` contains the machine-readable values used for the manuscript and supplementary material.

`manifests/frozen_files.sha256` contains SHA-256 checksums for the frozen protocol, manifests, and result files.
