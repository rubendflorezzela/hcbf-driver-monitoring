# HCBF Driver Monitoring

Reproducibility resources for:

**Beyond aggregate scores: Deployment-aware and non-compensatory benchmarking of vision-based eye-state recognition models for driver monitoring**

## Evaluation scope

- Clean predictive performance and calibration
- Deterministic corruption robustness
- Cross-domain zero-shot transfer
- Participant-safe target-domain training with out-of-fold evaluation
- Embedded TensorRT FP32 deployment evidence
- Black-box RISE faithfulness and stochastic stability
- Non-compensatory operational eligibility
- Retrospective four-model HCS preference-sensitivity audit

## Models

- MobileNetV3-Large
- ShuffleNetV2 x1.0
- EfficientNet-B0
- DeiT-Tiny
- RepViT-M1.0
- EfficientFormer-L1

## Protocol notes

MRL Eye uses `closed`/`open` labels, whereas RT-BENE uses `blink`/`open`. The shared binary coding is therefore open versus non-open; the source and target non-open labels are related but not identical.

The matched RT-BENE analysis uses participant-safe out-of-fold predictions on 11 participants (77,119 frames), with three fixed validation participants and mutually exclusive outer test folds of 4/4/3 participants. Target-domain models start from ImageNet-1K rather than MRL checkpoints.

The primary operational screen is a benchmarking criterion, not automotive safety certification. It combines numerical parity, a 33.333-ms P95 binocular-pair latency deadline, classwise recall of at least 0.50 under every non-clean RT-BENE OOF condition, and no class collapse.

## Installation

```bash
conda env create -f environment.yml
conda activate hcbf
python -m pip install -e .
```

## Verification

Run the test suite:

```bash
pytest
```

Verify the frozen results and checksums:

```bash
python scripts/reproduce.py verify
```

Regenerate the numerical figures:

```bash
python scripts/reproduce.py figures
```

Run both:

```bash
python scripts/reproduce.py all
```

Generated figures are written to `results/figures/`.

## Qualitative RISE figures

The qualitative figures require the original MRL Eye images and the frozen RISE saliency shards.

Main-manuscript qualitative figure:

```bash
python scripts/rise_figures.py main --root /path/to/local/HCBF-project
```

Supplementary galleries:

```bash
python scripts/rise_figures.py galleries --root /path/to/local/HCBF-project
```

The frozen XAI corpus contains no eligible open-eye sample for participant `s0008`; this is reflected in Supplementary Fig. S.8.

## Repository structure

```text
configs/                Frozen evaluation protocol
manifests/              Subject, checkpoint, XAI, and RISE manifests
results/frozen/         Machine-readable manuscript and supplement results
results/figures/        Regenerated numerical publication figures
src/hcbf/               Analysis and plotting modules
scripts/reproduce.py    Verification and numerical-figure entry point
scripts/rise_figures.py Qualitative RISE figure entry point
tests/                  Regression and methodological tests
docs/                   Reproducibility instructions
```

## Data

MRL Eye and RT-BENE are available from their original providers and are not redistributed in this repository. The repository contains processed tables, frozen protocols, manifests, and derived analysis resources used to support the manuscript.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. Dataset licenses remain with the original data providers.
