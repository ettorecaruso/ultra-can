# Ultra-CAN: a lightweight blind edge-AI receiver for ISAC in the Internet of Drones

Reference implementation and full experimental pipeline of the paper

> **Ultra-CAN: A Lightweight Blind Edge AI Receiver for Integrated Sensing and Chaos-Based Communications in Internet of Drones**

A single chaos-based waveform is decoded and ranged by one ultra-light neural receiver. The
node transmits a burst, regenerates the chaotic reference locally from the seed it drew for
that burst (monostatic ISAC, no state exchanged over the air) and reads the delay of the
dominant echo from the correlation profile, so telemetry and ranging come from the same burst
and the same forward pass.

This repository contains the channel/scene simulator, the receivers, the training and
evaluation code, the scripts that produce every figure and table of the paper, and the exact
result tree used for the manuscript.

## Layout

```
configs/                 YAML configuration (base model/data, channel variants, experiment profiles)
src/
  data/                  chaotic maps, channel models (geometry, 3GPP TDL, two-ray), dataset generation
  models/                Ultra-CAN (Conv1D attention pool) and Ultra-CAN-QKV, baselines, classical receivers
  training/              Trainer, losses (communication + ranging)
  evaluation/            metrics, ranging utilities, statistics
  experiments/           runner.py (8 experiments) + standalone experiments
  visualization/         attention and layer-activity tools
scripts/
  data/                  dataset generation shell entry points
  figures/               one script per paper figure
  tables/                one script per paper table
  diagnostics/           model-free controls (label observability, scene report, LPI)
  tools/                 latency bench, jamming-aware runner, reproduction probes
  style/                 shared Matplotlib style
notebooks/               Colab/local orchestration notebooks (see below)
results/full/            the result tree of the paper (CSV, checkpoints, provenance)
figures/                 the PDFs referenced by the paper
```

## Requirements

Python 3.10+ with TensorFlow 2.13+.

```bash
python -m pip install -r requirements.txt
```

## Reproducing the paper

Everything is driven from the repository root. The two supported entry points are the
notebooks (recommended, works on Colab) and the shell/Python scripts (any machine).

### 1. Datasets

```bash
bash scripts/data/generate_all_datasets.sh
```

Datasets are content-addressed under `data/raw/<hash>/`; reruns reuse them and the hash
depends on the channel/scene configuration, so it changes only when the channel changes.

### 2. Experiments

```bash
python src/experiments/runner.py --experiments all --mode full
```

`runner.py` accepts `--experiments` (comma separated, or `all`), `--mode {fast,full}`,
`--model`, `--scenario`, `--channel`, `--no-regen`. The registered experiments are:

| Experiment | What it produces |
|---|---|
| `ber_vs_snr` | the four received-BER scenarios (single echo, multi-echo, reduced Doppler, peers) and their checkpoints |
| `classical_receivers` | DCSK correlator, matched filter, energy detector on the noise-free echo-only link |
| `jamming` | CW / barrage / partial-band sweeps over the JSR grid |
| `jamming_interpretability` | per-layer cosine retention and the decision margin |
| `frequency_agility` | hopping versus dwell, JSR and jammer reaction |
| `channel_generalization` | the TDL/two-ray cross-channel benchmark, with and without in-domain retraining |
| `peer_estimation` | peer-aware ranging outcomes and the fail-safe trade-off |
| `final_report` | weight table and aggregated report |

Additional experiments have their own entry points:

| Experiment | Command |
|---|---|
| Unified 7-way benchmark | `python src/experiments/unified_benchmark.py --scenario k3_doppler_full` |
| Blind vs reference-aided (price of blindness) | `python src/experiments/blind_vs_oracle.py --scenario k3_doppler_full` |
| Ranging benchmark with per-burst dumps | `python src/experiments/ranging_benchmark.py --scenario k3_doppler_full` |
| Ambiguous pile (negative control) | `python src/experiments/ambiguity.py --scenario k3_ambiguity_mix` |
| Velocity from a track of bursts | `python src/experiments/velocity_track.py` |
| On-board footprint and latency | `python src/experiments/board_footprint.py` |

### 3. Figures and tables

```bash
bash scripts/make_all.sh
```

Each script in `scripts/figures/` writes one PDF into `figures/`; each script in
`scripts/tables/` writes the CSVs into `results/full/tables/`. Model-free controls live in
`scripts/diagnostics/` (label observability, swarm-scene report, LPI detectability).

### 4. Notebooks

`notebooks/` mirrors the paper section by section and can be run end to end on Colab or
locally:

- `00_Setup_and_Datasets.ipynb` — environment check and dataset generation.
- `01_Communication_Benchmark.ipynb` — `ber_vs_snr`, `classical_receivers`, unified benchmark, BER figures.
- `02_Sensing_Ranging_and_Velocity.ipynb` — ranging benchmark, ambiguity control, velocity track, observability diagnostics.
- `03_Cross_Channel_and_Jamming.ipynb` — cross-channel generalization, jamming and interpretability, frequency agility, LPI.
- `04_Report_Figures_and_Tables.ipynb` — final report, latency bench, all tables and figures.
- `orchestrator.ipynb` — prepares the environment and runs 00–04 in order.

The notebooks detect the repository root automatically, so they work both on Colab (after
cloning into `/content`) and from a local checkout.

## Results

`results/full/` contains the tree produced by the runs reported in the paper: per-scenario
`metrics.csv`, `best_model.keras` checkpoints, `config_used.yaml` provenance, and the derived
tables under `results/full/tables/`. It is the input of `scripts/make_all.sh`, so the figures
and tables can be rebuilt without retraining.

## Reproducibility notes

- Training uses a fixed seed per backbone, `105,000` training symbols and the
  "at least 100 bit errors" testing criterion with online generation.
- Operating-region metrics are the pooled BER over SNR ≥ 5 dB and the minimum SNR at which
  BER ≤ 1e-4 is reached; the per-point 1σ relative uncertainty at the 100-error rule is ≈10%,
  so differences below that level are not claimed.
- Latency numbers are a relative complexity index measured on one reference general-purpose
  CPU (single-threaded TensorFlow, static graph); they are not a measurement on a
  microcontroller.

